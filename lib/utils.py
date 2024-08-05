import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs
import torch.nn as nn

def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


"""


def load_graphdata_channel_my(filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param filename: 文件名
    :param num_of_hours: 小时数
    :param num_of_days: 天数
    :param num_of_weeks: 周数
    :param DEVICE: 设备
    :param batch_size: 批大小
    :param shuffle: 是否打乱数据
    :param percent: 数据比例
    :return: 训练、验证和测试的DataLoader
    '''

    print('load file:', filename)
    file_data = np.load(filename)

    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_target = file_data['train_target']  # (10181, 307, 3, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length * percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]

    val_x = file_data['val_x']
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_target = file_data['test_target']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)


    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)

    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)

    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)

    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor,  test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    print('train:', train_x_tensor.size(),  train_target_tensor.size())
    print('val:', val_x_tensor.size(),  val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min

"""

def load_graphdata_channel_my(filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''
    
    
    # file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    # dirpath = os.path.dirname(graph_signal_matrix_filename)

    # filename = os.path.join(dirpath,
                            #file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')
    
    filename = filename
    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)

    # ---------------
    #train_x = train_x[:, :, 0:1, :]
    train_x = train_x[:, :, :, :]  # 保留所有
    train_target = file_data['train_target']  # (10181, 307, 3, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    # ---------------
    # val_x = val_x[:, :, 0:1, :]
    val_x = val_x[:, :, :, :]  # 保留
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    # ---------------
    # test_x = test_x[:, :, 0:1, :] # 保留
    test_x = test_x[:, :, :, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值

    #train_target_norm = max_min_normalization(train_target, _max[:, :, :, :], _min[:, :, :, :])
    #test_target_norm = max_min_normalization(test_target, _max[:, :, :, :], _min[:, :, :, :])
    #val_target_norm = max_min_normalization(val_target, _max[:, :, :, :], _min[:, :, :, :])

    #train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    #test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    #val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

    #  ------- train_loader -------
    #train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = train_x[:, :, :, -1:]  # (B, N, 3(F), 1(T)),最后已知traffic flow作为decoder 的初始输入

    #train_decoder_input_start_flow = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    #train_decoder_input_start_speed = train_x[:, :, 2:3, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic speed作为decoder 的初始输入
    #train_decoder_input_start_flow = np.squeeze(train_decoder_input_start_flow, 2)  # (B,N,T(1))
    #train_decoder_input_start_speed = np.squeeze(train_decoder_input_start_speed, 2)  # (B,N,T(1))

    train_decoder_input = np.concatenate((train_decoder_input_start, train_target[:, :, :, :-1]), axis=3)  # (B, N, 3, T)

    #train_decoder_input_flow = np.concatenate((train_decoder_input_start_flow, train_target[:, :, :-1]),axis=2)  # (B, N, T)
    #train_decoder_input_speed = np.concatenate((train_decoder_input_start_speed, train_target[:, :, :-1]),axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N,3, T)
    #train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N,3 T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N,3 T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    #val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = val_x[:, :, :, -1:]  # (B, N, 3(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    #val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    #val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target[:, :, :, :-1]), axis=3)  # (B, N,3 ，T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N,3， T)
    #val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, 3 T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, 3 T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    #test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = test_x[:, :, :, -1:]  # (B, N, 3(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    #test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    #test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target[:, :, :, :-1]), axis=3)  # (B, N, 3 T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N,3 T)
    #test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N,3 T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N,3 T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min

# 应该改为解码器一步一步输入
def compute_val_loss_sttn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :param limit: int, optional limit for number of batches
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():
        val_loader_length = len(val_loader)  # number of batches

        tmp = []  # record all batch losses

        for batch_index, batch_data in enumerate(val_loader):
            # encoder_inputs, labels: b,n,c,t
            encoder_inputs, decoder_inputs, labels = batch_data
            # 推理阶段，将decoder_inputs置为0 ，将decoder_inputs信息掩盖置为0
            decoder_inputs[:, :, :, :] = 0
            flow_labels = labels[:, :, 0, :]  # b,n,t
            speed_labels = labels[:, :, -1, :]

            # 定义一个函数来应用卷积并调整维度
            def apply_conv_and_permute(tensor, in_channels, out_channels):
                conv_layer = nn.Conv2d(in_channels, out_channels, 1).to("cuda:0")  # 确保卷积层在目标设备上
                tensor = conv_layer(tensor)
                tensor = tensor.permute(0, 3, 2, 1)  # 统一进行维度变换
                return tensor

            # encoder输入处理
            encoder_inputs = encoder_inputs.permute(0, 2, 3, 1)  # (32,3,170,12)

            flow_src = encoder_inputs[:, [0], :, :]  # b,1,n,t (32,1,170,12)
            speed_src = encoder_inputs[:, [-1], :, :]  # b,1,n,t (32,1,170,12)

            # 应用到源数据
            flow_src = apply_conv_and_permute(flow_src, 1, 256)
            speed_src = apply_conv_and_permute(speed_src, 1, 256)

            # 解码器输入处理
            dec_input = decoder_inputs.permute(0, 2, 3, 1)  # 调整decoder_inputs的维度

            # 处理decoder的输入
            flow_dec_input = dec_input[:, [0], :, :]  # 注意添加unsqueeze以匹配卷积层的输入要求
            speed_dec_input = dec_input[:, [-1], :, :]

            flow_dec_input = apply_conv_and_permute(flow_dec_input, 1, 256)  # 先移除unsqueeze的维度再进行处理
            speed_dec_input = apply_conv_and_permute(speed_dec_input, 1, 256)

            #  src和dec_input的维度相同[B,V,T,E][32,170,12,256]
            # Encoder step
            flow_enc_src, speed_enc_src = net.Transformer.encode(flow_src, speed_src)  # (32,170,12,64)
            # 编码器输出 [B,V,T,E][32,170,12,256]

            # Decoder step with step-by-step decoding
            predict_length = labels.shape[2]  # T

            # 初始化解码器输入，保持形状为 [B, V, T, E]
            decoder_inputs_flow = torch.zeros_like(flow_dec_input)
            decoder_inputs_speed = torch.zeros_like(speed_dec_input)

            # 初始化掩码，只允许看到当前和之前的时间步
            for step in range(predict_length):
                # 取当前时间步的输入，保持形状为 [B, V, T, E]
                current_decoder_inputs_flow = decoder_inputs_flow.clone()
                current_decoder_inputs_speed = decoder_inputs_speed.clone()

                # 解码，使用完整的解码器输入和掩码
                predict_output_flow, predict_output_speed = net.Transformer.decode(
                    flow_enc_src, speed_enc_src, current_decoder_inputs_flow, current_decoder_inputs_speed
                )

                # 将预测结果滑动到解码器输入的下一步
                if step + 1 < predict_length:
                    decoder_inputs_flow[:, :, step + 1, :] = predict_output_flow[:, :, step, :]
                    decoder_inputs_speed[:, :, step + 1, :] = predict_output_speed[:, :, step, :]

            # 定义卷积层
            # 输出维度
            # predict_output_flow: [B, V, T, E]
            B, V, T, E = predict_output_flow.shape
            output_T_dim = 12
            conv2 = nn.Conv2d(T, output_T_dim, 1).to("cuda:0")
            # 缩小通道数，降到1维。
            conv3 = nn.Conv2d(E, 1, 1).to("cuda:0")
            relu = nn.ReLU()

            # 对 predict_output_flow 进行操作
            flow_out = predict_output_flow.permute(0, 2, 1, 3)  # [B, T, V, E]
            flow_out = relu(conv2(flow_out))  # 等号左边 out shape: [B, output_T_dim, N, C] (32,12,170,64)
            flow_out = flow_out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim] (32,64,170,12)
            flow_out = conv3(flow_out)  # 等号左边 out shape: [B, 1, N, output_T_dim] (32,1,170,12)
            flow_out = flow_out.squeeze(1)  # (32,170,12)

            # 对 predict_output_speed 进行操作
            speed_out = predict_output_speed.permute(0, 2, 1, 3)  # [B, T, V, E]
            speed_out = relu(conv2(speed_out))  # 等号左边 out shape: [B, output_T_dim, N, C] (32,12,170,64)
            speed_out = speed_out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim] (32,64,170,12)
            speed_out = conv3(speed_out)  # 等号左边 out shape: [B, 1, N, output_T_dim] (32,1,170,12)
            speed_out = speed_out.squeeze(1)  # (32,170,12)

            # Calculate loss
            loss = criterion(flow_out, flow_labels) + criterion(speed_out, speed_labels)
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

def compute_val_loss_fsst_encode_only(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    
    :param net: 
    :param val_loader: 
    :param criterion: 
    :param sw: 
    :param epoch: 
    :param limit: 
    :return: 
    ''''''
    for rnn, compute mean loss on validation set
    :param
    net: model
    :param
    val_loader: torch.utils.data.utils.DataLoader
    :param
    criterion: torch.nn.MSELoss
    :param
    sw: tensorboardX.SummaryWriter
    :param
    global_step: int, current
    global_step
    :param
    limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            # encoder_inputs, labels: b,n,c,t
            encoder_inputs, _, labels = batch_data

            flow_labels = labels[:, :, 0, :]  # b,n,t
            speed_labels = labels[:, :, -1, :]

            #  encoder_inputs.permute(0, 2, 1, 3)--> b,c,n,t
            flow_outputs, speed_outputs = net(encoder_inputs.permute(0, 2, 1, 3))

            loss = criterion(flow_outputs, flow_labels) + criterion(speed_outputs, speed_labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss
