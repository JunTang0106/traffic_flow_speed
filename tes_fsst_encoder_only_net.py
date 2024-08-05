import torch
import torch.nn as nn
import os
import sys
import torch

sys.path.append('./lib/')
from lib.utils import load_graphdata_channel_my, get_adjacency_matrix
from FS_ST_encoder import FS_ST_Transformer
from torch.nn import SmoothL1Loss
# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    # 参数和数据文件路径
    filename = 'PEMS08/pems08_r1_d0_w0_2loss_crossattention_1.npz'
    adj_filename = './PEMS08/distance.csv'
    params_path = 'Experiment/PEMS08_embed_size64_2loss'

    print('params_path:', params_path)

    num_of_hours, num_of_days, num_of_weeks = 1, 0, 0

    # 训练参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    learning_rate = 0.01

    # 加载数据
    train_loader, _, _, _, _, _, _, _ = load_graphdata_channel_my(
        filename, num_of_hours, num_of_days, num_of_weeks, device, batch_size)

    # 加载邻接矩阵
    num_of_vertices = 170
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices)
    A = torch.Tensor(adj_mx)

    # 模型参数
    in_channels = 1
    embed_size = 256
    time_num = 288
    num_layers = 3
    T_dim = 12
    output_T_dim = 12
    heads = 8
    cheb_K = 3
    forward_expansion = 4
    dropout = 0

    # 初始化模型
    net = FS_ST_Transformer(
        A,
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
        num_of_vertices,
        dropout
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)
    #criterion = nn.smoothL1Loss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)

    # 训练一轮
    net.train()
    for batch_index, batch_data in enumerate(train_loader):
        encoder_inputs, _, labels = batch_data
        encoder_inputs, labels = encoder_inputs.to(device), labels.to(device)

        flow_labels = labels[:, :, 0, :]
        speed_labels = labels[:, :, -1, :]

        optimizer.zero_grad()

        # 捕获可能的错误
        try:
            flow_outputs, speed_outputs = net(encoder_inputs.permute(0, 2, 1, 3))

            loss = criterion(flow_outputs, flow_labels) + criterion(speed_outputs, speed_labels)
            loss.backward()
            optimizer.step()

            print(f'Batch [{batch_index}/{len(train_loader)}], Loss: {loss.item()}')
            break  # 训练一轮后退出循环

        except RuntimeError as e:
            print(f"Error in batch {batch_index}: {e}")
            raise
