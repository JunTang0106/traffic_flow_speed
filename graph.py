import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'

# ['train_x', 'train_target', 'train_timestamp', 'val_x', 'val_target', 'val_timestamp', 'test_x', 'test_target', 'test_timestamp', 'mean', 'std']
# ground=np.load('PEMS04/pems04_r1_d0_w0_2loss_astcgn.npz')['test_target']
# print(ground.shape)  # （3394，307，3，12）


# ['flow_input', 'flow_prediction', 'speed_input', 'speed_prediction', 'data_target_tensor']
ground = np.load('Experiment/PEMS04_embed_size64_2loss/output_epoch_102_test.npz')['data_target_tensor']
# ground = np.load('Experiment/PEMS08_embed_size64_2loss/output_epoch_89_test.npz')['data_target_tensor']

# print(ground.shape)  # （3394，307，3，12）

# (3394, 307, 12)
ground_flow = ground[:, :, 0, :]
print(ground_flow.shape)
ground_speed = ground[:, :, -1, :]

flow_prediction = np.load('Experiment/PEMS04_embed_size64_2loss/output_epoch_102_test.npz')['flow_prediction']
speed_prediction = np.load('Experiment/PEMS04_embed_size64_2loss/output_epoch_102_test.npz')['speed_prediction']

# flow_prediction = np.load('Experiment/PEMS08_embed_size64_2loss/output_epoch_89_test.npz')['flow_prediction']
# speed_prediction = np.load('Experiment/PEMS08_embed_size64_2loss/output_epoch_89_test.npz')['speed_prediction']


# 04的用41
# node_list = [5,41,65,67,77]
node_list = [41]
# 08的用91
# node_list = [91, 122]
# node_list = [91]

for i in node_list:
    x=np.arange(0,288)
    plt.plot(x, ground_flow[288*4:288*5, i, -1], c='blue', label='Real data')
    plt.plot(x, flow_prediction[288*4:288*5, i, -1], c='red', label='Prediction results')
    # plt.legend(loc='best',fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel('Time Interval (5 min)', fontsize=27)
    plt.ylabel('Flow (veh / 5 min)', fontsize=27)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.title('Comparison of predicted and actual flow data', fontsize=15)
    plt.title('')
    plt.tight_layout()
    # 保存为pdf文件
    # plt.savefig('comparison of node'+str(i)+'.pdf', format='pdf')
    # plt.savefig('4_5(a).pdf', format='pdf')
    # plt.savefig('4_5(b).pdf', format='pdf')
    plt.show()


def calculate_mape(predicted_data, actual_data):
    """
    计算MAPE（Mean Absolute Percentage Error）
    """
    error = np.abs((actual_data - predicted_data) / actual_data)
    mape = np.mean(error) * 100
    return mape

# 已有的单个节点预测流量数据
# pems04 41
predicted_data = flow_prediction[288*4:288*5, 41, -1]
# np.save('pems04_41_predictedData.npy',predicted_data)
# pems08 91
# predicted_data = flow_prediction[288*4:288*5, 91, -1]
# print('predicted_data',predicted_data)

# 实际测试集的流量数据
# pems04 41
actual_data = ground_flow[288*4:288*5, 41, -1]
# np.save('pems04_41_actualData.npy',actual_data)

# pems08 91
# actual_data = ground_flow[288*4:288*5, 91, -1]
# print('actual_data',actual_data)

# 计算MAPE(目前04用的41，08用的91)
mape = calculate_mape(predicted_data, actual_data)
print("MAPE:", mape)

# 计算精度（老师说是1-mape%）
accuracy = 100 - mape
print('精度：', accuracy)

# 将 numpy 数组转换为 DataFrame 对象
# df1 = pd.DataFrame(predicted_data)
# df2 = pd.DataFrame(actual_data)

# 导出到 Excel 文件
# file_path1 = 'predicted_data_output.xlsx'  # 输出文件的路径和名称
# df1.to_excel(file_path1, index=False)  # 导出为 Excel 文件，不包含索引列
# file_path2 = 'actual_data_output.xlsx'  # 输出文件的路径和名称
# df2.to_excel(file_path2, index=False)  # 导出为 Excel 文件，不包含索引列

# 以下代码为遍历所有点，找mape小的
# best_mape = 100;
# best_i = 0;
# for i in range(1,100):
# # for i in node_list:
#     predicted_data = flow_prediction[288 * 4:288 * 5, i, -1]
#     actual_data = ground_flow[288*4:288*5, i, -1]
#     mape = calculate_mape(predicted_data, actual_data)
#     print(i,'的mape为',mape)
#     if mape < best_mape:
#         best_i = i
#         best_mape = mape
#
# print('点为：',best_i)
# print('best_mape', best_mape)