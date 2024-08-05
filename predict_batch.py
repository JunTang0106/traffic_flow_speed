# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:34:11 2021

@author: wzhangcd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
import numpy as np
import pandas as pd

import sys

sys.path.append('./lib/')
from lib.pkl_process import *
from lib.utils import get_adjacency_matrix, load_graphdata_channel_my, compute_val_loss_sttn, masked_mape_np, \
    re_normalization, max_min_normalization, re_max_min_normalization

from time import time
import shutil
import argparse
import configparser
from tensorboardX import SummaryWriter
import os

# from ST_Transformer_new import STTransformer, FinalModel  # STTN model with linear layer to get positional embedding
#from ST_Transformer_new_sinembedding import \
    #FinalModel_sinembedding  # STTN model with sin()/cos() to get positional embedding, the same as "Attention is all your need"
from FS_STTransformer_decoder import FS_ST_Transformer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import os
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

def re_normalization(x, mean, std):
    return (x * std) + mean

def masked_mape_np(y_true, y_pred, null_val=0):
    mask = np.not_equal(y_true, null_val)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype(np.float32), y_true))
    mape = np.nan_to_num(mask * mape)
    return np.mean(mape) * 100

def predict_and_save_results_my(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param global_step: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)  # nb of batch

        # 初始化预测和输入列表用于存储速度和流量数据
        flow_input_list = []
        speed_input_list = []
        flow_prediction_list = []
        speed_prediction_list = []

        def apply_conv_and_permute(tensor, in_channels, out_channels):
            conv_layer = nn.Conv2d(in_channels, out_channels, 1).to("cuda:0")  # 确保卷积层在目标设备上
            tensor = conv_layer(tensor)
            tensor = tensor.permute(0, 3, 2, 1)  # 统一进行维度变换
            return tensor

        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            # 将decoder_input置为0
            decoder_inputs[:, :, :, :] = 0

            # 将数据从 GPU 转到 CPU 并转换为 numpy 数组
            flow_input = encoder_inputs[:, :, 0, :].cpu().numpy()
            speed_input = encoder_inputs[:, :, -1, :].cpu().numpy()

            # 记录输入数据
            flow_input_list.append(flow_input)
            speed_input_list.append(speed_input)

            # 处理 encoder 输入
            encoder_inputs = encoder_inputs.permute(0, 2, 3, 1)  # (32,3,170,12)

            flow_src = encoder_inputs[:, [0], :, :]  # b,1,n,t (32,1,170,12)
            speed_src = encoder_inputs[:, [-1], :, :]  # b,1,n,t (32,1,170,12)

            # 应用到源数据
            flow_src = apply_conv_and_permute(flow_src, 1, 256)
            speed_src = apply_conv_and_permute(speed_src, 1, 256)

            # 处理 decoder 输入
            dec_input = decoder_inputs.permute(0, 2, 3, 1)  # 调整 decoder_inputs 的维度

            # 处理 decoder 的输入
            flow_dec_input = dec_input[:, [0], :, :]  # 注意添加 unsqueeze 以匹配卷积层的输入要求
            speed_dec_input = dec_input[:, [-1], :, :]

            flow_dec_input = apply_conv_and_permute(flow_dec_input, 1, 256)  # 先移除 unsqueeze 的维度再进行处理
            speed_dec_input = apply_conv_and_permute(speed_dec_input, 1, 256)

            # src 和 dec_input 的维度相同 [B,V,T,E][32,170,12,256]
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

                # 将预测结果填充到解码器输入的当前位置
                decoder_inputs_flow[:, :, step, :] = predict_output_flow[:, :, step, :]
                decoder_inputs_speed[:, :, step, :] = predict_output_speed[:, :, step, :]

            # 定义卷积层
            # 输出维度
            # predict_output_flow: [B, V, T, E]
            B, V, T, E = predict_output_flow.shape
            output_T_dim = 12
            conv2 = nn.Conv2d(T, output_T_dim, 1).to("cuda:0")
            # 缩小通道数，降到 1 维。
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

            # 记录预测数据
            flow_prediction_list.append(flow_out.detach().cpu().numpy())
            speed_prediction_list.append(speed_out.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        # 转换列表为 numpy 数组
        flow_input = np.concatenate(flow_input_list, axis=0)
        speed_input = np.concatenate(speed_input_list, axis=0)
        flow_prediction = np.concatenate(flow_prediction_list, axis=0)
        speed_prediction = np.concatenate(speed_prediction_list, axis=0)

        # 测试集时，需要逆归一化
        flow_input = re_normalization(flow_input, _mean[:, :, 0, :], _std[:, :, 0, :])
        speed_input = re_normalization(speed_input, _mean[:, :, -1, :], _std[:, :, -1, :])
        flow_prediction = re_normalization(flow_prediction, _mean[:, :, 0, :], _std[:, :, 0, :])
        speed_prediction = re_normalization(speed_prediction, _mean[:, :, -1, :], _std[:, :, -1, :])

        print('flow_input:', flow_input.shape)
        print('flow_prediction:', flow_prediction.shape)
        print('=================================')
        print('speed_input:', speed_input.shape)
        print('speed_prediction:', speed_prediction.shape)
        print('=================================')
        print('data_target_tensor:', data_target_tensor.shape)

        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, flow_input=flow_input, flow_prediction=flow_prediction,
                 speed_input=speed_input, speed_prediction=speed_prediction, data_target_tensor=data_target_tensor)

        # 初始化误差列表
        flow_mae_list = []
        flow_rmse_list = []
        flow_mape_list = []
        speed_mae_list = []
        speed_rmse_list = []
        speed_mape_list = []

        prediction_length = flow_prediction.shape[2]
        for i in range(prediction_length):
            flow_input_i = flow_input[:, :, i]
            flow_prediction_i = flow_prediction[:, :, i]
            flow_data_target_tensor_i = data_target_tensor[:, :, 0, i]

            speed_input_i = speed_input[:, :, i]
            speed_prediction_i = speed_prediction[:, :, i]
            speed_data_target_tensor_i = data_target_tensor[:, :, -1, i]

            # 计算流量误差
            flow_mae = mean_absolute_error(flow_data_target_tensor_i, flow_prediction_i)
            flow_rmse = mean_squared_error(flow_data_target_tensor_i, flow_prediction_i) ** 0.5
            flow_mape = masked_mape_np(flow_data_target_tensor_i, flow_prediction_i, 0)

            # 计算速度误差
            speed_mae = mean_absolute_error(speed_data_target_tensor_i, speed_prediction_i)
            speed_rmse = mean_squared_error(speed_data_target_tensor_i, speed_prediction_i) ** 0.5
            speed_mape = masked_mape_np(speed_data_target_tensor_i, speed_prediction_i, 0)

            print('current epoch: %s, predict %s points' % (global_step, i))
            print('flow_input:', flow_input_i[0])
            print('flow_predict:', flow_prediction_i[0])
            print('flow_target:', flow_data_target_tensor_i[0])
            print('=================================')
            print('speed_input:', speed_input_i[0])
            print('speed_predict:', speed_prediction_i[0])
            print('speed_target:', speed_data_target_tensor_i[0])
            print('=================================')
            print('flow MAE:', flow_mae)
            print('flow RMSE:', flow_rmse)
            print('flow MAPE:', flow_mape)
            print('speed MAE:', speed_mae)
            print('speed RMSE:', speed_rmse)
            print('speed MAPE:', speed_mape)

            flow_mae_list.append(flow_mae)
            flow_rmse_list.append(flow_rmse)
            flow_mape_list.append(flow_mape)
            speed_mae_list.append(speed_mae)
            speed_rmse_list.append(speed_rmse)
            speed_mape_list.append(speed_mape)

        flow_error_filename = os.path.join(params_path, 'flow_error_epoch_%s_%s' % (global_step, type))
        np.save(flow_error_filename, np.array(flow_mae_list))

        speed_error_filename = os.path.join(params_path, 'speed_error_epoch_%s_%s' % (global_step, type))
        np.save(speed_error_filename, np.array(speed_mae_list))

        # 打印总体误差结果----流量
        flow_mae = mean_absolute_error(data_target_tensor[:, :, 0, :].reshape(-1, 1), flow_prediction.reshape(-1, 1))
        flow_rmse = mean_squared_error(data_target_tensor[:, :, 0, :].reshape(-1, 1),
                                       flow_prediction.reshape(-1, 1)) ** 0.5
        flow_mape = masked_mape_np(data_target_tensor[:, :, 0, :].reshape(-1, 1), flow_prediction.reshape(-1, 1), 0)
        # 打印总体误差结果----速度
        speed_mae = mean_absolute_error(data_target_tensor[:, :, -1, :].reshape(-1, 1), speed_prediction.reshape(-1, 1))
        speed_rmse = mean_squared_error(data_target_tensor[:, :, -1, :].reshape(-1, 1),
                                        speed_prediction.reshape(-1, 1)) ** 0.5
        speed_mape = masked_mape_np(data_target_tensor[:, :, -1, :].reshape(-1, 1), speed_prediction.reshape(-1, 1), 0)

        print('================================')
        print('flow all MAE: %.2f' % (flow_mae))
        print('flow all RMSE: %.2f' % (flow_rmse))
        print('flow all MAPE: %.2f' % (flow_mape))
        print('speed all MAE: %.2f' % (speed_mae))
        print('speed all RMSE: %.2f' % (speed_rmse))
        print('speed all MAPE: %.2f' % (speed_mape))
        flow_excel_list.extend([flow_mae, flow_rmse, flow_mape])
        speed_excel_list.extend([speed_mae, speed_rmse, speed_mape])

        print('flow_excel_list', flow_excel_list)
        print('speed_excel_list', speed_excel_list)

    return flow_prediction, speed_prediction

"""
def predict_and_save_results_my(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param global_step: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)  # nb of batch

        # 初始化预测和输入列表用于存储速度和流量数据
        flow_input_list = []
        speed_input_list = []
        flow_prediction_list = []
        speed_prediction_list = []

        def apply_conv_and_permute(tensor, in_channels, out_channels):
            conv_layer = nn.Conv2d(in_channels, out_channels, 1).to("cuda:0")  # 确保卷积层在目标设备上
            tensor = conv_layer(tensor)
            tensor = tensor.permute(0, 3, 2, 1)  # 统一进行维度变换
            return tensor

        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            # 将decoder_input置为0
            decoder_inputs[:, :, :, :] = 0

            # 将数据从 GPU 转到 CPU 并转换为 numpy 数组
            flow_input = encoder_inputs[:, :, 0, :].cpu().numpy()
            speed_input = encoder_inputs[:, :, -1, :].cpu().numpy()

            # 记录输入数据
            flow_input_list.append(flow_input)
            speed_input_list.append(speed_input)

            # 处理 encoder 输入
            encoder_inputs = encoder_inputs.permute(0, 2, 3, 1)  # (32,3,170,12)

            flow_src = encoder_inputs[:, [0], :, :]  # b,1,n,t (32,1,170,12)
            speed_src = encoder_inputs[:, [-1], :, :]  # b,1,n,t (32,1,170,12)

            # 应用到源数据
            flow_src = apply_conv_and_permute(flow_src, 1, 256)
            speed_src = apply_conv_and_permute(speed_src, 1, 256)

            # 处理 decoder 输入
            dec_input = decoder_inputs.permute(0, 2, 3, 1)  # 调整 decoder_inputs 的维度

            # 处理 decoder 的输入
            flow_dec_input = dec_input[:, [0], :, :]  # 注意添加 unsqueeze 以匹配卷积层的输入要求
            speed_dec_input = dec_input[:, [-1], :, :]

            flow_dec_input = apply_conv_and_permute(flow_dec_input, 1, 256)  # 先移除 unsqueeze 的维度再进行处理
            speed_dec_input = apply_conv_and_permute(speed_dec_input, 1, 256)

            # src 和 dec_input 的维度相同 [B,V,T,E][32,170,12,256]
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
            # 缩小通道数，降到 1 维。
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

            # 记录预测数据
            flow_prediction_list.append(flow_out.detach().cpu().numpy())
            speed_prediction_list.append(speed_out.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        # 转换列表为 numpy 数组
        flow_input = np.concatenate(flow_input_list, axis=0)
        speed_input = np.concatenate(speed_input_list, axis=0)
        flow_prediction = np.concatenate(flow_prediction_list, axis=0)
        speed_prediction = np.concatenate(speed_prediction_list, axis=0)

        # 测试集时，需要逆归一化
        flow_input = re_normalization(flow_input, _mean[:, :, 0, :], _std[:, :, 0, :])
        speed_input = re_normalization(speed_input, _mean[:, :, -1, :], _std[:, :, -1, :])
        flow_prediction = re_normalization(flow_prediction, _mean[:, :, 0, :], _std[:, :, 0, :])
        speed_prediction = re_normalization(speed_prediction, _mean[:, :, -1, :], _std[:, :, -1, :])

        print('flow_input:', flow_input.shape)
        print('flow_prediction:', flow_prediction.shape)
        print('=================================')
        print('speed_input:', speed_input.shape)
        print('speed_prediction:', speed_prediction.shape)
        print('=================================')
        print('data_target_tensor:', data_target_tensor.shape)

        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, flow_input=flow_input, flow_prediction=flow_prediction,
                 speed_input=speed_input, speed_prediction=speed_prediction, data_target_tensor=data_target_tensor)

        # 初始化误差列表
        flow_excel_list = []
        speed_excel_list = []

        prediction_length = flow_prediction.shape[2]
        for i in range(prediction_length):
            flow_input = flow_input[:, :, i]
            flow_predict = flow_prediction[:, :, i]
            flow_target = data_target_tensor[:, :, 0, i]
            # flow_ape = masked_mae(flow_predict, flow_target, 0)

            speed_input = speed_input[:, :, i]
            speed_predict = speed_prediction[:, :, i]
            speed_target = data_target_tensor[:, :, -1, i]
            # speed_ape = masked_mae(speed_predict, speed_target, 0)

            # 计算流量误差
            flow_mae = mean_absolute_error(flow_data_target_tensor_i, flow_prediction_i)
            flow_rmse = mean_squared_error(flow_data_target_tensor_i, flow_prediction_i) ** 0.5
            flow_mape = masked_mape_np(flow_data_target_tensor_i, flow_prediction_i, 0)

            # 计算速度误差
            speed_mae = mean_absolute_error(speed_data_target_tensor_i, speed_prediction_i)
            speed_rmse = mean_squared_error(speed_data_target_tensor_i, speed_prediction_i) ** 0.5
            speed_mape = masked_mape_np(speed_data_target_tensor_i, speed_prediction_i, 0)
            print('current epoch: %s, predict %s points' % (global_step, i))
            print('flow_input:', flow_input[0])
            print('flow_predict:', flow_predict[0])
            print('flow_target:', flow_target[0])
            print('=================================')
            print('speed_input:', speed_input[0])
            print('speed_predict:', speed_predict[0])
            print('speed_target:', speed_target[0])
            print('=================================')
            print('flow error:', flow_ape.item())
            print('speed error:', speed_ape.item())
            flow_excel_list.append(flow_ape.item())
            speed_excel_list.append(speed_ape.item())

        flow_error_filename = os.path.join(params_path, 'flow_error_epoch_%s_%s' % (global_step, type))
        np.save(flow_error_filename, np.array(flow_excel_list))

        speed_error_filename = os.path.join(params_path, 'speed_error_epoch_%s_%s' % (global_step, type))
        np.save(speed_error_filename, np.array(speed_excel_list))

    return flow_prediction, speed_prediction
"""

"""


def predict_and_save_results_my(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param global_step: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)  # nb of batch

        # 初始化预测和输入列表用于存储速度和流量数据
        speed_prediction = []
        speed_input = []
        flow_prediction = []
        flow_input = []

        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            # 将decoder_input置为0
            decoder_inputs[:, :, 0, :] = 0
            # print('encoder_inputs', encoder_inputs.shape)
            # print('decoder_inputs', decoder_inputs.shape)

            flow_input.append(encoder_inputs[:, :, 0, :].cpu().numpy())
            speed_input.append(encoder_inputs[:, :, -1, :].cpu().numpy())
            flow_input.append(decoder_inputs[:, :, 0, :].cpu().numpy())
            speed_input.append(decoder_inputs[:, :, -1, :].cpu().numpy())

            flow_outputs, speed_outputs = net.Transformer.encode(flow_input, speed_input)
            # 对解码器逐步解码
            for i in range(flow_outputs.shape[1]):
                flow_outputs[:, i, :] = net.Transformer.decode(flow_outputs[:, i, :], speed_outputs[:, i, :])
                decoder_inputs.permute(0, 2, 1, 3)

            flow_prediction.append(flow_outputs.detach().cpu().numpy())
            speed_prediction.append(speed_outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        flow_input = np.concatenate(flow_input, 0)
        speed_input = np.concatenate(speed_input, 0)

        # 测试集时，需要逆归一化
        flow_input = re_normalization(flow_input, _mean[:, :, 0, :], _std[:, :, 0, :])
        speed_input = re_normalization(speed_input, _mean[:, :, -1, :], _std[:, :, -1, :])

        flow_prediction = np.concatenate(flow_prediction, 0)
        speed_prediction = np.concatenate(speed_prediction, 0)

        print('flow_input:', flow_input.shape)
        print('flow_prediction:', flow_prediction.shape)
        print('=================================')
        print('speed_input:', speed_input.shape)
        print('speed_prediction:', speed_prediction.shape)
        print('=================================')
        print('data_target_tensor:', data_target_tensor.shape)

        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, flow_input=flow_input, flow_prediction=flow_prediction,
                 speed_input=speed_input, speed_prediction=speed_prediction , data_target_tensor=data_target_tensor)

        # 初始化误差列表
        flow_excel_list = []
        speed_excel_list = []

        prediction_length = flow_prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == flow_prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))

            flow_data_target_tensor_i = data_target_tensor[:, :, 0, i]
            speed_data_target_tensor_i = data_target_tensor[:, :, -1, i]

            flow_prediction_i = flow_prediction[:, :, i]
            speed_prediction_i = speed_prediction[:, :, i]

            # 计算流量误差
            flow_mae = mean_absolute_error(flow_data_target_tensor_i, flow_prediction_i)
            flow_rmse = mean_squared_error(flow_data_target_tensor_i, flow_prediction_i) ** 0.5
            flow_mape = masked_mape_np(flow_data_target_tensor_i, flow_prediction_i, 0)

            # 计算速度误差
            speed_mae = mean_absolute_error(speed_data_target_tensor_i, speed_prediction_i)
            speed_rmse = mean_squared_error(speed_data_target_tensor_i, speed_prediction_i) ** 0.5
            speed_mape = masked_mape_np(speed_data_target_tensor_i, speed_prediction_i, 0)

            print('flow_MAE: %.2f' % (flow_mae))
            print('flow_RMSE: %.2f' % (flow_rmse))
            print('flow_MAPE: %.2f' % (flow_mape))
            print('speed_MAE: %.2f' % (speed_mae))
            print('speed_RMSE: %.2f' % (speed_rmse))
            print('speed_MAPE: %.2f' % (speed_mape))

            flow_excel_list.extend([flow_mae, flow_rmse, flow_mape])
            speed_excel_list.extend([speed_mae, speed_rmse, speed_mape])

        # 打印总体误差结果----流量
        flow_mae = mean_absolute_error(data_target_tensor[:, :, 0, :].reshape(-1, 1), flow_prediction.reshape(-1, 1))
        flow_rmse = mean_squared_error(data_target_tensor[:, :, 0, :].reshape(-1, 1), flow_prediction.reshape(-1, 1)) ** 0.5
        flow_mape = masked_mape_np(data_target_tensor[:, :, 0, :].reshape(-1, 1), flow_prediction.reshape(-1, 1), 0)
        # 打印总体误差结果----速度
        speed_mae = mean_absolute_error(data_target_tensor[:, :, -1, :].reshape(-1, 1), speed_prediction.reshape(-1, 1))
        speed_rmse = mean_squared_error(data_target_tensor[:, :, -1, :].reshape(-1, 1), speed_prediction.reshape(-1, 1)) ** 0.5
        speed_mape = masked_mape_np(data_target_tensor[:, :, -1, :].reshape(-1, 1), speed_prediction.reshape(-1, 1), 0)

        print('================================')
        print('flow all MAE: %.2f' % (flow_mae))
        print('flow all RMSE: %.2f' % (flow_rmse))
        print('flow all MAPE: %.2f' % (flow_mape))
        print('speed all MAE: %.2f' % (speed_mae))
        print('speed all RMSE: %.2f' % (speed_rmse))
        print('speed all MAPE: %.2f' % (speed_mape))
        flow_excel_list.extend([flow_mae, flow_rmse, flow_mape])
        speed_excel_list.extend([speed_mae, speed_rmse, speed_mape])

        print('flow_excel_list', flow_excel_list)
        print('speed_excel_list', speed_excel_list)
"""

def predict_main(params_filename, global_step, data_loader, data_target_tensor, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_my(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)


if __name__ == '__main__':
    # Best Epoch during Training
    best_epoch = 14

    # Same Setting as train_my.py

    # params_path = 'Experiment/PEMS04_embed_size64_2loss'  # Path for saving network parameters
    # print('params_path:', params_path)
    # filename = 'PEMS04/pems04_r1_d0_w0_2loss_astcgn.npz'  # Data generated by prepareData.py
    # filename = 'PEMS04/pems04_r1_d0_w0_2loss_crossattention_1.npz'  # Data generated by prepareData.py
    # adj_filename = './PEMS04/distance.csv'  # path for adjacency_matrix

    params_path = 'Experiment/PEMS08_embed_size64_2loss'  # Path for saving network parameters
    print('params_path:', params_path)
    filename = 'PEMS08/pems08_r1_d0_w0_2loss_crossattention_1.npz'
    # # filename = 'PEMS08/pems08_r1_d0_w0_2loss_astcgn_1.npz'  # Data generated by prepareData.py
    adj_filename = './PEMS08/distance.csv'  # path for adjacency_matrix

    num_of_hours, num_of_days, num_of_weeks = 1, 0, 0  # The same setting as prepareData.py

    # Training Hyparameter
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = DEVICE
    # batch_size = 16  # pems04
    batch_size = 16  # pems08
    learning_rate = 0.01
    epochs = 80
    # num_of_vertices = 307  # pems04
    num_of_vertices = 170  # pems08

    # Generate Data Loader
    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel_my(
        filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)

    ### Adjacency Matrix Import
    # 邻接矩阵
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename=None)
    # adj_mx = pd.read_csv('./PEMSD7/W_25.csv', header = None)
    # adj_mx = import_pkl('/home/wzhangcd@HKUST/Commonpkg/adj_mx_tran_89.pkl')
    # adj_mx = np.array(adj_mx)
    A = adj_mx
    A = torch.Tensor(A)

    ### Training Hyparameter
    in_channels = 1  # Channels of input
    embed_size = 256  # Dimension of hidden embedding features
    time_num = 288
    num_layers = 3  # Number of ST Block
    T_dim = 12  # Input length, should be the same as prepareData.py
    output_T_dim = 12  # Output Expected length
    heads = 8  # Number of Heads in MultiHeadAttention
    cheb_K = 3  # Order for Chebyshev Polynomials (Eq 2)
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0


    # 带 flow_speed cross-attention 模块的 sttn
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
        dropout)

    net.to(device)

    predict_main(params_path, best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')
