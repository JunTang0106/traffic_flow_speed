# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from GCN_models import GCN
# from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

class FlowSpeed(nn.Module):
    def __init__(self, num_of_nodes):
        super(FlowSpeed, self).__init__()
        self.Kj = nn.Parameter(torch.ones(size=(num_of_nodes, 1)))
        self.Vf = nn.Parameter(torch.ones(size=(num_of_nodes, 1)))
        nn.init.xavier_normal_(self.Kj.data)
        nn.init.xavier_normal_(self.Vf.data)

    def forward(self, speed):
        flow = self.Kj * (speed - torch.pow(speed, 2) / (self.Vf + 1e-5))
        return flow



# 更改的crossattention图连接方式
class TemporalScaledDotProductforCrossAttention(nn.Module):
    def __init__(self):
        super(TemporalScaledDotProductforCrossAttention, self).__init__()
        self.flow_speed = FlowSpeed(12)

    def forward(self, Qf, Kf, Vf, Qs, Ks, Vs, mask=None):
        B, n_heads, len1, len2, d_k = Qf.shape

        # Transform Qs to Qf_from_speed using FlowSpeed model
        Qf_from_speed = self.flow_speed(Qs)

        # Compute scores
        scores_flow_flow = torch.matmul(Qf, Kf.transpose(-1, -2)) / np.sqrt(d_k)
        scores_flow_speed = torch.matmul(Kf, Qf_from_speed.transpose(-1, -2)) / np.sqrt(d_k)
        scores_speed_flow = torch.matmul(Ks, Qf.transpose(-1, -2)) / np.sqrt(d_k)
        scores_speed_speed = torch.matmul(Qs, Ks.transpose(-1, -2)) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores_flow_flow = scores_flow_flow.masked_fill(mask == 0, -1e9)
            scores_flow_speed = scores_flow_speed.masked_fill(mask == 0, -1e9)
            scores_speed_flow = scores_speed_flow.masked_fill(mask == 0, -1e9)
            scores_speed_speed = scores_speed_speed.masked_fill(mask == 0, -1e9)

        # Apply softmax to obtain attention weights
        z_flow_flow = torch.nn.functional.softmax(scores_flow_flow, dim=-1)
        z_flow_speed = torch.nn.functional.softmax(scores_flow_speed, dim=-1)
        z_speed_flow = torch.nn.functional.softmax(scores_speed_flow, dim=-1)
        z_speed_speed = torch.nn.functional.softmax(scores_speed_speed, dim=-1)

        # Compute context vectors
        context_flow_flow = torch.matmul(z_flow_flow, Vf)
        context_flow_speed = torch.matmul(z_flow_speed, Vf)
        context_speed_flow = torch.matmul(z_speed_flow, Vs)
        context_speed_speed = torch.matmul(z_speed_speed, Vs)

        return context_flow_flow, context_flow_speed, context_speed_flow, context_speed_speed
class TemporalMultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TemporalMultiHeadCrossAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 用Linear来做投影矩阵

        self.W_fV = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_fK = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_fQ = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sV = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sK = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sQ = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)

        self.TemporalScaledDotProductforCrossAttention = TemporalScaledDotProductforCrossAttention()
        # self.ScaledDotProductforCrossAttention = ScaledDotProductforCrossAttention()

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, flow_input_Q, flow_input_K, flow_input_V, speed_input_Q, speed_input_K, speed_input_V, mask=None):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        Bf, Nf, Tf, Cf = flow_input_Q.shape
        Bs, Ns, Ts, Cs = speed_input_Q.shape

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Qf = self.W_fQ(flow_input_Q).view(Bf, Nf, Tf, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        Kf = self.W_fK(flow_input_K).view(Bf, Nf, Tf, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        Vf = self.W_fV(flow_input_V).view(Bf, Nf, Tf, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]
        Qs = self.W_sQ(speed_input_Q).view(Bs, Ns, Ts, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        Ks = self.W_sK(speed_input_K).view(Bs, Ns, Ts, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        Vs = self.W_sV(speed_input_V).view(Bs, Ns, Ts, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context_flow_flow, context_flow_speed, context_speed_flow, context_speed_speed = self.TemporalScaledDotProductforCrossAttention(Qf, Kf, Vf, Qs, Ks, Vs, mask)  # [B, h, N, T, d_k]

        context_flow_flow = context_flow_flow.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context_flow_flow = context_flow_flow.reshape(Bf, Nf, Tf, self.heads * self.head_dim)  # [B, N, T, C]
        output_flow_flow = self.fc_out(context_flow_flow)  # [batch_size, len_q, d_model]

        context_flow_speed = context_flow_speed.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context_flow_speed = context_flow_speed.reshape(Bf, Nf, Tf, self.heads * self.head_dim)  # [B, N, T, C]
        output_flow_speed = self.fc_out(context_flow_speed)  # [batch_size, len_q, d_model]

        context_speed_flow = context_speed_flow.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context_speed_flow = context_speed_flow.reshape(Bs, Ns, Ts, self.heads * self.head_dim)  # [B, N, T, C]
        output_speed_flow = self.fc_out(context_speed_flow)  # [batch_size, len_q, d_model]

        context_speed_speed = context_speed_speed.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context_speed_speed = context_speed_speed.reshape(Bs, Ns, Ts, self.heads * self.head_dim)  # [B, N, T, C]
        output_speed_speed = self.fc_out(context_speed_speed)  # [batch_size, len_q, d_model]

        return output_flow_flow, output_flow_speed, output_speed_flow, output_speed_speed
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        # Temporal embedding One hot
        self.time_num = time_num
        # self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        #         self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding

        self.t_cross_attention = TemporalMultiHeadCrossAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 门控机制融合方法1，直接线性层
        self.linear = nn.Linear(2, 1)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)
    def forward(self, flow_value, flow_key, flow_query, speed_value, speed_key, speed_query, mask=None):  # 都是 (32,170,12,64)
        # 空间位置嵌入

        Bf, Nf, Tf, Cf = flow_query.shape  # Bf:32, Cf:64, Nf:170 , Tf:12
        Bs, Ns, Ts, Cs = speed_query.shape  # 同上 Bs:, Cs:, Ns: , Ts:

        flow_D_T = get_sinusoid_encoding_table(Tf, Cf).to('cuda:0')
        flow_D_T = flow_D_T.expand(Bf, Nf, Tf, Cf)  # (32,170,12,64)

        speed_D_T = get_sinusoid_encoding_table(Ts, Cs).to('cuda:0')
        speed_D_T = speed_D_T.expand(Bs, Ns, Ts, Cs)  # (32,170,12,64)

        # temporal embedding加到query。 原论文采用concatenated
        flow_query = flow_query + flow_D_T
        speed_query = speed_query + speed_D_T

        attention_flow_flow, attention_flow_speed, attention_speed_flow, attention_speed_speed = self.t_cross_attention(flow_query, flow_query, flow_query, speed_query, speed_query, speed_query, mask)  # (B, N, T, C) (32,170,12,64)

        # 门控机制融合 attention_flow_flow 和 attention_flow_speed 得到 attention_flow
        # 方法1 直接concat然后线性变换，没写好
        attention_flow = torch.stack((attention_flow_flow, attention_flow_speed), dim=-1)  # output: b,n,t,c,2 (32,170,12,64,2)
        attention_flow = self.linear(attention_flow)  # 应用全连接层 (32,170,12,64,1)
        # attention_flow = self.feed_forward1(attention_flow)  # 应用全连接层 (32,170,12,64,1)
        attention_flow = attention_flow.squeeze(4)  # 去除新增的维度 (32,170,12,64)

        attention_speed = torch.stack((attention_speed_speed, attention_speed_flow), dim=-1)  # output: b,n,t,c,2 (32,170,12,64,2)
        attention_speed = self.linear(attention_speed)  # 应用全连接层 (32,170,12,64,1)
        # attention_speed = self.feed_forward1(attention_speed)  # 应用全连接层 (32,170,12,64,1)
        attention_speed = attention_speed.squeeze(4)  # 去除新增的维度 (32,170,12,64)


        # Add skip connection, run through normalization and finally dropout
        flow_x = self.dropout(self.norm1(attention_flow + flow_query))  # (32,170,12,64)
        flow_forward = self.feed_forward(flow_x)  # (32,170,12,64)
        flow_out = self.dropout(self.norm2(flow_forward + flow_x))  # (32,170,12,64)

        speed_x = self.dropout(self.norm1(attention_speed + speed_query))
        speed_forward = self.feed_forward(speed_x)
        speed_out = self.dropout(self.norm2(speed_forward + speed_x))

        return flow_out, speed_out
class SpatialScaledDotProductforCrossAttention(nn.Module):
    def __init__(self, num_of_vertices):
        super(SpatialScaledDotProductforCrossAttention, self).__init__()

        self.flow_speed = FlowSpeed(num_of_vertices)
        # self.linear = nn.Linear(32, 32)

    def forward(self, Qf, Kf, Vf, Qs, Ks, Vs, mask = None):  # (32,2,12,170,32)
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Qf.shape   # b 32,nhead 2,len1 12,len2 170,d_k 32

        # 把 Qs 经过 greenshields 模型变成 Qf_from_speed，用于与Kf相乘，原有的Ks继续与Qs相乘
        # 在这之前试试用一个 linear 层
        # Ks = self.linear(Ks)  # (32,2,12,170,32)
        Qf_from_speed = self.flow_speed(Qs)  # (32,2,12,170,32)

        # Qf 和 Kf 计算出一个相似度然后经过softmax 乘到 Vf
        scores_flow_flow = torch.matmul(Qf, Kf.transpose(-1, -2)) / np.sqrt(d_k)
        z_flow_flow = nn.Softmax(dim=-1)(scores_flow_flow)
        context_flow_flow = torch.matmul(z_flow_flow, Vf)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]

        # Kf 和 Qf_from_speed 计算出一个相似度然后经过softmax 乘到 Vf
        scores_flow_speed = torch.matmul(Kf, Qf_from_speed.transpose(-1, -2)) / np.sqrt(d_k)
        z_flow_speed = nn.Softmax(dim=-1)(scores_flow_speed)
        context_flow_speed = torch.matmul(z_flow_speed, Vf)

        # Ks 和 Qf 计算出一个相似度然后经过softmax 乘到 Vs
        scores_speed_flow = torch.matmul(Ks, Qf.transpose(-1, -2)) / np.sqrt(d_k)
        z_speed_flow = nn.Softmax(dim=-1)(scores_speed_flow)
        context_speed_flow = torch.matmul(z_speed_flow, Vs)

        # Qs 和 Ks 计算出一个相似度然后经过softmax 乘到 Vs
        scores_speed_speed = torch.matmul(Qs, Ks.transpose(-1, -2)) / np.sqrt(d_k)
        z_speed_speed = nn.Softmax(dim=-1)(scores_speed_speed)
        context_speed_speed = torch.matmul(z_speed_speed, Vs)


        return context_flow_flow, context_flow_speed, context_speed_flow, context_speed_speed
class SpatialMultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_size, heads, num_of_vertices):
        super(SpatialMultiHeadCrossAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_fV = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_fK = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_fQ = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sV = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sK = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sQ = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)

        self.SpatialScaledDotProductforCrossAttention = SpatialScaledDotProductforCrossAttention(num_of_vertices)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, flow_input_Q, flow_input_K, flow_input_V, speed_input_Q, speed_input_K, speed_input_V, mask=None):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        Bf, Nf, Tf, Cf = flow_input_Q.shape
        Bs, Ns, Ts, Cs = speed_input_Q.shape
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Qf = self.W_fQ(flow_input_Q).view(Bf, Nf, Tf, self.heads, self.head_dim).transpose(1, 3)  # Qf: [B, h, T, N, d_k] (32,2,12,170,32)
        Kf = self.W_fK(flow_input_K).view(Bf, Nf, Tf, self.heads, self.head_dim).transpose(1, 3)  # Kf: [B, h, T, N, d_k]
        Vf = self.W_fV(flow_input_V).view(Bf, Nf, Tf, self.heads, self.head_dim).transpose(1, 3)  # Vf: [B, h, T, N, d_k]
        Qs = self.W_sQ(speed_input_Q).view(Bs, Ns, Ts, self.heads, self.head_dim).transpose(1, 3)  # Qs: [B, h, T, N, d_k]
        Ks = self.W_sK(speed_input_K).view(Bs, Ns, Ts, self.heads, self.head_dim).transpose(1, 3)  # Ks: [B, h, T, N, d_k]
        Vs = self.W_sV(speed_input_V).view(Bs, Ns, Ts, self.heads, self.head_dim).transpose(1, 3)  # Vs: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context_flow_flow, context_flow_speed, context_speed_flow, context_speed_speed = self.SpatialScaledDotProductforCrossAttention(Qf, Kf, Vf, Qs, Ks, Vs, mask)  # [B, h, T, N, d_k]

        context_flow_flow = context_flow_flow.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context_flow_flow = context_flow_flow.reshape(Bf, Nf, Tf, self.heads * self.head_dim)  # [B, N, T, C]
        output_flow_flow = self.fc_out(context_flow_flow)  # [batch_size, len_q, d_model]

        context_flow_speed = context_flow_speed.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context_flow_speed = context_flow_speed.reshape(Bf, Nf, Tf, self.heads * self.head_dim)  # [B, N, T, C]
        output_flow_speed = self.fc_out(context_flow_speed)  # [batch_size, len_q, d_model]

        context_speed_flow = context_speed_flow.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context_speed_flow = context_speed_flow.reshape(Bs, Ns, Ts, self.heads * self.head_dim)  # [B, N, T, C]
        output_speed_flow = self.fc_out(context_speed_flow)  # [batch_size, len_q, d_model]

        context_speed_speed = context_speed_speed.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context_speed_speed = context_speed_speed.reshape(Bs, Ns, Ts, self.heads * self.head_dim)  # [B, N, T, C]
        output_speed_speed = self.fc_out(context_speed_speed)  # [batch_size, len_q, d_model]

        return output_flow_flow, output_flow_speed, output_speed_flow, output_speed_speed




class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, cheb_K, dropout, forward_expansion, num_of_vertices):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = adj.to('cuda:0')

        self.s_cross_attention = SpatialMultiHeadCrossAttention(embed_size, heads, num_of_vertices)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 门控机制融合方法1，直接线性层
        self.linear = nn.Linear(2, 1)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # 调用GCN
        self.gcn = GCN(embed_size, embed_size * 2, embed_size, adj, cheb_K, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, flow_value, flow_key, flow_query, speed_value, speed_key, speed_query, mask=None):  # 以pems08为例，均为 (32,170,12,64)
        # 空间位置嵌入

        Bf, Nf, Tf, Cf = flow_query.shape  # Bf:32, Nf: 170, Tf: 12, Cf:64
        Bs, Ns, Ts, Cs = speed_query.shape  # Bs:32, Ns: 170, Ts: 12, Cs:64

        flow_D_S = get_sinusoid_encoding_table(Nf, Cf).to('cuda:0')  # （170，64）
        flow_D_S = flow_D_S.expand(Bf, Tf, Nf, Cf)  # [B, T, N, C]相当于在第2维复制了T份, 第一维复制B份 (32,12,170,64)
        flow_D_S = flow_D_S.permute(0, 2, 1, 3)  # [B, N, T, C] (32,170,12,64)

        speed_D_S = get_sinusoid_encoding_table(Ns, Cs).to('cuda:0')  # （170，64）
        speed_D_S = speed_D_S.expand(Bs, Ts, Ns, Cs)  # [B, T, N, C]相当于在第2维复制了T份, 第一维复制B份 (32,12,170,64)
        speed_D_S = speed_D_S.permute(0, 2, 1, 3)  # [B, N, T, C] (32,170,12,64)

        # GCN 部分

        flow_X_G = torch.Tensor(Bf, Nf, 0, Cf).to('cuda:0')  # (32,170,0,64)
        speed_X_G = torch.Tensor(Bf, Nf, 0, Cf).to('cuda:0')  # (32,170,0,64)
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)

        for tf in range(flow_query.shape[2]):
            of = self.gcn(flow_query[:, :, tf, :], self.adj)  # [B, N, C]
            of = of.unsqueeze(2)  # shape [N, 1, C] [B, N, 1, C]
            #             print(o.shape)
            flow_X_G = torch.cat((flow_X_G, of), dim=2)
        # flow_X_G 为 (32,170,12,64)

        for ts in range(speed_query.shape[2]):
            os = self.gcn(speed_query[:, :, ts, :], self.adj)  # [B, N, C]
            os = os.unsqueeze(2)  # shape [N, 1, C] [B, N, 1, C]
            #             print(o.shape)
            speed_X_G = torch.cat((speed_X_G, os), dim=2)
        # speed_X_G 为 (32,170,12,64)

        # Spatial Transformer 部分
        flow_query = flow_query + flow_D_S
        speed_query = speed_query + speed_D_S

        # 运用自创的 cross-attention 机制计算出四种 attention，然后前两个做 gate 得到 attention_flow
        # 其中 attention_flow_flow 是来自 flow 的 Q 和来自 flow 的 K 做 self attention，乘上来自 flow 的 V
        # 其中 attention_flow_speed 是来自 flow 的 Q 和来自 speed 的 K 做 cross attention，乘上来自 flow 的 V
        # 其中 attention_speed_flow 是来自 speed 的 Q 和来自 flow 的 K 做 cross attention，乘上来自 speed 的 V
        # 其中 attention_speed_speed 是来自 speed 的 Q 和来自 speed 的 K 做 self attention，乘上来自 speed 的 V
        attention_flow_flow, attention_flow_speed, attention_speed_flow, attention_speed_speed = self.s_cross_attention(flow_query, flow_query, flow_query, speed_query, speed_query, speed_query, mask)  # (B, N, T, C) 三个都是(32,170,12,64)

        # 门控机制融合 attention_flow_flow 和 attention_flow_speed 得到 attention_flow
        # 方法1 直接concat然后线性变换，没写好
        attention_flow = torch.stack((attention_flow_flow, attention_flow_speed),dim=-1)  # output: b,n,t,c,2 (32,170,12,64,2)
        attention_flow = self.linear(attention_flow)  # 应用全连接层 (32,170,12,64,1)
        # attention_flow = self.feed_forward1(attention_flow)  # 应用全连接层 (32,170,12,64,1)
        attention_flow = attention_flow.squeeze(4)  # 去除新增的维度 (32,170,12,64)

        attention_speed = torch.stack((attention_speed_speed, attention_speed_flow),dim=-1)  # output: b,n,t,c,2 (32,170,12,64,2)
        attention_speed = self.linear(attention_speed)  # 应用全连接层 (32,170,12,64,1)
        # attention_speed = self.feed_forward1(attention_speed)  # 应用全连接层 (32,170,12,64,1)
        attention_speed = attention_speed.squeeze(4)  # 去除新增的维度 (32,170,12,64)

        # Add skip connection, run through normalization and finally dropout
        flow_x = self.dropout(self.norm1(attention_flow + flow_query))
        flow_forward = self.feed_forward(flow_x)
        flow_U_S = self.dropout(self.norm2(flow_forward + flow_x))  # (32,170,12,64)

        speed_x = self.dropout(self.norm1(attention_speed + speed_query))
        speed_forward = self.feed_forward(speed_x)
        speed_U_S = self.dropout(self.norm2(speed_forward + speed_x))  # (32,170,12,64)

        # 融合 STransformer and GCN
        flow_g = torch.sigmoid(self.fs(flow_U_S) + self.fg(flow_X_G))  # (7) (32,170,12,64)
        flow_out = flow_g * flow_U_S + (1 - flow_g) * flow_X_G  # (8) (32,170,12,64)

        speed_g = torch.sigmoid(self.fs(speed_U_S) + self.fg(speed_X_G))  # (7)
        speed_out = speed_g * speed_U_S + (1 - speed_g) * speed_X_G  # (8)

        return flow_out, speed_out  # (B, N, T, C)


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout, forward_expansion, num_of_vertices):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion, num_of_vertices)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, flow_value, flow_key, flow_query, speed_value, speed_key, speed_query):
        # value,  key, query: [N, T, C] [B, N, T, C] 以 pems08 为例, 都是 (32,170,12,64)
        # Add skip connection,run through normalization and finally dropout
        flow_x1, speed_x1 = self.STransformer(flow_value, flow_key, flow_query, speed_value, speed_key, speed_query)  # (32,170,12,64)
        flow_x1 = self.norm1(flow_x1 + flow_query)
        speed_x1 = self.norm1(speed_x1 + speed_query)
        flow_x2, speed_x2 = self.TTransformer(flow_x1, flow_x1, flow_x1, speed_x1, speed_x1, speed_x1)
        flow_x2 = self.dropout(self.norm2(flow_x2 + flow_x1))
        speed_x2 = self.dropout(self.norm2(speed_x2 + speed_x1))
        return flow_x2, speed_x2


### Encoder
class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            num_of_vertices,
            cheb_K,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    cheb_K,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    num_of_vertices=num_of_vertices
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, flow_x, speed_x):
        # x: [N, T, C]  [B, N, T, C] 以pems08为例,flow 和 speed 都是 (32,170,12,64)
        # out = self.dropout(x)
        flow_x = self.dropout(flow_x)
        speed_x = self.dropout(speed_x)
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            # out = layer(out, out, out, t)
            flow_x, speed_x = layer(flow_x, flow_x, flow_x, speed_x, speed_x, speed_x)
        # return out
        return flow_x, speed_x


class STTransformerBlock_decoder(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout, forward_expansion, num_of_vertices):
        super(STTransformerBlock_decoder, self).__init__()
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion, num_of_vertices)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)  # 确保ReLU不是inplace

    def forward(self, flow_dec_input, speed_dec_input, flow_memory, speed_memory):
        flow_value, flow_key, flow_query = flow_memory, flow_memory, flow_memory
        speed_value, speed_key, speed_query = speed_memory, speed_memory, speed_memory

        flow_dec_q, flow_dec_k, flow_dec_v = flow_dec_input, flow_dec_input, flow_dec_input
        speed_dec_q, speed_dec_k, speed_dec_v = speed_dec_input, speed_dec_input, speed_dec_input

        tgt_mask = subsequent_mask(speed_value.size(-2)).to("cuda:0")

        flow_dec_self_T, speed_dec_self_T = self.TTransformer(flow_dec_q, flow_dec_k, flow_dec_v, speed_dec_q,
                                                              speed_dec_k, speed_dec_v, tgt_mask)
        flow_x1 = self.norm1(flow_dec_self_T + flow_dec_q)
        speed_x1 = self.norm1(speed_dec_self_T + speed_dec_v)

        flow_x2, speed_x2 = self.TTransformer(flow_value, flow_key, flow_x1, speed_value, speed_key, speed_x1)
        flow_x2 = self.norm2(flow_x2 + flow_x1)
        speed_x2 = self.norm2(speed_x2 + speed_x1)

        flow_x3, speed_x3 = self.STransformer(flow_x2, flow_x2, flow_x2, speed_x2, speed_x2, speed_x2)
        flow_x3 = self.norm3(flow_x3 + flow_x2)
        speed_x3 = self.norm3(speed_x3 + speed_x2)

        flow_x3 = self.dropout(flow_x3)
        speed_x3 = self.dropout(speed_x3)

        return flow_x3, speed_x3


class Decoder(nn.Module):
    # 堆叠多层 ST-Transformer_decoder Block
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            num_of_vertices,
            cheb_K,
            dropout,
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock_decoder(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    cheb_K,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    num_of_vertices=num_of_vertices
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, flow_x, speed_x, flow_dec_input, speed_dec_input):
        # x: [N, T, C]  [B, N, T, C] 以pems08为例,flow 和 speed 都是 (32,170,12,64)
        # out = self.dropout(x)
        flow_x = self.dropout(flow_x)
        speed_x = self.dropout(speed_x)
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            # out = layer(out, out, out, t)
            flow_x, speed_x = layer(flow_dec_input, speed_dec_input, flow_x, speed_x)

        return flow_x, speed_x

### Transformer
class Transformer(nn.Module):
    def __init__(
            self,
            adj,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            num_of_vertices,
            cheb_K,
            dropout,

            device="cuda:0"
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            num_of_vertices,
            cheb_K,
            dropout
        )

        self.decoder = Decoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            num_of_vertices,
            cheb_K,
            dropout
        )
        self.device = device
    def forward(self, flow_src, speed_src, flow_dec_input, speed_dec_input):
        ## scr: [N, T, C]   [B, N, T, C]
        # enc_src = self.encoder(src, t)
        flow_enc_output, speed_enc_output = self.encode(flow_src, speed_src)  # (32,170,12,64)

        flow_dec, speed_dec = self.decode(flow_enc_output, speed_enc_output, flow_dec_input, speed_dec_input)
        # return enc_src  # [B, N, T, C]
        return flow_dec, speed_dec  # [B, N, T, C]

    def encode(self, flow_src, speed_src):
        flow_enc_src, speed_enc_src = self.encoder(flow_src, speed_src)  # (32,170,12,64)
        return flow_enc_src, speed_enc_src
    def decode(self, flow_enc_src, speed_enc_src, flow_dec_input, speed_dec_input):
        flow_dec, speed_dec = self.decoder(flow_enc_src, speed_enc_src, flow_dec_input, speed_dec_input)
        return flow_dec, speed_dec
class FS_ST_Transformer(nn.Module):
    def __init__(
            self,
            adj,
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
            dropout=0
    ):
        super(FS_ST_Transformer, self).__init__()

        self.forward_expansion = forward_expansion
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.Transformer = Transformer(
            adj,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            num_of_vertices,
            cheb_K,
            dropout=0
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 维度都以 pems08 为例，x和dec_input的维度都为(32,3,170,12)
        flow_x = x[:, [0], :, :]  # b,1,n,t (32,1,170,12)
        speed_x = x[:, [-1], :, :]  # b,1,n,t (32,1,170,12)

        # input_Transformer = self.conv1(x) # 以pems08为例，(32,64,170,12)
        # input_Transformer = input_Transformer.permute(0, 2, 3, 1)  # 以pems08为例，(32,170,12,64)

        flow_input_encoder = self.conv1(flow_x)  # (32,64,170,12)
        flow_input_encoder = flow_input_encoder.permute(0, 2, 3, 1)  # (32,170,12,64)
        speed_input_encoder = self.conv1(speed_x)  # (32,64,170,12)
        speed_input_encoder = speed_input_encoder.permute(0, 2, 3, 1)  # (32,170,12,64)

        # flow_dec_input = dec_input[:, [0], :, :]
        # speed_dec_input = dec_input[:, [-1], :, :]

        # flow_dec_input = self.conv1(flow_dec_input)  #
        # flow_dec_input = flow_dec_input.permute(0, 2, 3, 1)  #
        # speed_dec_input = self.conv1(speed_dec_input)  #
        # speed_dec_input =speed_dec_input.permute(0, 2, 3, 1)  # [B,V,T,E]

        # output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        # output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        flow_dec_input, speed_dec_input = self.Transformer.encode(flow_input_encoder, speed_input_encoder)
        flow_output_Transformer, speed_output_Transformer = self.Transformer(flow_input_encoder, speed_input_encoder, flow_dec_input, speed_dec_input)  # [B, N, T, C]


        flow_output_Transformer = flow_output_Transformer.permute(0, 2, 1, 3)  # (32,12,170,64)
        speed_output_Transformer = speed_output_Transformer.permute(0, 2, 1, 3)  # (32,12,170,64)

        # 输出前 flow 和 speed 分别经过两个全连接层

        # out = self.relu(self.conv2(output_Transformer))  # 等号左边 out shape: [B, output_T_dim, N, C]
        # out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim]
        # out = self.conv3(out)  # 等号左边 out shape: [B, 1, N, output_T_dim]
        # out = out.squeeze(1)

        flow_out = self.relu(self.conv2(flow_output_Transformer))  # 等号左边 out shape: [B, output_T_dim, N, C] (32,12,170,64)
        flow_out = flow_out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim] (32,64,170,12)
        flow_out = self.conv3(flow_out)  # 等号左边 out shape: [B, 1, N, output_T_dim] (32,1,170,12)
        flow_out = flow_out.squeeze(1)  # (32,170,12)

        speed_out = self.relu(self.conv2(speed_output_Transformer))  # 等号左边 out shape: [B, output_T_dim, N, C] (32,12,170,64)
        speed_out = speed_out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim] (32,64,170,12)
        speed_out = self.conv3(speed_out)  # 等号左边 out shape: [B, 1, N, output_T_dim] (32,1,170,12)
        speed_out = speed_out.squeeze(1)  # (32,170,12)

        return flow_out, speed_out  # [B, N, output_dim]
        # return out shape: [N, output_dim]

    # 创建掩码
def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   # 1 means reachable; 0 means unreachable
