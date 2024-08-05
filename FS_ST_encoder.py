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


class SpatialScaledDotProductforCrossAttention(nn.Module):
    def __init__(self, num_of_vertices):
        super(SpatialScaledDotProductforCrossAttention, self).__init__()

        self.flow_speed = FlowSpeed(num_of_vertices)
        # self.linear = nn.Linear(32, 32)

    def forward(self, Qf, Kf, Vf, Qs, Ks, Vs, mask = None):
        B, n_heads, len1, len2, d_k = Qf.shape

        # 把 Qs 经过 greenshields 模型变成 Qf_from_speed，用于与Kf相乘，原有的Ks继续与Qs相乘
        Qf_from_speed = self.flow_speed(Qs)

        # Qf 和 Kf 计算出一个相似度然后经过softmax 乘到 Vf
        scores_flow_flow = torch.matmul(Qf, Kf.transpose(-1, -2)) / np.sqrt(d_k)

        # Kf 和 Qf_from_speed 计算出一个相似度然后经过softmax 乘到 Vf
        scores_flow_speed = torch.matmul(Kf, Qf_from_speed.transpose(-1, -2)) / np.sqrt(d_k)

        # Ks 和 Qf 计算出一个相似度然后经过softmax 乘到 Vs
        scores_speed_flow = torch.matmul(Ks, Qf.transpose(-1, -2)) / np.sqrt(d_k)

        # Qs 和 Ks 计算出一个相似度然后经过softmax 乘到 Vs
        scores_speed_speed = torch.matmul(Qs, Ks.transpose(-1, -2)) / np.sqrt(d_k)

        if mask is not None:
            scores_flow_flow = scores_flow_flow.masked_fill(mask == 0, -1e9)
            scores_flow_speed = scores_flow_speed.masked_fill(mask == 0, -1e9)
            scores_speed_flow = scores_speed_flow.masked_fill(mask == 0, -1e9)
            scores_speed_speed = scores_speed_speed.masked_fill(mask == 0, -1e9)

            print("scores_flow_flow after: ")
            print(scores_flow_flow)
        # Apply softmax to obtain attention weights
        z_flow_flow = nn.Softmax(dim=-1)(scores_flow_flow)
        z_flow_speed = nn.Softmax(dim=-1)(scores_flow_speed)
        z_speed_flow = nn.Softmax(dim=-1)(scores_speed_flow)
        z_speed_speed = nn.Softmax(dim=-1)(scores_speed_speed)

        # Compute context vectors
        context_flow_flow = torch.matmul(z_flow_flow, Vf)
        context_flow_speed = torch.matmul(z_flow_speed, Vf)
        context_speed_flow = torch.matmul(z_speed_flow, Vs)
        context_speed_speed = torch.matmul(z_speed_speed, Vs)

        return context_flow_flow, context_flow_speed, context_speed_flow, context_speed_speed

class SpatialMultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_size, heads, num_of_vertices):
        super(SpatialMultiHeadCrossAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.W_fV = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_fK = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_fQ = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sV = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sK = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_sQ = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)

        self.SpatialScaledDotProductforCrossAttention = SpatialScaledDotProductforCrossAttention(num_of_vertices)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, flow_input_Q, flow_input_K, flow_input_V, speed_input_Q, speed_input_K, speed_input_V, mask=None):
        Bf, Nf, Tf, Cf = flow_input_Q.shape
        Bs, Ns, Ts, Cs = speed_input_Q.shape

        if mask is not None:
            mask = mask

        Qf = self.W_fQ(flow_input_Q).view(Bf, Nf, Tf, self.heads, self.head_dim).transpose(1, 3)
        Kf = self.W_fK(flow_input_K).view(Bf, Nf, Tf, self.heads, self.head_dim).transpose(1, 3)
        Vf = self.W_fV(flow_input_V).view(Bf, Nf, Tf, self.heads, self.head_dim).transpose(1, 3)
        Qs = self.W_sQ(speed_input_Q).view(Bs, Ns, Ts, self.heads, self.head_dim).transpose(1, 3)
        Ks = self.W_sK(speed_input_K).view(Bs, Ns, Ts, self.heads, self.head_dim).transpose(1, 3)
        Vs = self.W_sV(speed_input_V).view(Bs, Ns, Ts, self.heads, self.head_dim).transpose(1, 3)

        context_flow_flow, context_flow_speed, context_speed_flow, context_speed_speed = self.SpatialScaledDotProductforCrossAttention(Qf, Kf, Vf, Qs, Ks, Vs, mask)

        context_flow_flow = context_flow_flow.permute(0, 3, 2, 1, 4).reshape(Bf, Nf, Tf, self.heads * self.head_dim)
        output_flow_flow = self.fc_out(context_flow_flow)

        context_flow_speed = context_flow_speed.permute(0, 3, 2, 1, 4).reshape(Bf, Nf, Tf, self.heads * self.head_dim)
        output_flow_speed = self.fc_out(context_flow_speed)

        context_speed_flow = context_speed_flow.permute(0, 3, 2, 1, 4).reshape(Bs, Ns, Ts, self.heads * self.head_dim)
        output_speed_flow = self.fc_out(context_speed_flow)

        context_speed_speed = context_speed_speed.permute(0, 3, 2, 1, 4).reshape(Bs, Ns, Ts, self.heads * self.head_dim)
        output_speed_speed = self.fc_out(context_speed_speed)

        return output_flow_flow, output_flow_speed, output_speed_flow, output_speed_speed

class ST_block(nn.Module):
    def __init__(self, embed_size, heads, adj, cheb_K, dropout, forward_expansion, num_of_vertices):
        super(ST_block, self).__init__()
        self.adj = adj.to('cuda:0')
        self.heads = heads
        self.embed_size = embed_size
        self.num_of_vertices = num_of_vertices
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(2, 1)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.gcn = GCN(embed_size, embed_size * 2, embed_size, adj, cheb_K, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

        # Add gate layers
        self.flow_gate = nn.Linear(embed_size * 2, embed_size)
        self.speed_gate = nn.Linear(embed_size * 2, embed_size)

        # Add cross-attention module
        self.cross_attention = SpatialMultiHeadCrossAttention(embed_size, heads, num_of_vertices)

    def get_shift_list(self, T, window_size):
        idxs = np.arange(T)
        window_list = np.arange(-(window_size-1)//2, (window_size-1)//2+1, 1)
        shift_list = []
        for i in window_list:
            tmp = idxs + i
            tmp[tmp < 0] = tmp[tmp < 0] + T
            tmp[tmp >= T] = tmp[tmp >= T] - T
            shift_list.append(tmp)
        return np.array(shift_list)

    def forward(self, flow_query, flow_key, flow_value, speed_query, speed_key, speed_value, mask=None):
        Bf, Nf, Tf, Cf = flow_query.shape
        Bs, Ns, Ts, Cs = speed_query.shape

        flow_D_S = get_sinusoid_encoding_table(Nf, Cf).to('cuda:0')
        flow_D_S = flow_D_S.expand(Bf, Tf, Nf, Cf).permute(0, 2, 1, 3)

        speed_D_S = get_sinusoid_encoding_table(Ns, Cs).to('cuda:0')
        speed_D_S = speed_D_S.expand(Bs, Ts, Ns, Cs).permute(0, 2, 1, 3)

        flow_D_T = get_sinusoid_encoding_table(Tf, Cf).to('cuda:0')
        flow_D_T = flow_D_T.expand(Bf, Nf, Tf, Cf)

        speed_D_T = get_sinusoid_encoding_table(Ts, Cs).to('cuda:0')
        speed_D_T = speed_D_T.expand(Bs, Ns, Ts, Cs)

        flow_query = flow_query + flow_D_S + flow_D_T
        speed_query = speed_query + speed_D_S + speed_D_T


        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)
        adj = self.adj.to('cuda:0')
        # 创建一个列表来存储每个时间步的输出
        flow_X_G_list = []
        speed_X_G_list = []

        for tf in range(flow_query.shape[2]):
            of = self.gcn(flow_query[:, :, tf, :], adj)  # of shape: (B, N, C)
            flow_X_G_list.append(of.unsqueeze(2))  # 保持时间维度

        for ts in range(speed_query.shape[2]):
            os = self.gcn(speed_query[:, :, ts, :], adj)  # os shape: (B, N, C)
            speed_X_G_list.append(os.unsqueeze(2))  # 保持时间维度

        # 在时间维度上拼接所有的时间步结果
        flow_X_G = torch.cat(flow_X_G_list, dim=2)  # 最终形状 (B, N, T, C)
        speed_X_G = torch.cat(speed_X_G_list, dim=2)  # 最终形状 (B, N, T, C)

        # Attention mechanism using sliding window and concatenated time information
        window_size = 5  # Example window size
        shift_list = self.get_shift_list(Tf, window_size)

        res_flow = 0
        res_speed = 0

        for shift in shift_list:
            query_flow = flow_query[:, :, shift, :]
            key_flow = flow_key[:, :, shift, :]  # Use flow_key here
            value_flow = flow_value[:, :, shift, :]

            query_speed = speed_query[:, :, shift, :]
            key_speed = speed_key[:, :, shift, :]  # Use speed_key here
            value_speed = speed_value[:, :, shift, :]

            output_flow_flow, output_flow_speed, output_speed_flow, output_speed_speed = self.cross_attention(
                query_flow, key_flow, value_flow, query_speed, key_speed, value_speed, mask
            )

            # Apply gate mechanism to combine outputs
            combined_flow = torch.cat((output_flow_flow, output_flow_speed), dim=-1)
            gated_flow = torch.sigmoid(self.flow_gate(combined_flow))
            res_flow += gated_flow * output_flow_flow + (1 - gated_flow) * output_flow_speed

            combined_speed = torch.cat((output_speed_speed, output_speed_speed), dim=-1)
            gated_speed = torch.sigmoid(self.speed_gate(combined_speed))
            res_speed += gated_speed * output_speed_speed + (1 - gated_speed) * output_speed_speed

        res_flow /= len(shift_list)
        res_speed /= len(shift_list)

        flow_x = self.norm1(res_flow + flow_query)
        flow_forward = self.feed_forward(flow_x)
        flow_U_S = self.norm2(flow_forward + flow_x)

        speed_x = self.norm1(res_speed + speed_query)
        speed_forward = self.feed_forward(speed_x)
        speed_U_S = self.norm2(speed_forward + speed_x)

        flow_g = torch.sigmoid(self.fs(flow_U_S) + self.fg(flow_X_G))
        flow_out = flow_g * flow_U_S + (1 - flow_g) * flow_X_G

        speed_g = torch.sigmoid(self.fs(speed_U_S) + self.fg(speed_X_G))
        speed_out = speed_g * speed_U_S + (1 - speed_g) * speed_X_G

        return flow_out, speed_out



class encoder_layer(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout, forward_expansion, num_of_vertices):
        super(encoder_layer, self).__init__()
        self.ST_block = ST_block(embed_size, heads, adj, cheb_K, dropout, forward_expansion, num_of_vertices)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        # 添加前馈神经网络部分
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, flow_value, flow_key, flow_query, speed_value, speed_key, speed_query):
        # value, key, query: [N, T, C] [B, N, T, C] 以 pems08 为例, 都是 (32,170,12,64)

        # 进行时空变换
        flow_x1, speed_x1 = self.ST_block(flow_value, flow_key, flow_query, speed_value, speed_key,
                                              speed_query)  # (32,170,12,64)

        # 加上残差连接和归一化
        flow_x1 = self.norm1(flow_x1 + flow_query)
        speed_x1 = self.norm1(speed_x1 + speed_query)

        # 通过前馈神经网络
        flow_x2 = self.feed_forward(flow_x1)
        speed_x2 = self.feed_forward(speed_x1)

        # 加上残差连接和归一化
        flow_out = self.dropout(self.norm2(flow_x2 + flow_x1))
        speed_out = self.dropout(self.norm2(speed_x2 + speed_x1))

        return flow_out, speed_out


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
                encoder_layer(
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


        self.device = device
    def forward(self, flow_src, speed_src):
        ## scr: [N, T, C]   [B, N, T, C]
        # enc_src = self.encoder(src, t)
        flow_enc_output, speed_enc_output = self.encode(flow_src, speed_src)  # (32,170,12,64)

        return flow_enc_output, speed_enc_output  # [B, N, T, C]

    def encode(self, flow_src, speed_src):
        flow_enc_src, speed_enc_src = self.encoder(flow_src, speed_src)  # (32,170,12,64)
        return flow_enc_src, speed_enc_src

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


        flow_output_Transformer, speed_output_Transformer = self.Transformer(flow_input_encoder, speed_input_encoder)

        flow_output_Transformer = flow_output_Transformer.permute(0, 2, 1, 3)  # (32,12,170,64)
        speed_output_Transformer = speed_output_Transformer.permute(0, 2, 1, 3)  # (32,12,170,64)

        flow_out = self.relu(self.conv2(flow_output_Transformer))  # 等号左边 out shape: [B, output_T_dim, N, C] (32,12,170,64)
        flow_out = flow_out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim] (32,64,170,12)
        flow_out = self.conv3(flow_out)  # 等号左边 out shape: [B, 1, N, output_T_dim] (32,1,170,12)
        flow_out = flow_out.squeeze(1)  # (32,170,12)

        speed_out = self.relu(self.conv2(speed_output_Transformer))  # 等号左边 out shape: [B, output_T_dim, N, C] (32,12,170,64)
        speed_out = speed_out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim] (32,64,170,12)
        speed_out = self.conv3(speed_out)  # 等号左边 out shape: [B, 1, N, output_T_dim] (32,1,170,12)
        speed_out = speed_out.squeeze(1)  # (32,170,12)

        return flow_out, speed_out  # [B, N, output_dim]




def main():

    # 测试代码
    B, N, T, C = 2, 170, 12, 64  # Batch size, Number of nodes, Time steps, Embedding size
    adj = torch.rand(N, N)  # Adjacency matrix
    flow_query = torch.rand(B, N, T, C).to('cuda:0')
    flow_key = torch.rand(B, N, T, C).to('cuda:0')
    flow_value = torch.rand(B, N, T, C).to('cuda:0')
    speed_query = torch.rand(B, N, T, C).to('cuda:0')
    speed_key = torch.rand(B, N, T, C).to('cuda:0')
    speed_value = torch.rand(B, N, T, C).to('cuda:0')

    st_block = ST_block(embed_size=C, heads=8, adj=adj, cheb_K=3, dropout=0.1, forward_expansion=4,
                        num_of_vertices=N).to('cuda:0')
    flow_out, speed_out = st_block(flow_query, flow_key, flow_value, speed_query, speed_key, speed_value)
    print(flow_out.shape, speed_out.shape)
    return