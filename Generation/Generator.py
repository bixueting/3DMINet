# encoding=utf-8
from datetime import datetime
import numpy as np
import pandas as pd
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import open3d as o3d
import heapq
np.seterr(divide='ignore',invalid='ignore')
#from pointnet_util import sample_and_group 

# add for shape-preserving Loss
from collections import namedtuple
# from pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
cudnn.benchnark=True
from Generation.modules import *
from torch.nn import AvgPool2d, Conv1d, Conv2d, Embedding, LeakyReLU, Module

neg = 0.01
neg_2 = 0.2


# def iss(pc0):
#     pc = pc0.detach().cpu().numpy()

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc)
#     # pcd.normals = o3d.utility.Vector3dVector(point_cloud[['nx', 'ny', 'nz']].values)

#     # build search tree:
#     search_tree = o3d.geometry.KDTreeFlann(pcd)

#     # point handler:
#     points = np.asarray(pcd.points)

#     point_eigen_values = {'id': [], 'lambda_0': [], 'lambda_1': [], 'lambda_2': []}

#     # num rnn cache:
#     num_rnn_cache = {}
#     # set radius
#     radius = 0.1
#     # heapq for non-maximum suppression:
#     pq = []
    
#     for idx_center, center in enumerate(points):
#         # find radius nearest neighbors:
#         [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(center, radius)

#         # for each point get its nearest neighbors count:
#         w = []
#         direction = []
#         for idx_neighbor in np.asarray(idx_neighbors[1:]):
#             # check cache:
#             if not idx_neighbor in num_rnn_cache:
#                 [k_, _, _] = search_tree.search_radius_vector_3d(points[idx_neighbor], radius)
#                 num_rnn_cache[idx_neighbor] = k_
#             # update:
#             w.append(num_rnn_cache[idx_neighbor])
#             direction.append(points[idx_neighbor] - center)
        
#         # calculate covariance matrix:
#         w = np.asarray(w)
#         direction = np.asarray(direction)
        
#         #cov0 = (1.0 / w.sum()) * np.dot(direction.T, np.dot(np.diag(w), direction))
#         lenth = len(w)
#         cov = np.zeros([3,3])
#         for j in range(lenth):
#             b=direction[j].reshape(direction[j].shape[0],1)  # 一维数组实现转置用reshape
#             b=np.mat(b)
#             cov = cov + np.dot(b,np.mat(direction[j]))*w[j]
        
#         cov = np.asarray(cov)
#         cov = cov / w.sum()
#         # print(abs(cov[0] - cov0[0]) <= 1e-5)  判断两种方式求的协方差矩阵是否相同
#         # get eigenvalues:
#         e, _ = np.linalg.eig(cov)   #计算矩阵特征值和特征向量
#         e = e[e.argsort()[::-1]]  #argsort返回从大到小排列的索引，将特征值从小到大排列

#         # add to pq:
#         heapq.heappush(pq, (-e[2], idx_center))  #将每个点的索引和其最大特征值的符号送入堆

#         # add to dataframe:
#         point_eigen_values['id'].append(idx_center)
#         point_eigen_values['lambda_0'].append(e[0])
#         point_eigen_values['lambda_1'].append(e[1])
#         point_eigen_values['lambda_2'].append(e[2])
#     # print(pq)
#     # non-maximum suppression:
#     suppressed = set()
#     while pq:
#         _, idx_center = heapq.heappop(pq)  # 堆中的最小元素弹出，因为前面取了负号，其实是最大的特征值
#         if not idx_center in suppressed:
#             # suppress its neighbors:
#             [_, idx_neighbors, _] = search_tree.search_radius_vector_3d(points[idx_center], radius)
#             for idx_neighbor in np.asarray(idx_neighbors[1:]):
#                 suppressed.add(idx_neighbor)
#         else:
#             continue

#     # format:        
    
#     point_eigen_values = pd.DataFrame.from_dict(point_eigen_values)
#     # print(point_eigen_values)
#     # first apply non-maximum suppression:
#     point_eigen_values = point_eigen_values.loc[point_eigen_values['id'].apply(lambda id: not id in suppressed), point_eigen_values.columns]

#     # then apply decreasing ratio test:
#     point_eigen_values = point_eigen_values.loc[
#         (point_eigen_values['lambda_0'] > point_eigen_values['lambda_1']) &
#         (point_eigen_values['lambda_1'] > point_eigen_values['lambda_2']),
#         point_eigen_values.columns
#     ]
#     point_eigen_values = point_eigen_values.sort_values('lambda_2', axis=0, ascending=False, ignore_index=True)
#     # print(type(point_eigen_values['id'].values))
#     # paint background as grey:
#     # pcd.paint_uniform_color([0.95, 0.95, 0.95])
#     # paint keypoints as red:
#     # np.asarray(pcd.colors)[point_eigen_values['id'].values[0], :] = [1.0, 0.0, 0.0]

#     # o3d.visualization.draw_geometries([pcd])    
#     return point_eigen_values['id'].values[0]





# class Local_op(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Local_op, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.bn2 = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
#         x = x.permute(0, 1, 3, 2)   
#         x = x.reshape(-1, d, s) 
#         batch_size, _, N = x.size()
#         x = F.relu(self.bn1(self.conv1(x))) # B, D, N
#         x = F.relu(self.bn2(self.conv2(x))) # B, D, N
#         x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x = x.reshape(b, n, -1).permute(0, 2, 1)
#         return x

class OffsetAttention1(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)       
        self.relu = nn.ReLU()
        self.sa1 = SA_Layer(channels) 
        
    def forward(self, x):

        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x1 = self.sa1(x)
        

        # x0 = x[0].permute(1, 0) 
        # point_eigen_max_value = iss(x0)
        # m0 = map1[0][point_eigen_max_value]
        # s = m0.argsort()

        return x1


class OffsetAttention2(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(64, channels, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        # self.bn2 = nn.BatchNorm1d(channels)

        # self.sa1 = SA_Layer(channels)
        self.sa1 = SA_Layer_Vis(channels)
        # self.sa2 = SA_Layer(channels)
        # self.sa3 = SA_Layer(channels)
        # self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        # batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        # x = self.relu(self.bn2(self.conv2(x)))

        
        # x = self.sa2(x)
        # x3 = self.sa3(x2)
        # x4 = self.sa4(x3)
        x1, map1 = self.sa1(x)   # map1 NxN
        
        #x = torch.cat((x1, x2, x3, x4), dim=1)

        
        return x1, map1

class SA_Layer_Vis(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_Vis, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x, energy
        # return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        Conv = EqualConv1d if use_eql else nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out




class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts
        self.np = opts.np
        self.nk = opts.nk//2
        self.nz = opts.nz
        softmax = opts.softmax
        self.off = opts.off
        self.use_attn = opts.attn
        self.use_head = opts.use_head

        Conv = EqualConv1d if self.opts.eql else nn.Conv1d
        Linear = EqualLinear if self.opts.eql else nn.Linear

        dim = 128
        self.head = nn.Sequential(
            Conv(3 + self.nz, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_attn:
            self.attn = Attention(dim + 512)

        self.global_conv = nn.Sequential(
            Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )


        # self.tail = nn.Sequential(
        #     Conv1d(512+dim, 256, 1),
        #     nn.LeakyReLU(neg, inplace=True),
        #     Conv1d(256, 64, 1),
        #     nn.LeakyReLU(neg, inplace=True),
        #     Conv1d(64, 3, 1),
        #     nn.Tanh()
        # )

        self.tail = nn.Sequential(
            Conv1d(dim+512, 256, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(256, 64, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(64, 3, 1),
            nn.Tanh()
        )

        if self.use_head:
            self.pc_head = nn.Sequential(
                Conv(3, dim // 2, 1),
                nn.LeakyReLU(inplace=True),
                Conv(dim // 2, dim, 1),
                nn.LeakyReLU(inplace=True),
            )
            # self.EdgeConv1 = EdgeBlock(dim, dim, self.nk)
            # print('yes')
            self.adain1 = AdaptivePointNorm(dim, dim)
            # self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)
        else:
            # self.EdgeConv1 = EdgeBlock(3, 64, self.nk)
            print('no')
            self.adain1 = AdaptivePointNorm(64, dim)
            # self.EdgeConv2 = EdgeBlock(64, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)

        # self.conv_input = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        # self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        # self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.offset_attention1 = OffsetAttention1()
        self.offset_attention2 = OffsetAttention2()
        # self.conv_output = nn.Conv1d(1024, 128, kernel_size=1, bias=False)



    def forward(self, x, z):

        B,N,_ = x.size()
        #print(N)

        if self.opts.z_norm:
            z = z / (z.norm(p=2, dim=-1, keepdim=True)+1e-8)

        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N    FE

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        #print(pc.shape)

        # x1 = self.EdgeConv1(pc)   # FG1
        # x1, att_map = self.offset_attention1(pc)
        x1 = self.offset_attention1(pc)
        #print(x1.shape)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        #print(x1.shape)
        #x2 = self.EdgeConv2(x1)
        x2, att_map = self.offset_attention2(x1)
        #print(x2.shape)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)  #16 128 2048
        #print('x2:',x2.shape)
        # x_l = self.offset_attentions(x2)  # 
        #print('xl',x_l.shape)
        # feat_global = torch.max(x_l, 2, keepdim=True)[0]
        # feat_global = feat_global.view(B, -1)
        # feat_global = self.global_conv(feat_global)
        # feat_global = feat_global.view(B, -1, 1)
        # feat_global = feat_global.repeat(1, 1, N)

        # feat_cat = torch.cat((feat_global, x2), dim=1)
        #print(feat_cat.shape)
        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)



        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_o = self.tail(feat_cat)                   # Bx3x256

        x1_p = pc + x1_o if self.off else x1_o
        

        # x0 = x1_p[0].permute(1, 0)
        # xn = x0.cuda().data.cpu().numpy()
        # import random
        # k = random.choice(np.random.randint(0, 2048, 3))
        
        # sorts = att_map[0][k].argsort()
        # sorts = torch.flip(sorts, dims=[0])
        # # print(sorts.shape)
        # xx = xn[sorts]

        # import matplotlib.pyplot as plt
        # colors = plt.get_cmap("RdYlGn")(np.linspace(0.05, 0.95, 2048))
        # # colors = plt.get_cmap("hot")(np.linspace(0.1, 0.9, 2048))
        # colors[0, :3] = np.array([0, 0, 0])

        # arr = np.concatenate([xx, colors[:, :3]], axis=1)

        # save_dir = 'attention_map_3Dimaging'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # f = open('current_epoch.txt')
        # ec = f.read()

        # pcc_name = os.path.join(save_dir, "epoch%s.txt" % (ec))
        # np.savetxt(pcc_name, arr)

        return x1_p

    def interpolate(self, x, z1, z2, selection, alpha, use_latent = False):

        if not use_latent:

            ## interpolation
            z = z1
            z[:, selection == 1] = z1[:, selection == 1] * (1 - alpha) + z2[:, selection == 1] * (alpha)

            B, N, _ = x.size()
            if self.opts.z_norm:
                z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            style = torch.cat([x, z], dim=-1)
            style = style.transpose(2, 1).contiguous()
            style = self.head(style)  # B,C,N

        else:
            # interplolation
            B, N, _ = x.size()
            if self.opts.z_norm:
                z1 = z1 / (z1.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                z2 = z2 / (z2.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            style_1 = torch.cat([x, z1], dim=-1)
            style_1 = style_1.transpose(2, 1).contiguous()
            style_1 = self.head(style_1)  # B,C,N

            style_2 = torch.cat([x, z2], dim=-1)
            style_2 = style_2.transpose(2, 1).contiguous()
            style_2 = self.head(style_2)  # B,C,N

            style = style_1
            style[:, :, selection == 1] = style_1[:, :, selection == 1] * (1 - alpha) + style_2[:, :, selection == 1] * alpha

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        # x1 = self.EdgeConv1(pc)
        x1 = self.offset_attention1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        # x2 = self.EdgeConv2(x1)
        x2 = self.offset_attention2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_o = self.tail(feat_cat)  # Bx3x256

        x1_p = pc + x1_o if self.off else x1_o

        return x1_p


