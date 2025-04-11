import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_conv import *
from torch.autograd import Variable
import pdb
from .preprocess import BigSTPreprocess
from .model import Model
from src.base.model import BaseModel

def sample_period(x, time_num):
    # trainx (B, N, T, F)
    history_length = x.shape[-2]
    idx_list = [i for i in range(history_length)]
    period_list = [idx_list[i:i+12] for i in range(0, history_length, time_num)]
    period_feat = [x[:,:,sublist,0] for sublist in period_list]
    period_feat = torch.stack(period_feat)
    period_feat = torch.mean(period_feat, dim=0)
    
    return period_feat

class BigST(nn.Module):
    """
    Paper: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks
    Link: https://dl.acm.org/doi/10.14778/3641204.3641217
    Official Code: https://github.com/usail-hkust/BigST?tab=readme-ov-file
    Venue: VLDB 2024
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, seq_num, in_dim, out_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim,
                 time_emb_dim,output_len, \
                 use_residual, use_bn, use_spatial, use_long, dropout, time_of_day_size, day_of_week_size,
                 supports=None, edge_indices=None,preprocess_args=None,preprocess_path=None):
        super(BigST, self).__init__()
        self.output_len =output_len

        self.tau = tau
        self.layer_num = 3
        self.in_dim = in_dim
        self.random_feature_dim = random_feature_dim

        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_spatial = use_spatial
        self.use_long = use_long

        self.dropout = dropout
        self.activation = nn.ReLU()
        self.supports = supports

        self.time_num = time_of_day_size
        self.week_num = day_of_week_size

        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        self.input_emb_layer = nn.Conv2d(seq_num * in_dim, hid_dim, kernel_size=(1, 1), bias=True)

        self.W_1 = nn.Conv2d(node_emb_dim + time_emb_dim * 2, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim + time_emb_dim * 2, hid_dim, kernel_size=(1, 1), bias=True)

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        if self.use_long:
            self.feat_extractor = BigSTPreprocess(preprocess_args)
            self.load_pre_trained_model(preprocess_path)

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        for i in range(self.layer_num):
            self.linear_conv.append(
                linearized_conv(hid_dim * 4, hid_dim * 4, self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm(hid_dim * 4))

        if self.use_long:
            self.regression_layer = nn.Conv2d(hid_dim * 4 * 2 + hid_dim + seq_num, out_dim, kernel_size=(1, 1),
                                              bias=True)
        else:
            self.regression_layer = nn.Conv2d(hid_dim * 4 * 2, out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        x = history_data[..., range(self.in_dim)]

        # x: (B, N, T, D)
        B, T, N, D = x.size()
        
        x=x.reshape(B, N, T,  D)

        # # 修复后的时间索引处理
        # time_index = (x[:, :, -1, 1] * self.time_num).long().clamp(min=0, max=self.time_num - 1)
        # week_index = x[:, :, -1, 2].long().clamp(min=0, max=self.week_num - 1)
        # 
        # time_emb = self.time_emb_layer[time_index]
        # week_emb = self.week_emb_layer[week_index]
        
        time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).type(torch.LongTensor)]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]

        # time_emb = self.time_emb_layer[(x[:, -1, :, 1] * self.time_num).type(torch.LongTensor)]#torch.Size([64, 12, 32])
        # week_emb = self.week_emb_layer[(x[:, -1, :, 2]).type(torch.LongTensor)]

        # input embedding
        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)torch.Size([64, 36, 716, 1])

        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)

        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)torch.Size([64, 36, 716, 1])
        
        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)torch.Size([64, 128, 716, 1])

        # linearized spatial convolution
        x_pool = [x]  # (B, dim*4, N, 1)
        node_vec1 = self.W_1(x_g)  # (B, dim, N, 1)
        node_vec2 = self.W_2(x_g)  # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        for i in range(self.layer_num):
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)

            if self.use_residual:
                x = x + residual

            if self.use_bn:
                x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)torch.Size([64, 716, 1, 128])
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)#torch.Size([64, 128, 716, 1])


        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)  # (B, dim*4, N, 1)torch.Size([64, 256, 716, 1])

        x = self.activation(x)  # (B, dim*4, N, 1)torch.Size([64, 256, 716, 1])


        if self.use_long:
            
            feat = []
            for i in range(history_data.shape[0]):
                with torch.no_grad():
                     feat_sample = self.feat_extractor(history_data[[i],:,:,:], future_data, batch_seen, epoch, train)
                feat.append(feat_sample['feat'])

            feat = torch.cat(feat, dim=0)
            feat_period = sample_period(history_data, self.time_num)
            feat = torch.cat([feat, feat_period], dim=2)
            
            feat = feat.permute(0, 2, 1).unsqueeze(-1)  # (B, F, N, 1)
            x = torch.cat([x, feat], dim=1)
            x = self.regression_layer(x)  # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)
        else:
            x = self.regression_layer(x)  # (B, N, T)torch.Size([64, 1, 716, 1])
            x = x.squeeze(-1).permute(0, 2, 1)


        return x.transpose(1, 2).unsqueeze(-1)#torch.Size([6976, 1, 716]) torch.Size([6976, 12, 716])


        # return {"prediction": x.transpose(1,2).unsqueeze(-1)
        #       , "node_vec1": node_vec1_prime
        #       , "node_vec2": node_vec2_prime
        #       , "supports": self.supports
        #       , 'use_spatial': self.use_spatial}

    # def __init__(self,preprocess_path, preprocess_args,use_long,in_dim,out_dim,time_num,input_dim,output_dim,node_num):
    #     super(BigST, self).__init__()
    # 
    #     self.use_long = use_long
    #     self.in_dim = in_dim
    #     self.out_dim = out_dim
    #     self.time_num = time_num
    #     self.bigst = Model(**bigst_args)
    #     # self.input_dim=in_dim
    #     # self.output_dim=out_dim
    #     # self.node_num=node_num
    # 
    #     if self.use_long:
    #         self.feat_extractor = BigSTPreprocess(**preprocess_args)
    #         self.load_pre_trained_model(preprocess_path)
    #         
    # def load_pre_trained_model(self, preprocess_path):
    #     """Load pre-trained model"""
    # 
    #     # load parameters
    #     checkpoint_dict = torch.load(preprocess_path)
    #     self.feat_extractor.load_state_dict(checkpoint_dict["model_state_dict"])
    #     # freeze parameters
    #     for param in self.feat_extractor.parameters():
    #         param.requires_grad = False
    # 
    #     self.feat_extractor.eval()
    # 
    # 
    # def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
    #     history_data = history_data.transpose(1,2) # (B, N, T, D)
    #     x = history_data[:, :, -self.out_dim:]         # (batch_size, in_len, data_dim)
    # 
    #     if self.use_long:
    #         feat = []
    #         for i in range(history_data.shape[0]):
    #             with torch.no_grad():
    #                  feat_sample = self.feat_extractor(history_data[[i],:,:,:], future_data, batch_seen, epoch, train)
    #             feat.append(feat_sample['feat'])
    # 
    #         feat = torch.cat(feat, dim=0)
    #         feat_period = sample_period(history_data, self.time_num)
    #         feat = torch.cat([feat, feat_period], dim=2)
    # 
    #         return self.bigst(x, feat)
    # 
    #     else:
    #         return self.bigst(x)

        