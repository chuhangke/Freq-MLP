import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, seq_len,pred_len,individual,enc_in,cut_freq):
        super(FITS, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in
        self.output_len=pred_len

        self.dominance_freq=cut_freq # 720/24
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor) -> torch.Tensor:
        # RIN
        x = history_data[..., 0]  # B, L, N
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+2),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # energy compemsation for the length change
        # dom_x=x-low_x
        
        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        xy = xy[:, -self.pred_len:, :]
        return xy.unsqueeze(-1)#这个输出是模型处理后的主输出，代表了模型对原始输入序列 x 的预测结果,这个输出提供了一个中间处理步骤的结果，即频率域处理后再转换回时域的数据，但未加上原始均值

