import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MultiLayerPerceptron
from numbers import Number
from torch.autograd import Variable


class Freq(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, input_len,output_len,individual,num_nodes,cut_freq,node_dim,input_dim,embed_dim,num_layer,temp_dim_tid,temp_dim_diw,time_of_day_size,day_of_week_size,if_T_i_D,if_D_i_W,if_node):
        super().__init__()

        self.seq_len = input_len#model_args["input_len"]
        self.pred_len = output_len#model_args["output_len"]
        self.individual = individual#model_args["individual"]
        self.channels = num_nodes#model_args["num_nodes"]

        self.dominance_freq=cut_freq#model_args["cut_freq"] # 720/24
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
    ############################################################
        # attributes
        self.num_nodes = num_nodes#model_args["num_nodes"]
        self.node_dim = node_dim#model_args["node_dim"]
        self.input_len = input_len#model_args["input_len"]
        self.input_dim = input_dim#model_args["input_dim"]
        self.embed_dim = embed_dim#model_args["embed_dim"]
        self.output_len = output_len#model_args["output_len"]
        self.num_layer = num_layer#model_args["num_layer"]
        self.temp_dim_tid = temp_dim_tid#model_args["temp_dim_tid"]
        self.temp_dim_diw = temp_dim_diw#model_args["temp_dim_diw"]
        self.time_of_day_size = time_of_day_size#model_args["time_of_day_size"]
        self.day_of_week_size = day_of_week_size#model_args["day_of_week_size"]

        self.if_time_in_day = if_T_i_D#model_args["if_T_i_D"]
        self.if_day_in_week = if_D_i_W#model_args["if_D_i_W"]
        self.if_spatial = if_node#model_args["if_node"]
        self.incompressible_flow = bool(1)
        self.conv1 = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # embedding layer
        # self.time_series_emb_layer = NConvolution2D(
        # self.input_dim * self.input_len, self.embed_dim, 1, 1)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
                          int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        #print(self.hidden_dim)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim -21, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        # regression
        # self.regression_layer = NConvolution2D(
        # self.hidden_dim -32, self.output_len, 1, 1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor):
        """Feed forward of Freq-MLP.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        # prepare data

        # RIN
        x = history_data
        B=x.shape[0]
        L=x.shape[1]
        N=x.shape[2]
        C=x.shape[3]
        x=x.resize(B,L,N*C)#[B,L,N*C]
        #print(x.shape)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        #print(x_var.shape)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, 0:self.dominance_freq, :]  # LPF
        #print(low_specx.shape)
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 2), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        #print(low_specxy.shape)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change

        # dom_x=x-low_x

        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        input_data=xy#.resize(32,12,170,1)
        input_data = input_data[:, -self.pred_len:, :]
        input_data=input_data.resize(B,L,N,C)
        #print(input_data.shape)


        #input_data = history_data[..., range(self.input_dim)]
        #predict_data = future_data[..., range(self.input_dim)]  # 32,12,307,3
        # print(future_data.size(1))
        time_Interval = 1
        time_step = 12
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_start_emb = self.time_in_day_emb[
                (t_i_d_data[:, 0, :] * self.time_of_day_size).type(torch.LongTensor)]
            time_in_day_end_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]

            future_time_in_day_start_emb = time_in_day_end_emb
            future_time_in_day_end_emb = future_time_in_day_start_emb + time_step * time_Interval  # self.time_in_day_emb[(f_t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]


        else:
            # time_in_day_emb = None

            time_in_day_start_emb = None
            time_in_day_end_emb = None

            future_time_in_day_start_emb = time_in_day_end_emb
            future_time_in_day_end_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            # f_d_i_w_data = future_data[..., 2]

            # day_in_week_emb = self.day_in_week_emb[(
            # d_i_w_data[:, -1, :]).type(torch.LongTensor)]

            day_in_week_start_emb = self.day_in_week_emb[(
                d_i_w_data[:, 0, :]).type(torch.LongTensor)]
            day_in_week_end_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]

            future_day_in_week_start_emb = day_in_week_end_emb
            future_day_in_week_end_emb = future_day_in_week_start_emb + time_step * time_Interval  # self.day_in_week_emb[(f_d_i_w_data[:, -1,:]).type(torch.LongTensor)]

        else:
            # day_in_week_emb = None
            day_in_week_start_emb = None
            day_in_week_end_emb = None

            future_day_in_week_start_emb = day_in_week_end_emb
            future_time_in_day_end_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # future_time series embedding
        # future_batch_size, _, future_num_nodes, _ = predict_data.shape
        # predict_data = predict_data.transpose(1, 2).contiguous()
        # predict_data = predict_data.view(
        # future_batch_size, future_num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # future_time_series_emb = self.time_series_emb_layer(predict_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        future_tem_emb = []
        tem_emb_start = []
        # if time_in_day_emb is not None:
        # tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))

        if time_in_day_start_emb is not None:
            tem_emb_start.append(time_in_day_start_emb.transpose(1, 2).unsqueeze(-1))
        if time_in_day_end_emb is not None:
            tem_emb.append(time_in_day_end_emb.transpose(1, 2).unsqueeze(-1))

        if future_time_in_day_start_emb is not None:
            future_tem_emb.append(future_time_in_day_start_emb.transpose(1, 2).unsqueeze(-1))
        if future_time_in_day_end_emb is not None:
            future_tem_emb.append(future_time_in_day_end_emb.transpose(1, 2).unsqueeze(-1))

        # if day_in_week_emb is not None:
        # tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        if day_in_week_start_emb is not None:
            tem_emb_start.append(day_in_week_start_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_end_emb is not None:
            tem_emb.append(day_in_week_end_emb.transpose(1, 2).unsqueeze(-1))

        if future_day_in_week_start_emb is not None:
            future_tem_emb.append(future_day_in_week_start_emb.transpose(1, 2).unsqueeze(-1))
        if future_day_in_week_end_emb is not None:
            future_tem_emb.append(future_day_in_week_end_emb.transpose(1, 2).unsqueeze(-1))
        # if future_time_series_emb is not  None:
        # tem_emb.append(future_tem_emb)
        # concate all embeddings

        hidden = torch.cat(tem_emb_start + [time_series_emb] + node_emb + tem_emb, dim=1)
        #print(hidden.shape)
        # encoding
        hidden = self.encoder(hidden)
        batch_size, hidden_dim, nodes_num = hidden.squeeze().size()
        hidden = hidden.squeeze().transpose(-1, -2)
        hidden = hidden.reshape(-1, hidden_dim)

        """                                                                
                mu : [batch_size,z_dim]
                std : [batch_size,z_dim]        
        """
        # hidden=hidden.view(-1, hidden.size(-2))#torch.Size([32, 114560]) torch.Size([3665920, 1])

        # print("hidden:"+str(self.hidden_dim/2))
        mu = hidden[:, :int(hidden_dim / 2)]
        std = F.softplus(hidden[:, int(hidden_dim / 2):] - 5, beta=1)
        # print(std.shape)
        # print(mu.shape)
        encoding = self.reparametrize_n(mu, std, 1)
        encoding = encoding.view(batch_size, nodes_num, -1).transpose(-1, -2).unsqueeze(-1)
        # encoding = encoding.squeeze().view(batch_size, 320, num_nodes, 1)
        # print(encoding.shape)
        # print(std.shape)
        # encoding=torch.cat(tem_emb_start + [encoding] + node_emb + tem_emb , dim=1)
        # regression
        prediction = self.regression_layer(encoding)  # logit = self.decode(encoding)
        # print(prediction.shape)  # 32,12,358,1
        mu = torch.mean(prediction)
        std = torch.exp(prediction)

        return (mu, std), prediction

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor
