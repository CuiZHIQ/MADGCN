import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpatialGCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, order=2):
        super().__init__()
        self.gcn = gcn(in_dim, out_dim, dropout, support_len=1, order=order)
        self.out_dim = out_dim
        
    def forward(self, x, adj):
        batch_time, num_nodes, channels = x.shape
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        if adj.dim() == 2:
            support = [adj]
        elif adj.dim() == 3:
            if adj.shape[0] == batch_time:
                support = [adj.mean(dim=0)]
            else:
                support = [adj.mean(dim=0)]
        else:
            raise ValueError(f"Unsupported adjacency matrix shape: {adj.shape}")
            
        output = self.gcn(x, support)
        output = output.squeeze(-1).permute(0, 2, 1)
        return output 

class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.randn(cycle_len, channel_size) * 0.1, requires_grad=True)

    def forward(self, index, length):
        batch_size = index.shape[0]
        outputs = []
        
        for i in range(batch_size):
            cycle_idx = index[i].item() % self.cycle_len
            rolled_data = torch.roll(self.data, shifts=-cycle_idx, dims=0)
            if length <= self.cycle_len:
                output = rolled_data[:length]
            else:
                num_repeats = length // self.cycle_len
                remainder = length % self.cycle_len
                repeated = rolled_data.repeat(num_repeats, 1)
                if remainder > 0:
                    output = torch.cat([repeated, rolled_data[:remainder]], dim=0)
                else:
                    output = repeated
            outputs.append(output)
        
        return torch.stack(outputs)

class EnhancedSeasonalModule(nn.Module):
    def __init__(self, in_dim, cycle_len=24, gamma=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.cycle_len = cycle_len
        self.recurrent_cycle = RecurrentCycle(cycle_len, in_dim)
        self.W_c = nn.Linear(in_dim, in_dim)
        self.b_c = nn.Parameter(torch.zeros(in_dim))
        self.enhancement_conv = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, seasonal_component, cycle_index=None):
        batch_size, seq_len, num_nodes, channels = seasonal_component.shape
        if cycle_index is None:
            cycle_index = torch.arange(batch_size, device=seasonal_component.device) % self.cycle_len
        enhanced_components = []
        for n in range(num_nodes):
            node_seasonal = seasonal_component[:, :, n, :]
            cycle_enhanced = self.recurrent_cycle(cycle_index, seq_len)
            combined = node_seasonal * cycle_enhanced
            combined_reshaped = combined.view(-1, channels)
            transformed = self.W_c(combined_reshaped) + self.b_c
            transformed = transformed.view(batch_size, seq_len, channels)
            conv_input = transformed.permute(0, 2, 1)
            conv_output = self.activation(self.enhancement_conv(conv_input))
            conv_output = conv_output.permute(0, 2, 1)
            enhanced = node_seasonal + self.gamma * self.norm(conv_output)
            enhanced_components.append(enhanced)
        enhanced_seasonal = torch.stack(enhanced_components, dim=2)
        return enhanced_seasonal

class PatchMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.dropout(x_conv)
        x = residual + x_conv
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class PatchMixerBackbone(nn.Module):
    def __init__(self, seq_len, patch_len, stride, in_channel, d_model=128, 
                 n_layers=3, kernel_size=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        if seq_len % stride != 0:
            self.patch_num += 1
        self.patch_embedding = nn.Linear(patch_len * in_channel, d_model)
        self.mixer_layers = nn.ModuleList([
            PatchMixerLayer(d_model, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]
            patches.append(patch.reshape(batch_size, -1))
        if len(patches) * self.stride < seq_len:
            last_patch = x[:, -self.patch_len:, :]
            patches.append(last_patch.reshape(batch_size, -1))
        patches = torch.stack(patches, dim=1)
        x = self.patch_embedding(patches)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x

class STLDecomposition(nn.Module):
    def __init__(self, period=24, num_iterations=2):
        super().__init__()
        self.period = period
        self.num_iterations = num_iterations
        
    def simple_moving_average(self, x, window_size):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, seq_len = x.shape
        kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
        x_padded = F.pad(x.unsqueeze(1), (window_size//2, window_size//2), mode='replicate')
        smoothed = F.conv1d(x_padded, kernel, padding=0)
        return smoothed.squeeze(1)
    
    def extract_trend(self, x):
        window_size = max(3, self.period // 2)
        if window_size % 2 == 0:
            window_size += 1
        return self.simple_moving_average(x, window_size)
    
    def extract_seasonal(self, detrended):
        batch_size, seq_len = detrended.shape
        seasonal = torch.zeros_like(detrended)
        for i in range(self.period):
            indices = torch.arange(i, seq_len, self.period, device=detrended.device)
            if len(indices) > 1:
                subseries = detrended[:, indices]
                seasonal_value = torch.median(subseries, dim=1)[0]
                seasonal[:, indices] = seasonal_value.unsqueeze(1)
        return seasonal
    
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        trends = torch.zeros_like(x)
        seasonals = torch.zeros_like(x)
        residuals = torch.zeros_like(x)
        for f in range(n_features):
            series = x[:, :, f]
            trend = self.extract_trend(series)
            for iteration in range(self.num_iterations):
                detrended = series - trend
                seasonal = self.extract_seasonal(detrended)
                deseasonalized = series - seasonal
                trend = self.extract_trend(deseasonalized)
            residual = series - trend - seasonal
            trends[:, :, f] = trend
            seasonals[:, :, f] = seasonal
            residuals[:, :, f] = residual
        return trends, seasonals, residuals
