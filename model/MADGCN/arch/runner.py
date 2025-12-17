import torch
import torch.nn as nn
from .layers import STLDecomposition, EnhancedSeasonalModule
from .graph import DynamicCAG, build_ppg


class MADGCNRunner:
    def __init__(self, model, num_nodes, in_dim=1, cycle_len=24, gamma=0.1, 
                 stl_iterations=2, lag_r=12, lag_z=24, 
                 w_trend=0.3, w_season=0.3, w_causal=0.4, learnable_cag_weights=True,
                 device='cuda', use_dynamic_cag=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.num_nodes = num_nodes
        self.cycle_len = cycle_len
        self.use_dynamic_cag = use_dynamic_cag
        
        self.stl_decomposer = STLDecomposition(
            period=cycle_len, 
            num_iterations=stl_iterations
        ).to(self.device)
        
        self.seasonal_enhancer = EnhancedSeasonalModule(
            in_dim, 
            cycle_len, 
            gamma
        ).to(self.device)
        
        self.ppg_mx = torch.eye(num_nodes).to(self.device)
        self.cag_mx = torch.eye(num_nodes).to(self.device)
        
        if use_dynamic_cag:
            self.dynamic_cag = DynamicCAG(
                num_nodes, 
                w_trend=w_trend, 
                w_season=w_season, 
                w_causal=w_causal,
                lag_r=lag_r, 
                lag_z=lag_z,
                learnable_weights=learnable_cag_weights
            ).to(self.device)
    
    def set_adj(self, ppg_mx, cag_mx=None):
        if isinstance(ppg_mx, np.ndarray):
            ppg_mx = torch.from_numpy(ppg_mx).float()
        self.ppg_mx = ppg_mx.to(self.device)
        if cag_mx is not None:
            if isinstance(cag_mx, np.ndarray):
                cag_mx = torch.from_numpy(cag_mx).float()
            self.cag_mx = cag_mx.to(self.device)
    
    def time_series_decomposition(self, data):
        B, T, N, C = data.shape
        data_reshaped = data.permute(0, 2, 1, 3).reshape(B * N, T, C)
        trends, seasons, residuals = self.stl_decomposer(data_reshaped)
        trends = trends.reshape(B, N, T, C).permute(0, 2, 1, 3)
        seasons = seasons.reshape(B, N, T, C).permute(0, 2, 1, 3)
        residuals = residuals.reshape(B, N, T, C).permute(0, 2, 1, 3)
        return trends, seasons, residuals

    def forward(self, history_data, future_data=None, meteorology=None):
        history_data = history_data.to(self.device)
        if future_data is not None:
            future_data = future_data.to(self.device)
        if meteorology is not None:
            meteorology = meteorology.to(self.device)
        
        trends, seasons, residuals = self.time_series_decomposition(history_data)
        batch_size = history_data.shape[0]
        cycle_index = torch.arange(batch_size, device=self.device) % self.cycle_len
        seasons_enh = self.seasonal_enhancer(seasons, cycle_index)
        
        if self.use_dynamic_cag:
            cag_matrix = self.dynamic_cag(trends, seasons_enh, residuals, meteorology)
        else:
            cag_matrix = self.cag_mx
        
        prediction = self.model(
            history_data=history_data,
            ppg_matrix=self.ppg_mx,
            cag_matrix=cag_matrix,
            trends=trends,
            seasons=seasons,
            seasons_enh=seasons_enh,
            residuals=residuals
        )
        return prediction, future_data
    
    def train_step(self, history_data, future_data, optimizer, criterion, meteorology=None):
        self.model.train()
        if self.use_dynamic_cag:
            self.dynamic_cag.train()
        optimizer.zero_grad()
        prediction, target = self.forward(history_data, future_data, meteorology)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def eval_step(self, history_data, future_data, criterion, meteorology=None):
        self.model.eval()
        if self.use_dynamic_cag:
            self.dynamic_cag.eval()
        with torch.no_grad():
            prediction, target = self.forward(history_data, future_data, meteorology)
            loss = criterion(prediction, target)
        return loss.item(), prediction, target
    
    def get_trainable_params(self):
        params = list(self.model.parameters())
        params += list(self.stl_decomposer.parameters())
        params += list(self.seasonal_enhancer.parameters())
        if self.use_dynamic_cag:
            params += list(self.dynamic_cag.parameters())
        return params
    
    def get_config(self):
        config = {
            'num_nodes': self.num_nodes,
            'cycle_len': self.cycle_len,
            'use_dynamic_cag': self.use_dynamic_cag,
            'stl_iterations': self.stl_decomposer.num_iterations,
            'seasonal_gamma': self.seasonal_enhancer.gamma,
        }
        if self.use_dynamic_cag:
            w_t, w_s, w_c = self.dynamic_cag.weights
            config.update({
                'cag_w_trend': w_t.item(),
                'cag_w_season': w_s.item(),
                'cag_w_causal': w_c.item(),
                'lag_r': self.dynamic_cag.granger.lag_r,
                'lag_z': self.dynamic_cag.granger.lag_z,
            })
        return config


try:
    import numpy as np
except ImportError:
    np = None
