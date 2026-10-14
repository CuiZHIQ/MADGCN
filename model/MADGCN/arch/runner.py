import torch
import torch.nn as nn
from .layers import STLDecomposition, EnhancedSeasonalModule
from .graph import DynamicCAG


class MADGCNRunner(nn.Module):
    def __init__(self, model, num_nodes, in_dim=1, cycle_len=24, gamma=0.1, 
                 stl_iterations=2, lag_r=12, lag_z=24, 
                 w_trend=0.3, w_season=0.3, w_causal=0.4, learnable_cag_weights=True,
                 device='cuda', use_dynamic_cag=False, adaptive_period=False):
        super().__init__()
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.num_nodes = num_nodes
        self.cycle_len = cycle_len
        self.use_dynamic_cag = use_dynamic_cag
        
        self.stl_decomposer = STLDecomposition(
            period=cycle_len, 
            num_iterations=stl_iterations,
            adaptive_period=adaptive_period
        ).to(device)
        
        self.seasonal_enhancer = EnhancedSeasonalModule(
            in_dim, 
            cycle_len, 
            gamma
        ).to(device)
        
        self.register_buffer('ppg_mx', torch.eye(num_nodes, device=device))
        self.register_buffer('cag_mx', torch.eye(num_nodes, device=device))
        
        if use_dynamic_cag:
            self.dynamic_cag = DynamicCAG(
                num_nodes, 
                w_trend=w_trend, 
                w_season=w_season, 
                w_causal=w_causal,
                lag_r=lag_r, 
                lag_z=lag_z,
                learnable_weights=learnable_cag_weights
            ).to(device)

    @property
    def device(self):
        return self.ppg_mx.device
    
    def set_adj(self, ppg_mx, cag_mx=None):
        self.ppg_mx = torch.as_tensor(ppg_mx, dtype=torch.float32, device=self.device)
        if cag_mx is not None:
            self.cag_mx = torch.as_tensor(cag_mx, dtype=torch.float32, device=self.device).transpose(-1, -2).contiguous()
    
    def time_series_decomposition(self, data):
        B, T, N, C = data.shape
        data_reshaped = data.permute(0, 2, 1, 3).reshape(B * N, T, C)
        trends, seasons, residuals = self.stl_decomposer(data_reshaped)
        trends = trends.reshape(B, N, T, C).permute(0, 2, 1, 3)
        seasons = seasons.reshape(B, N, T, C).permute(0, 2, 1, 3)
        residuals = residuals.reshape(B, N, T, C).permute(0, 2, 1, 3)
        return trends, seasons, residuals

    def forward(self, history_data, future_data=None, meteorology=None, cycle_index=None):
        history_data = history_data.to(self.device)
        if future_data is not None:
            future_data = future_data.to(self.device)
        if meteorology is not None:
            meteorology = meteorology.to(self.device)
            if meteorology.shape[:3] != history_data.shape[:3]:
                raise ValueError('Meteorology must contain only the matching historical window')
        
        trends, seasons, residuals = self.time_series_decomposition(history_data)
        batch_size = history_data.shape[0]
        if cycle_index is None:
            cycle_index = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            cycle_index = torch.as_tensor(cycle_index, dtype=torch.long, device=self.device).expand(batch_size)
        seasons_enh = self.seasonal_enhancer(seasons, cycle_index)
        
        if self.use_dynamic_cag:
            cag_matrix = self.dynamic_cag(trends, seasons_enh, residuals, meteorology).transpose(-1, -2)
        else:
            cag_matrix = self.cag_mx
        
        prediction = self.model(
            history_data=history_data,
            ppg_matrix=self.ppg_mx,
            cag_matrix=cag_matrix,
            trends=trends,
            seasons=seasons,
            seasons_enh=seasons_enh,
            residuals=residuals,
            meteorology=meteorology
        )
        return prediction, future_data
    
    def train_step(self, history_data, future_data, optimizer, criterion, meteorology=None, cycle_index=None):
        self.train()
        optimizer.zero_grad()
        prediction, target = self(history_data, future_data, meteorology, cycle_index)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def eval_step(self, history_data, future_data, criterion, meteorology=None, cycle_index=None):
        self.eval()
        with torch.no_grad():
            prediction, target = self(history_data, future_data, meteorology, cycle_index)
            loss = criterion(prediction, target)
        return loss.item(), prediction, target
    
    def get_trainable_params(self):
        return list(self.parameters())
    
    def get_config(self):
        config = {
            'num_nodes': self.num_nodes,
            'cycle_len': self.cycle_len,
            'use_dynamic_cag': self.use_dynamic_cag,
            'stl_iterations': self.stl_decomposer.num_iterations,
            'adaptive_period': self.stl_decomposer.adaptive_period,
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
