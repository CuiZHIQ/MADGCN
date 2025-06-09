import torch
import numpy as np
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import load_adj
from .madgcn_layers import STLDecomposition, EnhancedSeasonalModule

class MADGCNRunner(SimpleTimeSeriesForecastingRunner):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stl_decomposer = STLDecomposition(
            period=cfg.MODEL.PARAM.get("cycle_len", 24),
            num_iterations=2
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg.GPU else 'cpu')
        self.stl_decomposer = self.stl_decomposer.to(self.device)
        
        # 加载PPG邻接矩阵 (Physical Proximity Graph)
        if isinstance(cfg.MODEL.PARAM.get("predefined_A"), str):
            ppg_mx = load_adj(cfg.MODEL.PARAM["predefined_A"])
            self.ppg_mx = torch.FloatTensor(ppg_mx).to(self.device)
        else:
            self.ppg_mx = cfg.MODEL.PARAM.get("predefined_A")
            if self.ppg_mx is not None:
                self.ppg_mx = torch.FloatTensor(self.ppg_mx).to(self.device)
        
        # 加载CAG邻接矩阵 (Causal-Aware Graph) 
        if isinstance(cfg.MODEL.PARAM.get("predefined_CAG"), str):
            cag_mx = load_adj(cfg.MODEL.PARAM["predefined_CAG"])
            self.cag_mx = torch.FloatTensor(cag_mx).to(self.device)
        else:
            self.cag_mx = cfg.MODEL.PARAM.get("predefined_CAG")
            if self.cag_mx is not None:
                self.cag_mx = torch.FloatTensor(self.cag_mx).to(self.device)
            else:
                # 如果没有提供CAG，使用PPG作为默认值
                print("警告：未提供CAG矩阵，使用PPG作为默认CAG")
                self.cag_mx = self.ppg_mx
    
    def time_series_decomposition(self, data):
        """
        STL时间序列分解
        """
        batch_size, seq_len, num_nodes, num_features = data.shape
        data_reshaped = data.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, num_features)
        trends_raw, seasons_raw, residuals_raw = self.stl_decomposer(data_reshaped)
        trends = trends_raw.reshape(batch_size, num_nodes, seq_len, num_features).permute(0, 2, 1, 3)
        seasons = seasons_raw.reshape(batch_size, num_nodes, seq_len, num_features).permute(0, 2, 1, 3)
        residuals = residuals_raw.reshape(batch_size, num_nodes, seq_len, num_features).permute(0, 2, 1, 3)
        return trends, seasons, residuals

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, 
                train: bool = True, **kwargs) -> tuple:
        """
        完整的MADGCN前向传播流程
        """
        future_data, history_data = data
        history_data = history_data.to(self.device)
        future_data = future_data.to(self.device)
        
        # 1. 时间序列分解
        trends, seasons, residuals = self.time_series_decomposition(history_data)
        
        # 2. 准备模型输入，直接使用预定义的邻接矩阵
        model_kwargs = {
            "ppg_matrix": self.ppg_mx,     # Physical Proximity Graph
            "cag_matrix": self.cag_mx,     # Causal-Aware Graph  
            "trends": trends,
            "seasons": seasons,
            "residuals": residuals
        }
        
        # 3. 模型前向传播
        y_hat = self.model(
            history_data=history_data,
            future_data=future_data,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            **model_kwargs
        )
        return y_hat, future_data 