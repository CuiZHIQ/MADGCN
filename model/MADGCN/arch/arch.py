import torch
import torch.nn as nn
import torch.nn.functional as F
from .madgcn_layers import PatchMixerBackbone, EnhancedSeasonalModule, SpatialGCN

class MADGCN(nn.Module):
    """
    MADGCN模型架构
    使用预定义的PPG和CAG邻接矩阵进行空间图卷积
    """
    def __init__(self, num_nodes, in_dim, out_dim, embed_dim, gcn_depth,
                 seq_length, horizon, layers, patch_len, stride,
                 predefined_A=None, predefined_CAG=None, dropout=0.3, 
                 cycle_len=24, gamma=0.1, fusion_alpha=0.7, 
                 d_model=128, mixer_kernel_size=8, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.cycle_len = cycle_len
        
        # 注册PPG邻接矩阵 (Physical Proximity Graph)
        if predefined_A is not None:
            self.register_buffer('ppg_matrix', torch.FloatTensor(predefined_A))
        else:
            self.ppg_matrix = None
            
        # 注册CAG邻接矩阵 (Causal-Aware Graph)  
        if predefined_CAG is not None:
            self.register_buffer('cag_matrix', torch.FloatTensor(predefined_CAG))
        else:
            # 如果没有提供CAG，使用PPG作为默认值
            self.cag_matrix = self.ppg_matrix
            
        # Enhanced Seasonal Enhancement Module
        self.seasonal_enhancer = EnhancedSeasonalModule(in_dim, cycle_len, gamma)
        
        # 空间图卷积网络 - PPG用于物理邻近性建模
        self.ppg_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        
        # 空间图卷积网络 - CAG用于因果关系建模
        self.cag_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        
        # 融合权重
        self.register_buffer('fusion_alpha', torch.tensor(fusion_alpha))
        
        # 时序混合器输入维度：两个GCN特征 + 趋势 + 增强的季节性
        temporal_input_dim = embed_dim * 2 + in_dim * 2
        
        # PatchMixer for temporal modeling
        self.temporal_mixer = PatchMixerBackbone(
            seq_len=seq_length,
            patch_len=patch_len, 
            stride=stride,
            in_channel=temporal_input_dim,
            d_model=d_model,
            n_layers=layers, 
            kernel_size=mixer_kernel_size,
            dropout=dropout
        )
        
        # 预测层
        self.prediction_layer = nn.Linear(d_model, horizon * out_dim)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        MADGCN前向传播
        Args:
            history_data: 历史数据 [batch_size, seq_len, num_nodes, in_dim]
            ppg_matrix: 物理邻近图矩阵 [num_nodes, num_nodes]
            cag_matrix: 因果感知图矩阵 [num_nodes, num_nodes]
            trends: 趋势分量 [batch_size, seq_len, num_nodes, in_dim]
            seasons: 季节性分量 [batch_size, seq_len, num_nodes, in_dim]
            residuals: 残差分量 [batch_size, seq_len, num_nodes, in_dim]
        """
        # 从kwargs中获取输入数据
        ppg_matrix = kwargs.get('ppg_matrix', self.ppg_matrix)
        cag_matrix = kwargs.get('cag_matrix', self.cag_matrix)
        trends = kwargs['trends']
        seasons = kwargs['seasons'] 
        residuals = kwargs['residuals']
        
        batch_size, seq_len, num_nodes, _ = history_data.shape
        
        # 1. 季节性增强
        cycle_index = torch.arange(batch_size, device=history_data.device) % self.cycle_len
        seasons_enh = self.seasonal_enhancer(seasons, cycle_index)
        
        # 2. 空间图卷积 - 使用预定义的邻接矩阵
        residuals_reshaped = residuals.view(batch_size * seq_len, num_nodes, -1)
        
        # PPG-based 空间建模
        if ppg_matrix is not None:
            features_ppg = self.ppg_gcn(residuals_reshaped, ppg_matrix)
        else:
            features_ppg = torch.zeros(batch_size * seq_len, num_nodes, self.embed_dim, 
                                     device=residuals.device)
        
        # CAG-based 因果关系建模
        if cag_matrix is not None:
            features_cag = self.cag_gcn(residuals_reshaped, cag_matrix)
        else:
            features_cag = torch.zeros(batch_size * seq_len, num_nodes, self.embed_dim,
                                     device=residuals.device)
        
        # 3. 特征融合
        features_ppg = features_ppg.view(batch_size, seq_len, num_nodes, -1)
        features_cag = features_cag.view(batch_size, seq_len, num_nodes, -1)
        fused_features = self.fusion_alpha * features_cag + (1 - self.fusion_alpha) * features_ppg
        
        # 4. 时序建模输入准备
        temporal_input = torch.cat([fused_features, trends, seasons_enh], dim=-1)
        temporal_input = temporal_input.permute(0, 2, 1, 3)  # [batch, nodes, seq, features]
        temporal_input = temporal_input.reshape(batch_size * num_nodes, seq_len, -1)
        
        # 5. 时序混合器
        temporal_output = self.temporal_mixer(temporal_input)  # [batch*nodes, d_model]
        
        # 6. 预测
        prediction = self.prediction_layer(temporal_output)  # [batch*nodes, horizon*out_dim]
        prediction = prediction.view(batch_size, num_nodes, self.horizon, self.out_dim)
        prediction = prediction.permute(0, 2, 1, 3)  # [batch, horizon, nodes, out_dim]
        
        return prediction 