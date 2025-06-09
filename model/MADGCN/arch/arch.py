import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PatchMixerBackbone, EnhancedSeasonalModule, SpatialGCN

class MADGCN(nn.Module):
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

        if predefined_A is not None:
            self.register_buffer('ppg_matrix', torch.FloatTensor(predefined_A))
        else:
            self.ppg_matrix = None
        
        if predefined_CAG is not None:
            self.register_buffer('cag_matrix', torch.FloatTensor(predefined_CAG))
        else:
            self.cag_matrix = self.ppg_matrix

        self.seasonal_enhancer = EnhancedSeasonalModule(in_dim, cycle_len, gamma)
        self.ppg_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        self.cag_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        self.register_buffer('fusion_alpha', torch.tensor(fusion_alpha))

        temporal_input_dim = embed_dim + in_dim + in_dim
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
        self.prediction_layer = nn.Linear(d_model, horizon * out_dim)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        ppg_matrix = kwargs.get('ppg_matrix', self.ppg_matrix)
        cag_matrix = kwargs.get('cag_matrix', self.cag_matrix)
        trends = kwargs['trends']
        seasons = kwargs['seasons'] 
        residuals = kwargs['residuals']
        
        batch_size, seq_len, num_nodes, _ = history_data.shape

        cycle_index = torch.arange(batch_size, device=history_data.device) % self.cycle_len
        seasons_enh = self.seasonal_enhancer(seasons, cycle_index)

        residuals_reshaped = residuals.reshape(batch_size * seq_len, num_nodes, -1)
        
        features_ppg = self.ppg_gcn(residuals_reshaped, ppg_matrix)
        features_cag = self.cag_gcn(residuals_reshaped, cag_matrix)
        
        features_ppg = features_ppg.view(batch_size, seq_len, num_nodes, -1)
        features_cag = features_cag.view(batch_size, seq_len, num_nodes, -1)
        fused_features = self.fusion_alpha * features_cag + (1 - self.fusion_alpha) * features_ppg
        
        temporal_input = torch.cat([fused_features, trends, seasons_enh], dim=-1)
        
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        
        temporal_output = self.temporal_mixer(temporal_input)
        
        prediction = self.prediction_layer(temporal_output)
        
        prediction = prediction.view(batch_size, num_nodes, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        
        return prediction 
