import torch
import torch.nn as nn
from .layers import PatchMixerBackbone, SpatialGCN, RevIN


class MADGCN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, embed_dim, gcn_depth,
                 seq_length, horizon, layers, patch_len, stride,
                 dropout=0.3, fusion_alpha=0.7, learnable_alpha=False,
                 d_model=128, mixer_kernel_size=8, use_revin=True, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.horizon = horizon
        self.learnable_alpha = learnable_alpha
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(in_dim)

        self.ppg_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        self.cag_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        
        if learnable_alpha:
            self._alpha = nn.Parameter(torch.tensor(fusion_alpha))
        else:
            self.register_buffer('_alpha', torch.tensor(fusion_alpha))

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

    @property
    def fusion_alpha(self):
        return torch.sigmoid(self._alpha) if self.learnable_alpha else self._alpha

    def forward(self, history_data, ppg_matrix, cag_matrix, trends, seasons, residuals, seasons_enh=None, **kwargs):
        batch_size, seq_len, num_nodes, _ = history_data.shape
        if seasons_enh is None:
            seasons_enh = seasons
        
        if self.use_revin:
            history_flat = history_data.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
            history_flat = self.revin(history_flat, 'norm')
        
        residuals_reshaped = residuals.reshape(batch_size * seq_len, num_nodes, -1)
        features_ppg = self.ppg_gcn(residuals_reshaped, ppg_matrix)
        features_cag = self.cag_gcn(residuals_reshaped, cag_matrix)
        features_ppg = features_ppg.view(batch_size, seq_len, num_nodes, -1)
        features_cag = features_cag.view(batch_size, seq_len, num_nodes, -1)
        
        alpha = self.fusion_alpha
        fused_features = alpha * features_cag + (1 - alpha) * features_ppg
        
        temporal_input = torch.cat([fused_features, trends, seasons_enh], dim=-1)
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        temporal_output = self.temporal_mixer(temporal_input)
        prediction = self.prediction_layer(temporal_output)
        prediction = prediction.view(batch_size * num_nodes, self.horizon, self.out_dim)
        
        if self.use_revin:
            prediction = self.revin(prediction, 'denorm')
        
        prediction = prediction.view(batch_size, num_nodes, self.horizon, self.out_dim)
        prediction = prediction.permute(0, 2, 1, 3)
        return prediction
