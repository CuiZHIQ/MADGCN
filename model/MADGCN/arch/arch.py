import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SpatialGCN, RecurrentCycle, PatchMixerLayer

class EnhancedSeasonalModule(nn.Module):
    def __init__(self, in_dim, cycle_len=24, gamma=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.cycle_len = cycle_len
        self.recurrent_cycle = RecurrentCycle(cycle_len, in_dim)
        self.w_c = nn.Linear(in_dim, in_dim)
        self.b_c = nn.Parameter(torch.zeros(in_dim))
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, seasonal_component, cycle_index):
        batch_size, seq_len, num_nodes, _ = seasonal_component.shape
        cycle_patterns = self.recurrent_cycle(cycle_index, seq_len)
        cycle_patterns = cycle_patterns.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        combined = self.w_c(seasonal_component * cycle_patterns) + self.b_c
        enhanced = seasonal_component + self.gamma * self.norm(combined)
        return enhanced

class GC_Generator(nn.Module):
    def __init__(self, in_dim, num_nodes, embed_dim=16):
        super().__init__()
        self.node_embedding_mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        node_features = torch.mean(residuals, dim=[0, 1])
        node_embeds = self.node_embedding_mlp(node_features)
        adj = F.relu(torch.tanh(node_embeds @ node_embeds.T))
        return adj

class DualGraphGCN(nn.Module):
    def __init__(self, in_dim, embed_dim, dropout, gcn_depth, fusion_alpha=0.7):
        super().__init__()
        self.ppg_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        self.cag_gcn = SpatialGCN(in_dim, embed_dim, dropout, order=gcn_depth)
        self.register_buffer('fusion_alpha', torch.tensor(fusion_alpha))

    def forward(self, residuals: torch.Tensor, ppg_matrix: torch.Tensor, cag_matrix: torch.Tensor) -> torch.Tensor:
        features_ppg = self.ppg_gcn(residuals, ppg_matrix)
        features_cag = self.cag_gcn(residuals, cag_matrix)
        fused_features = self.fusion_alpha * features_cag + (1 - self.fusion_alpha) * features_ppg
        return fused_features

class PatchMixerBackbone(nn.Module):
    def __init__(self, seq_len, patch_len, stride, in_channel, d_model=128, 
                 n_layers=3, kernel_size=8, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        self.patch_embedding = nn.Linear(patch_len * in_channel, d_model)
        self.mixer_layers = nn.ModuleList([
            PatchMixerLayer(d_model, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size = x.shape[0]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch_size, self.patch_num, -1)
        x = self.patch_embedding(patches)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return x

class MADGCN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, embed_dim, gcn_depth,
                 seq_length, horizon, layers, patch_len, stride,
                 predefined_A=None, dropout=0.3, 
                 cycle_len=24, gamma=0.1, fusion_alpha=0.7, 
                 d_model=128, mixer_kernel_size=8, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.horizon = horizon
        
        if predefined_A is not None:
            self.register_buffer('ppg_matrix', torch.FloatTensor(predefined_A))
        else:
            self.ppg_matrix = None

        self.seasonal_enhancer = EnhancedSeasonalModule(in_dim, cycle_len, gamma)
        self.gc_generator = GC_Generator(in_dim, num_nodes, embed_dim=embed_dim)
        self.dual_gcn = DualGraphGCN(in_dim, embed_dim, dropout, gcn_depth, fusion_alpha)
        
        temporal_input_dim = embed_dim + in_dim * 2
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
        trends = kwargs['trends']
        seasons = kwargs['seasons'] 
        residuals = kwargs['residuals']
        batch_size, seq_len, num_nodes, _ = history_data.shape
        seasons_enh = self.seasonal_enhancer(seasons, kwargs['cycle_index'])
        cag_matrix = self.gc_generator(residuals)
        residuals_reshaped = residuals.reshape(batch_size * seq_len, num_nodes, -1)
        gcn_output = self.dual_gcn(residuals_reshaped, self.ppg_matrix, cag_matrix)
        gcn_output = gcn_output.view(batch_size, seq_len, num_nodes, -1)
        temporal_input = torch.cat([gcn_output, trends, seasons_enh], dim=-1)
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        temporal_output = self.temporal_mixer(temporal_input)
        prediction = self.prediction_layer(temporal_output)
        prediction = prediction.view(batch_size, num_nodes, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        return prediction 
