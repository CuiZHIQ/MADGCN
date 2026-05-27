import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class STLDecomposition(nn.Module):
    def __init__(self, period=24, num_iterations=2, adaptive_period=True):
        super().__init__()
        self.period = period
        self.num_iterations = num_iterations
        self.adaptive_period = adaptive_period

    def _select_period(self, x_flat, length):
        max_period = min(self.period, max(2, length // 2))
        if not self.adaptive_period or max_period < 3:
            return min(self.period, length)

        centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
        var = centered.pow(2).mean().clamp(min=1e-8)
        scores = []
        candidates = range(2, max_period + 1)
        for lag in candidates:
            corr = (centered[:, :, lag:] * centered[:, :, :-lag]).mean() / var
            scores.append(corr)
        best = torch.stack(scores).argmax().item()
        return list(candidates)[best]

    def _moving_average(self, x, period):
        kernel_size = max(3, period if period % 2 == 1 else period + 1)
        padding = kernel_size // 2
        x = F.pad(x, (padding, padding), mode='replicate')
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, 1, T)
        trend = torch.zeros_like(x_flat)
        seasonal = torch.zeros_like(x_flat)
        period = self._select_period(x_flat, T)
        
        for _ in range(self.num_iterations):
            detrended = x_flat - trend
            new_seasonal = torch.zeros_like(x_flat)
            for i in range(period):
                indices = torch.arange(i, T, period, device=x.device)
                if len(indices) > 0:
                    new_seasonal[:, :, indices] = detrended[:, :, indices].mean(dim=-1, keepdim=True)
            seasonal = new_seasonal
            deseasonalized = x_flat - seasonal
            trend = self._moving_average(deseasonalized, period)
            if trend.shape[-1] != T:
                trend = F.interpolate(trend, size=T, mode='linear', align_corners=False)
        
        residual = x_flat - trend - seasonal
        trend = trend.reshape(B, C, T).permute(0, 2, 1)
        seasonal = seasonal.reshape(B, C, T).permute(0, 2, 1)
        residual = residual.reshape(B, C, T).permute(0, 2, 1)
        return trend, seasonal, residual


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel = channel
        self.cycle_weight = nn.Parameter(torch.randn(cycle_len, channel) * 0.01)
        
    def forward(self, x, cycle_index):
        B, T, N, C = x.shape
        cycle_emb = self.cycle_weight[cycle_index % self.cycle_len]
        cycle_emb = cycle_emb.view(B, 1, 1, C).expand(-1, T, N, -1)
        return x * (1 + cycle_emb)


class EnhancedSeasonalModule(nn.Module):
    def __init__(self, in_dim, cycle_len, gamma=0.1):
        super().__init__()
        self.gamma = gamma
        self.cycle_conv = RecurrentCycle(cycle_len, in_dim)
        self.W_c = nn.Linear(in_dim, in_dim)
        
    def forward(self, seasonal, cycle_index):
        cycle_out = self.cycle_conv(seasonal, cycle_index)
        enhanced = self.W_c(cycle_out)
        return seasonal + self.gamma * enhanced


class SpatialGCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, order=2):
        super().__init__()
        self.order = order
        self.linear = nn.Linear(in_dim * (order + 1), out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def normalize_adj(self, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return adj / deg
        
    def forward(self, x, adj):
        adj = self.normalize_adj(adj)
        out = [x]
        h = x
        for _ in range(self.order):
            if adj.dim() == 2:
                h = torch.matmul(adj, h)
            else:
                if adj.dim() == 3:
                    repeat = h.shape[0] // adj.shape[0]
                    adj_batch = adj.repeat(repeat, 1, 1)
                elif adj.dim() == 4:
                    adj_batch = adj.reshape(-1, adj.shape[-2], adj.shape[-1])
                else:
                    raise ValueError(f"Unsupported adjacency shape: {tuple(adj.shape)}")
                h = torch.bmm(adj_batch, h)
            out.append(h)
        out = torch.cat(out, dim=-1)
        out = self.dropout(F.relu(self.linear(out)))
        return out


class PatchMixerLayer(nn.Module):
    def __init__(self, patch_num, d_model, kernel_size=8):
        super().__init__()
        self.patch_norm = nn.LayerNorm(d_model)
        self.inter_patch = nn.Sequential(
            nn.Conv1d(patch_num, patch_num, kernel_size, padding='same', groups=patch_num),
            nn.GELU(),
        )
        self.inter_proj = nn.Sequential(
            nn.Conv1d(patch_num, patch_num, 1),
            nn.GELU(),
        )
        self.feature_norm = nn.LayerNorm(patch_num)
        self.intra_patch = nn.Sequential(
            nn.Linear(patch_num, patch_num),
            nn.GELU(),
        )
        
    def forward(self, x):
        h = self.patch_norm(x)
        h = self.inter_patch(h)
        h = x + h
        h_s = self.inter_proj(h)
        h_t = h_s.permute(0, 2, 1)
        h_t = self.feature_norm(h_t)
        h_t = self.intra_patch(h_t)
        h_t = h_t.permute(0, 2, 1)
        out = h_s + h_t
        return out


class PatchMixerBackbone(nn.Module):
    def __init__(self, seq_len, patch_len, stride, in_channel, d_model, n_layers, kernel_size=8, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_channel = in_channel
        
        self.padding = nn.ReplicationPad1d((0, stride))
        self.patch_num = int((seq_len - patch_len) / stride + 1) + 1
        self.W_P = nn.Linear(patch_len * in_channel, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.mixer_layers = nn.ModuleList([
            PatchMixerLayer(self.patch_num, d_model, kernel_size) for _ in range(n_layers)
        ])
        
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * d_model, d_model)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        BN, T, C = x.shape
        
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(BN, self.patch_num, -1)
        x = self.W_P(x)
        x = self.dropout(x)
        
        u = self.head0(x)
        
        for layer in self.mixer_layers:
            x = layer(x)
        
        out = self.head1(x) + u
        out = self.final_norm(out)
        return out
