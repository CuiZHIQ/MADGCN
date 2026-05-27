import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_ppg(coords, tau=10.0, delta=100.0):
    N = len(coords)
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= delta:
                adj[i, j] = np.exp(-d**2 / tau**2)
    return adj


def cosine_similarity_matrix(x):
    x_norm = F.normalize(x, p=2, dim=-1)
    return torch.mm(x_norm, x_norm.t())


class GrangerCausality:
    def __init__(self, lag_r=12, lag_z=24):
        self.lag_r = lag_r
        self.lag_z = lag_z

    def _effective_lags(self, length):
        if length < 4:
            return 0, 0
        max_lag = max(1, (length - 2) // 3)
        return min(self.lag_r, max_lag), min(self.lag_z, max_lag)

    def _compute_single(self, residuals, meteorology):
        B, T, N, C = residuals.shape
        if meteorology is None:
            return torch.zeros(N, N, device=residuals.device)

        lag_r, lag_z = self._effective_lags(T)
        if lag_r == 0 or lag_z == 0:
            return torch.zeros(N, N, device=residuals.device)

        max_lag = max(lag_r, lag_z)
        causal_matrix = torch.zeros(N, N, device=residuals.device)
        R = residuals.mean(dim=0).mean(dim=-1)
        Z = meteorology.mean(dim=0)

        if T <= max_lag + 1:
            return torch.zeros(N, N, device=residuals.device)

        for i in range(N):
            for j in range(N):
                y = R[max_lag:, j]
                features = [torch.ones_like(y)]
                for lag in range(1, lag_r + 1):
                    features.append(R[max_lag - lag:T - lag, j])
                for lag in range(1, lag_z + 1):
                    features.append(Z[max_lag - lag:T - lag, i, :])

                X = torch.cat([
                    f.unsqueeze(-1) if f.dim() == 1 else f
                    for f in features
                ], dim=-1)
                if X.shape[0] == 0:
                    continue
                beta = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)
                causal_matrix[i, j] = beta[1 + lag_r:].abs().mean()

        max_val = causal_matrix.max()
        if max_val > 0:
            causal_matrix = causal_matrix / (max_val + 1e-8)
        return causal_matrix

    def compute(self, residuals, meteorology):
        return self._compute_single(residuals, meteorology)

    def compute_sequence(self, residuals, meteorology, min_history=4):
        B, T, N, _ = residuals.shape
        matrices = []
        last_valid = torch.zeros(N, N, device=residuals.device)
        for end in range(1, T + 1):
            if end < min_history:
                matrices.append(last_valid)
                continue
            cur = self._compute_single(residuals[:, :end], meteorology[:, :end])
            if torch.any(cur > 0):
                last_valid = cur
            matrices.append(last_valid)
        return torch.stack(matrices, dim=0)


class DynamicCAG(nn.Module):
    def __init__(self, num_nodes, w_trend=0.3, w_season=0.3, w_causal=0.4, 
                 lag_r=12, lag_z=24, learnable_weights=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            self._w_trend = nn.Parameter(torch.tensor(w_trend))
            self._w_season = nn.Parameter(torch.tensor(w_season))
            self._w_causal = nn.Parameter(torch.tensor(w_causal))
        else:
            self.register_buffer('_w_trend', torch.tensor(w_trend))
            self.register_buffer('_w_season', torch.tensor(w_season))
            self.register_buffer('_w_causal', torch.tensor(w_causal))
        
        self.granger = GrangerCausality(lag_r=lag_r, lag_z=lag_z)
    
    @property
    def weights(self):
        if self.learnable_weights:
            w_t = torch.softmax(torch.stack([self._w_trend, self._w_season, self._w_causal]), dim=0)
            return w_t[0], w_t[1], w_t[2]
        else:
            w_total = self._w_trend + self._w_season + self._w_causal
            return self._w_trend / w_total, self._w_season / w_total, self._w_causal / w_total
    
    def forward(self, trends, seasons, residuals, meteorology=None):
        B, T, N, C = trends.shape
        
        trend_flat = trends.mean(dim=0).mean(dim=-1).t()
        season_flat = seasons.mean(dim=0).mean(dim=-1).t()
        
        sim_trend = cosine_similarity_matrix(trend_flat)
        sim_season = cosine_similarity_matrix(season_flat)
        
        if meteorology is not None:
            causal_strength = self.granger.compute_sequence(residuals, meteorology)
        else:
            causal_strength = torch.eye(N, device=trends.device).unsqueeze(0).expand(T, -1, -1) * 0.5
        
        w_t, w_s, w_c = self.weights
        static_similarity = w_t * sim_trend + w_s * sim_season
        cag = static_similarity.unsqueeze(0) + w_c * causal_strength
        cag = cag.clamp(0, 1)
        eye = torch.eye(N, device=trends.device).unsqueeze(0)
        cag = cag * (1 - eye) + eye
        
        return cag
