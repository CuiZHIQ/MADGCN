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
    
    def compute(self, residuals, meteorology):
        B, T, N, C = residuals.shape
        if meteorology is None:
            return torch.zeros(N, N, device=residuals.device)
        
        max_lag = max(self.lag_r, self.lag_z)
        if T <= max_lag + 1:
            return torch.zeros(N, N, device=residuals.device)
        
        causal_matrix = torch.zeros(N, N, device=residuals.device)
        R = residuals.mean(dim=0).squeeze(-1)
        Z = meteorology.mean(dim=0).mean(dim=-1)
        
        valid_len = T - max_lag
        if valid_len < self.lag_r + self.lag_z + 1:
            return torch.zeros(N, N, device=residuals.device)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    try:
                        y = R[max_lag:, j]
                        features = []
                        for lag in range(1, self.lag_r + 1):
                            features.append(R[max_lag - lag:T - lag, j])
                        for lag in range(1, self.lag_z + 1):
                            features.append(Z[max_lag - lag:T - lag, i])
                        
                        min_len = min(len(y), min(len(f) for f in features))
                        if min_len > 0:
                            X = torch.stack([f[:min_len] for f in features], dim=1)
                            y = y[:min_len]
                            beta = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution
                            causal_matrix[i, j] = beta[self.lag_r:].abs().mean()
                    except:
                        pass
        
        max_val = causal_matrix.max()
        if max_val > 0:
            causal_matrix = causal_matrix / (max_val + 1e-8)
        return causal_matrix


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
        
        trend_flat = trends.mean(dim=0).squeeze(-1).t()
        season_flat = seasons.mean(dim=0).squeeze(-1).t()
        
        sim_trend = cosine_similarity_matrix(trend_flat)
        sim_season = cosine_similarity_matrix(season_flat)
        
        if meteorology is not None:
            causal_strength = self.granger.compute(residuals, meteorology)
        else:
            causal_strength = torch.eye(N, device=trends.device) * 0.5
        
        w_t, w_s, w_c = self.weights
        cag = w_t * sim_trend + w_s * sim_season + w_c * causal_strength
        
        cag = (cag + cag.t()) / 2
        cag = cag.clamp(0, 1)
        
        return cag
