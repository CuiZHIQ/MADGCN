import argparse
import json
from pathlib import Path

import numpy as np


def moving_average(x, kernel):
    pad_left = kernel // 2
    pad_right = kernel - 1 - pad_left
    padded = np.pad(x, ((pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        out[t] = padded[t:t + kernel].mean(axis=0)
    return out


def decompose(batch, period=24):
    period = min(period, max(2, batch.shape[0] // 2))
    kernel = period if period % 2 == 1 else period + 1
    trend = moving_average(batch, kernel)
    detrended = batch - trend[: batch.shape[0]]

    seasonal = np.zeros_like(batch)
    for offset in range(period):
        values = detrended[offset::period]
        if len(values):
            seasonal[offset::period] = values.mean(axis=0, keepdims=True)

    residual = batch - trend[: batch.shape[0]] - seasonal
    return trend[: batch.shape[0]], seasonal, residual


def cosine_similarity(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.clip(norm, 1e-8, None)
    return x @ x.T


def lagged_corr_causality(residual, covariates, max_lag):
    _, _, nodes = residual.shape
    features = covariates.shape[-1]
    cag = np.zeros((nodes, nodes), dtype=np.float32)

    for lag in range(1, max_lag + 1):
        dst = residual[:, lag:].reshape(-1, nodes)
        dst = dst - dst.mean(axis=0, keepdims=True)
        dst_norm = np.sqrt((dst ** 2).sum(axis=0))

        for feature in range(features):
            src = covariates[:, :-lag, :, feature].reshape(-1, nodes)
            src = src - src.mean(axis=0, keepdims=True)
            denom = np.clip(
                np.sqrt((src ** 2).sum(axis=0))[:, None] * dst_norm[None, :],
                1e-8,
                None,
            )
            cag += np.abs(src.T @ dst / denom)

    cag /= max_lag * features
    max_value = cag.max()
    if max_value > 0:
        cag /= max_value
    return cag


def load_windows(data_dir, seq_len, horizon, samples, input_dim):
    data_dir = Path(data_dir)
    data = np.load(data_dir / "his.npz")["data"].astype(np.float32)
    if input_dim is not None:
        data = data[..., :input_dim]

    idx = np.load(data_dir / "idx_train.npy")
    valid = idx[(idx >= seq_len - 1) & (idx + horizon < data.shape[0])]
    valid = valid[:samples]

    x_offsets = np.arange(-(seq_len - 1), 1)
    return np.stack([data[i + x_offsets] for i in valid], axis=0), valid


def build_cag_from_windows(windows, period=24, max_lag=6, w_trend=0.3, w_season=0.3, w_causal=0.4):
    target = windows[..., :1]
    covariates = windows[..., 1:]
    if covariates.shape[-1] == 0:
        raise ValueError("CAG construction needs at least one covariate channel.")

    trends, seasons, residuals = [], [], []
    for sample in target:
        trend, season, residual = decompose(sample, period=period)
        trends.append(trend[..., 0])
        seasons.append(season[..., 0])
        residuals.append(residual[..., 0])

    trends = np.stack(trends, axis=0)
    seasons = np.stack(seasons, axis=0)
    residuals = np.stack(residuals, axis=0)

    trend_sim = cosine_similarity(trends.mean(axis=0).T)
    season_sim = cosine_similarity(seasons.mean(axis=0).T)
    causal = lagged_corr_causality(residuals, covariates, max_lag)

    cag = w_trend * trend_sim + w_season * season_sim + w_causal * causal
    cag = np.clip(cag, 0, 1).astype(np.float32)
    np.fill_diagonal(cag, 1.0)
    return cag


def compute_static_cag(args):
    windows, valid = load_windows(
        args.data_dir,
        args.seq_len,
        args.horizon,
        args.samples,
        args.input_dim,
    )
    cag = build_cag_from_windows(
        windows,
        period=args.period,
        max_lag=args.max_lag,
        w_trend=args.w_trend,
        w_season=args.w_season,
        w_causal=args.w_causal,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, cag)

    return {
        "output": str(output),
        "shape": list(cag.shape),
        "samples": int(len(valid)),
        "min": float(cag.min()),
        "max": float(cag.max()),
        "mean": float(cag.mean()),
        "asym_mean_abs": float(np.abs(cag - cag.T).mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute a static CAG matrix for MADGCN.")
    parser.add_argument("--data-dir", required=True, help="Directory containing his.npz and idx_train.npy.")
    parser.add_argument("--output", required=True, help="Output .npy path for the static CAG matrix.")
    parser.add_argument("--input-dim", type=int, default=None, help="Optional number of channels to keep.")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--w-trend", type=float, default=0.3)
    parser.add_argument("--w-season", type=float, default=0.3)
    parser.add_argument("--w-causal", type=float, default=0.4)
    return parser.parse_args()


def main():
    stats = compute_static_cag(parse_args())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
