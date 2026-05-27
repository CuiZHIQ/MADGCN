import argparse
import json
from pathlib import Path

import numpy as np

from cag import cosine_similarity, decompose, lagged_corr_causality, load_windows


def effective_lag(length, max_lag):
    if length < 4:
        return 0
    return min(max_lag, max(1, (length - 2) // 3))


def build_dynamic_cag_for_window(window, period=24, max_lag=6, min_history=4,
                                 w_trend=0.3, w_season=0.3, w_causal=0.4):
    target = window[..., :1]
    covariates = window[..., 1:]
    if covariates.shape[-1] == 0:
        raise ValueError("Dynamic CAG construction needs at least one covariate channel.")

    trend, season, residual = decompose(target, period=period)
    trend = trend[..., 0]
    season = season[..., 0]
    residual = residual[..., 0]

    length, nodes = residual.shape
    out = np.zeros((length, nodes, nodes), dtype=np.float32)
    last = np.eye(nodes, dtype=np.float32)

    for end in range(1, length + 1):
        lag = effective_lag(end, max_lag)
        if end < min_history or lag == 0:
            out[end - 1] = last
            continue

        trend_sim = cosine_similarity(trend[:end].T)
        season_sim = cosine_similarity(season[:end].T)
        causal = lagged_corr_causality(
            residual[None, :end],
            covariates[None, :end],
            lag,
        )
        cag = w_trend * trend_sim + w_season * season_sim + w_causal * causal
        cag = np.clip(cag, 0, 1).astype(np.float32)
        np.fill_diagonal(cag, 1.0)
        last = cag
        out[end - 1] = cag

    return out


def compute_dynamic_cag(args):
    windows, valid = load_windows(
        args.data_dir,
        args.seq_len,
        args.horizon,
        args.samples,
        args.input_dim,
    )
    matrices = [
        build_dynamic_cag_for_window(
            window,
            period=args.period,
            max_lag=args.max_lag,
            min_history=args.min_history,
            w_trend=args.w_trend,
            w_season=args.w_season,
            w_causal=args.w_causal,
        )
        for window in windows
    ]
    matrices = np.stack(matrices, axis=0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.reduce == "mean":
        saved = matrices.mean(axis=0).astype(np.float32)
        np.save(output, saved)
    else:
        saved = matrices.astype(np.float32)
        np.savez_compressed(output, cag=saved, indices=valid)

    return {
        "output": str(output),
        "shape": list(saved.shape),
        "samples": int(len(valid)),
        "min": float(saved.min()),
        "max": float(saved.max()),
        "mean": float(saved.mean()),
        "asym_mean_abs": float(np.abs(saved - np.swapaxes(saved, -1, -2)).mean()),
        "reduce": args.reduce,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute dynamic CAG matrices for MADGCN.")
    parser.add_argument("--data-dir", required=True, help="Directory containing his.npz and idx_train.npy.")
    parser.add_argument("--output", required=True, help="Output .npy or .npz path.")
    parser.add_argument("--input-dim", type=int, default=None, help="Optional number of channels to keep.")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--w-trend", type=float, default=0.3)
    parser.add_argument("--w-season", type=float, default=0.3)
    parser.add_argument("--w-causal", type=float, default=0.4)
    parser.add_argument("--reduce", choices=["mean", "none"], default="mean")
    return parser.parse_args()


def main():
    stats = compute_dynamic_cag(parse_args())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
