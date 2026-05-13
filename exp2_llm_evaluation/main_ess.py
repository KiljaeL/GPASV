import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from config import HF_TOKEN
from graph import MODELS, build_edges, build_pairwise_matrices, run_ranking
from load_data import load_chatbot_arena_df, load_mtbench_single_df, prepare_benchmark_data
from sampler import _logsumexp, build_adjacency, log_unnormalized_pmf

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for MH ranking sampler (used if --param-pairs not set)")
parser.add_argument("--beta", type=float, default=1.0, help="Beta for MH ranking sampler (used if --param-pairs not set)")
parser.add_argument(
    "--param-pairs",
    type=str,
    nargs="+",
    default=None,
    metavar="A,B",
    help='One or more "alpha,beta" pairs, e.g. "1,1" "1,0" "0.5,1".',
)
parser.add_argument("--n-samples", type=int, default=1000, help="Number of permutations to draw per param pair")
parser.add_argument("--seed", type=int, default=42, help="Random seed; matched with main_subset.py / main_value.py")
args = parser.parse_args()


def parse_param_pairs():
    if args.param_pairs is not None:
        out = []
        for s in args.param_pairs:
            parts = s.strip().split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"Each --param-pairs value must be 'alpha,beta'; got {s!r}")
            a = float(parts[0].strip().strip(","))
            b = float(parts[1].strip().strip(","))
            out.append((a, b))
        return out
    return [(args.alpha, args.beta)]


def format_param_tag(alpha, beta):
    return f"a{alpha:g}_b{beta:g}".replace("/", "_")


def pair_label(pair):
    return format_param_tag(pair[0], pair[1])


def build_lam_and_out_adj(df_pairwise):
    W, T, models, _ = build_pairwise_matrices(df_pairwise)
    edges = build_edges(W, T, models)
    out_adj, _ = build_adjacency(len(models), edges)
    lam = [
        0 if m in ["gpt-4", "gpt-3.5-turbo", "claude-v1", "claude-instant-v1", "palm-2"] else 1
        for m in models
    ]
    return lam, out_adj


def log_ess_from_log_weights(log_weights):
    """Stable ESS = (sum w)^2 / sum w^2 from log-weights."""
    if not log_weights:
        return 0.0
    lse_1 = _logsumexp(log_weights)
    lse_2 = _logsumexp([2.0 * lw for lw in log_weights])
    if lse_1 == -np.inf or lse_2 == -np.inf:
        return 0.0
    return float(np.exp(2.0 * lse_1 - lse_2))


def ess_from_weights(weights):
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(weights ** 2)
    if weights.size == 0 or denom <= 0.0:
        return 0.0
    numer = np.sum(weights) ** 2
    return float(numer / denom)


def psis_smooth_weights_from_log_weights(log_weights):
    """
    Simple PSIS-style smoothing:
    - stabilize by shifting raw log-weights
    - fit a GPD to the largest tail weights
    - replace tail weights by fitted tail quantiles
    Returns:
        smoothed_weights, k_hat
    """
    log_weights = np.asarray(log_weights, dtype=float)
    n = log_weights.size
    if n == 0:
        return np.array([], dtype=float), np.nan

    max_logw = np.max(log_weights)
    if not np.isfinite(max_logw):
        return np.zeros(n, dtype=float), np.nan

    # Scale invariance of self-normalized weights lets us shift by max log-weight.
    weights = np.exp(log_weights - max_logw)

    if n < 5:
        return weights, np.nan

    tail_size = int(np.floor(min(0.2 * n, 3.0 * np.sqrt(n))))
    tail_size = max(1, min(tail_size, n - 1))

    order = np.argsort(weights)
    weights_sorted = weights[order].copy()
    threshold_idx = n - tail_size - 1
    threshold = weights_sorted[threshold_idx]
    tail_raw = weights_sorted[n - tail_size :]
    exceedances = tail_raw - threshold

    if np.all(exceedances <= 0.0):
        smoothed = np.empty_like(weights_sorted)
        smoothed[:] = weights_sorted
        out = np.empty_like(smoothed)
        out[order] = smoothed
        return out, 0.0

    try:
        k_hat, _, sigma_hat = genpareto.fit(exceedances, floc=0.0)
        probs = (np.arange(1, tail_size + 1, dtype=float) - 0.5) / tail_size
        smoothed_tail = threshold + genpareto.ppf(probs, c=k_hat, loc=0.0, scale=sigma_hat)
        smoothed_tail = np.maximum(smoothed_tail, threshold)
        smoothed_tail = np.nan_to_num(smoothed_tail, nan=tail_raw, posinf=tail_raw[-1], neginf=threshold)
    except Exception:
        k_hat = np.nan
        smoothed_tail = tail_raw

    smoothed_sorted = weights_sorted
    smoothed_sorted[n - tail_size :] = smoothed_tail

    smoothed = np.empty_like(smoothed_sorted)
    smoothed[order] = smoothed_sorted
    return smoothed, float(k_hat) if np.isfinite(k_hat) else np.nan


def main():
    seed = args.seed
    np.random.seed(seed)
    rng = random.Random(seed)
    print(f"Seed fixed: {seed} (random, numpy)", flush=True)

    param_pairs = parse_param_pairs()
    print(f"Param pairs (alpha, beta): {param_pairs}", flush=True)

    df_benchmark = load_mtbench_single_df()
    df_pairwise = load_chatbot_arena_df(HF_TOKEN)
    df_benchmark, df_pairwise = prepare_benchmark_data(df_benchmark, df_pairwise)

    lam, out_adj = build_lam_and_out_adj(df_pairwise)
    print(f"Prepared graph with {len(MODELS)} players", flush=True)

    samples_by_pair = {}
    for param_idx, (alpha, beta) in enumerate(param_pairs):
        print(f"Sampling pair {param_idx + 1}/{len(param_pairs)}: alpha={alpha}, beta={beta}", flush=True)
        samples, df_rank = run_ranking(
            df_pairwise,
            n_samples=args.n_samples,
            burn_in=100000,
            thin=1000,
            seed=seed + param_idx,
            report_every=100,
            alpha=alpha,
            beta=beta,
        )
        samples_by_pair[(alpha, beta)] = samples
        print(df_rank, flush=True)

    rows = []
    labels = [pair_label(pair) for pair in param_pairs]
    ess_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    psis_ess_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    khat_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    for q_idx, q_pair in enumerate(param_pairs):
        q_alpha, q_beta = q_pair
        q_label = pair_label(q_pair)
        q_samples = samples_by_pair[q_pair]
        print(f"ESS proposal q={q_label} ({q_idx + 1}/{len(param_pairs)})", flush=True)

        log_q = [
            log_unnormalized_pmf(pi, lam, q_alpha, q_beta, out_adj)
            for pi in q_samples
        ]

        for p_idx, p_pair in enumerate(param_pairs):
            p_alpha, p_beta = p_pair
            p_label = pair_label(p_pair)

            if p_pair == q_pair:
                # Diagonal: q = p, trivial case ESS = N, PSIS-ESS = N, k_hat = 0
                n = len(q_samples)
                ess_matrix.loc[q_label, p_label] = n
                psis_ess_matrix.loc[q_label, p_label] = n
                khat_matrix.loc[q_label, p_label] = 0.0
                rows.append(
                    {
                        "q_alpha": q_alpha,
                        "q_beta": q_beta,
                        "p_alpha": p_alpha,
                        "p_beta": p_beta,
                        "q_label": q_label,
                        "p_label": p_label,
                        "ess": n,
                        "ess_fraction": 1.0,
                        "psis_ess": n,
                        "psis_ess_fraction": 1.0,
                        "k_hat": 0.0,
                        "n_samples": n,
                    }
                )
                print(f"  target p={p_label} (diagonal) ESS=PSIS-ESS=N={n}, k_hat=0", flush=True)
                continue

            # Only fill diagonal + upper triangle (p_idx >= q_idx); skip lower triangle (p_idx < q_idx)
            if p_idx < q_idx:
                continue

            print(f"  target p={p_label}", flush=True)
            log_weights = []
            for pi, lq in zip(q_samples, log_q):
                lp = log_unnormalized_pmf(pi, lam, p_alpha, p_beta, out_adj)
                log_weights.append(lp - lq)

            ess = log_ess_from_log_weights(log_weights)
            smoothed_weights, k_hat = psis_smooth_weights_from_log_weights(log_weights)
            psis_ess = ess_from_weights(smoothed_weights)
            rows.append(
                {
                    "q_alpha": q_alpha,
                    "q_beta": q_beta,
                    "p_alpha": p_alpha,
                    "p_beta": p_beta,
                    "q_label": q_label,
                    "p_label": p_label,
                    "ess": ess,
                    "ess_fraction": ess / len(q_samples) if q_samples else np.nan,
                    "psis_ess": psis_ess,
                    "psis_ess_fraction": psis_ess / len(q_samples) if q_samples else np.nan,
                    "k_hat": k_hat,
                    "n_samples": len(q_samples),
                }
            )
            ess_matrix.loc[q_label, p_label] = ess
            psis_ess_matrix.loc[q_label, p_label] = psis_ess
            khat_matrix.loc[q_label, p_label] = k_hat
            print(
                f"    ESS={ess:.6f}, PSIS-ESS={psis_ess:.6f}, k_hat={k_hat:.6f} out of N={len(q_samples)}",
                flush=True,
            )

    df_ess = pd.DataFrame(rows)
    df_ess.attrs["param_pairs"] = param_pairs
    df_ess.attrs["seed"] = seed
    df_ess.attrs["n_samples"] = args.n_samples

    save_dir = Path("save/ess")
    save_dir.mkdir(parents=True, exist_ok=True)
    long_path = save_dir / f"ess_long_n{args.n_samples}.pkl"
    matrix_path = save_dir / f"ess_matrix_n{args.n_samples}.pkl"
    psis_matrix_path = save_dir / f"psis_ess_matrix_n{args.n_samples}.pkl"
    khat_matrix_path = save_dir / f"khat_matrix_n{args.n_samples}.pkl"
    df_ess.to_pickle(long_path)
    ess_matrix.to_pickle(matrix_path)
    psis_ess_matrix.to_pickle(psis_matrix_path)
    khat_matrix.to_pickle(khat_matrix_path)

    print(f"Saved long ESS dataframe: {long_path}", flush=True)
    print(f"Saved ESS matrix: {matrix_path}", flush=True)
    print(f"Saved PSIS-ESS matrix: {psis_matrix_path}", flush=True)
    print(f"Saved k_hat matrix: {khat_matrix_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
