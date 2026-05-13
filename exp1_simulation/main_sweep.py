import argparse
import math
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    from sampler import mh_sampler
    from sampler import mh_sampler_pasv, topological_order
    from main_accuracy import (
        _normalize_seed,
        _save_pickle,
        get_cpu_model_name,
        get_total_memory_gb,
        print_system_info,
        sample_greedy_order,
    )
else:
    from .sampler import mh_sampler
    from .sampler import mh_sampler_pasv, topological_order
    from .main_accuracy import (
        _normalize_seed,
        _save_pickle,
        get_cpu_model_name,
        get_total_memory_gb,
        print_system_info,
        sample_greedy_order,
    )


METHOD_PASV = "pasv"
METHOD_GPASV = "gpasv"
EPS_LAMBDA = 1e-12


def parse_float_list(arg: str) -> List[float]:
    return [float(x.strip()) for x in str(arg).split(",") if x.strip()]


def parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in str(arg).split(",") if x.strip()]


# -----------------------------
# Setup builders
# -----------------------------
def build_random_dag(n: int, edge_prob: float, seed: int) -> List[Tuple[int, int]]:
    """Bernoulli(p) edges i -> j for every i < j on vertices 0..n-1."""
    rng = np.random.RandomState(_normalize_seed(int(seed)))
    edges: List[Tuple[int, int]] = []
    n = int(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < float(edge_prob):
                edges.append((int(i), int(j)))
    return edges


def select_sweep_group(n: int, group_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(_normalize_seed(int(seed)))
    return np.sort(rng.choice(int(n), size=int(group_size), replace=False)).astype(np.int64)


def build_carriers(
    n: int,
    num_carriers: int,
    carrier_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(_normalize_seed(int(carrier_seed)))
    n = int(n)
    d = int(num_carriers)
    T_mask = np.empty((d, n), dtype=bool)
    remaining = np.arange(d, dtype=np.int64)
    while remaining.size > 0:
        candidates = rng.rand(remaining.size, n) < 0.5
        valid = candidates.any(axis=1)
        T_mask[remaining[valid]] = candidates[valid]
        remaining = remaining[~valid]
    coeffs = rng.uniform(0.5, 1.5, size=d).astype(np.float64)
    return T_mask, coeffs


def build_lambda_paper(n: int, sweep_group: np.ndarray, lam_bar: float) -> np.ndarray:
    n = int(n)
    lam_paper = np.ones(n, dtype=np.float64)
    lam_paper[np.asarray(sweep_group, dtype=np.int64)] = float(lam_bar)
    return lam_paper


def paper_lambda_to_code_lam(lam_paper: np.ndarray) -> np.ndarray:
    lam_paper = np.asarray(lam_paper, dtype=np.float64)
    return (-np.log(np.maximum(lam_paper, EPS_LAMBDA))).astype(np.float64)


def edges_to_unit_weight_tuples(edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    return [(int(u), int(v), 1.0) for (u, v) in edges]


# -----------------------------
# Rank summaries
# -----------------------------
def rank_summary_from_samples(
    samples: np.ndarray,
    sweep_group: np.ndarray,
    sweep_group_complement: np.ndarray,
) -> Dict[str, object]:
    samples = np.asarray(samples, dtype=np.int64)
    N, n = samples.shape
    inv_perm = np.empty_like(samples)
    pos_range = np.arange(n, dtype=np.int64)[None, :]
    np.put_along_axis(
        inv_perm, samples, np.broadcast_to(pos_range, samples.shape), axis=1,
    )
    positions = inv_perm.astype(np.float64) + 1.0  # 1-indexed forward rank

    G = np.asarray(sweep_group, dtype=np.int64)
    Gc = np.asarray(sweep_group_complement, dtype=np.int64)
    rbar_G = positions[:, G].mean(axis=1)
    rbar_Gc = positions[:, Gc].mean(axis=1)
    within_sd_G = positions[:, G].std(axis=1, ddof=0) if G.size > 1 else np.zeros(N)
    within_sd_Gc = positions[:, Gc].std(axis=1, ddof=0) if Gc.size > 1 else np.zeros(N)

    def _stats(x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return float("nan"), float("nan")
        if x.size == 1:
            return float(x[0]), 0.0
        return float(x.mean()), float(x.std(ddof=1))

    # Across-sample mean/SD of the per-sample scalar.
    mean_rbar_G, across_sd_rbar_G = _stats(rbar_G)
    mean_rbar_Gc, across_sd_rbar_Gc = _stats(rbar_Gc)
    mean_within_sd_G, across_sd_within_sd_G = _stats(within_sd_G)
    mean_within_sd_Gc, across_sd_within_sd_Gc = _stats(within_sd_Gc)

    per_player_mean = positions.mean(axis=0).astype(np.float64)
    if positions.shape[0] <= 1:
        per_player_std = np.zeros(n, dtype=np.float64)
    else:
        per_player_std = positions.std(axis=0, ddof=1).astype(np.float64)

    return {
        # Per-sample scalar r̄_· ; its across-sample mean and SD.
        "rbar_G_mean": float(mean_rbar_G),
        "rbar_G_std": float(across_sd_rbar_G),
        "rbar_Gc_mean": float(mean_rbar_Gc),
        "rbar_Gc_std": float(across_sd_rbar_Gc),
        # Within-permutation rank SD for G / G^c ; its across-sample mean and SD.
        "within_sd_G_mean": float(mean_within_sd_G),
        "within_sd_G_std": float(across_sd_within_sd_G),
        "within_sd_Gc_mean": float(mean_within_sd_Gc),
        "within_sd_Gc_std": float(across_sd_within_sd_Gc),
        "per_player_rank_mean": per_player_mean,
        "per_player_rank_std": per_player_std,
    }


# -----------------------------
# Value-sum summaries (sum-of-unanimity / sum-of-race games)
# -----------------------------
def value_sum_from_samples(
    samples: np.ndarray,
    sweep_group: np.ndarray,
    T_mask: np.ndarray,
    coeffs: np.ndarray,
    chunk_size: int = 500,
) -> Dict[str, float]:
    samples = np.asarray(samples, dtype=np.int64)
    N_MC, n = samples.shape
    inv_perm = np.empty_like(samples)
    pos_range = np.arange(n, dtype=np.int64)[None, :]
    np.put_along_axis(
        inv_perm, samples, np.broadcast_to(pos_range, samples.shape), axis=1,
    )

    T_mask = np.asarray(T_mask, dtype=bool)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    G_mask = np.zeros(int(n), dtype=bool)
    G_mask[np.asarray(sweep_group, dtype=np.int64)] = True

    sizes = T_mask.sum(axis=1).astype(np.int64)
    size_groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for s_val in np.unique(sizes):
        s_int = int(s_val)
        if s_int == 0:
            continue
        sel = np.where(sizes == s_int)[0]
        mem_s = np.stack(
            [np.where(T_mask[k])[0] for k in sel], axis=0,
        ).astype(np.int64)
        coef_s = coeffs[sel]
        size_groups.append((s_int, mem_s, coef_s))

    per_sample_unanimity = np.zeros(N_MC, dtype=np.float64)
    per_sample_race = np.zeros(N_MC, dtype=np.float64)

    chunk = int(max(1, chunk_size))
    for start in range(0, N_MC, chunk):
        stop = min(start + chunk, N_MC)
        inv_sub = inv_perm[start:stop]  # (N_chunk, n)
        for s_int, mem_s, coef_s in size_groups:
            d_s = mem_s.shape[0]
            pos_T = inv_sub[:, mem_s]  # (N_chunk, d_s, s_int)
            last_idx = pos_T.argmax(axis=2)
            first_idx = pos_T.argmin(axis=2)
            rng_d = np.arange(d_s, dtype=np.int64)[None, :]
            last_p = mem_s[rng_d, last_idx]
            first_p = mem_s[rng_d, first_idx]
            per_sample_unanimity[start:stop] += (
                G_mask[last_p] * coef_s[None, :]
            ).sum(axis=1)
            per_sample_race[start:stop] += (
                G_mask[first_p] * coef_s[None, :]
            ).sum(axis=1)

    def _stats(x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return float("nan"), float("nan")
        if x.size == 1:
            return float(x[0]), 0.0
        return float(x.mean()), float(x.std(ddof=1))

    u_mean, u_std = _stats(per_sample_unanimity)
    r_mean, r_std = _stats(per_sample_race)
    return {
        "value_sum_unanimity_G_mean": float(u_mean),
        "value_sum_unanimity_G_std": float(u_std),
        "value_sum_race_G_mean": float(r_mean),
        "value_sum_race_G_std": float(r_std),
    }


# -----------------------------
# Per-chain runner (single MH chain)
# -----------------------------
def _build_init_pi(
    *,
    method: str,
    n: int,
    edges: Sequence[Tuple[int, int]],
    lam_paper: np.ndarray,
    raw_edges_unit: Sequence[Tuple[int, int, float]],
    init_seed: int,
) -> np.ndarray:
    if method == METHOD_PASV:
        return topological_order(int(n), edges, seed=int(init_seed))
    if method == METHOD_GPASV:
        return sample_greedy_order(
            int(n),
            np.asarray(lam_paper, dtype=np.float64),
            raw_edges_unit,
            int(init_seed),
        )
    raise ValueError(f"Unknown method: {method}")


def run_one_chain(
    *,
    method: str,
    n: int,
    edges: Sequence[Tuple[int, int]],
    lam_paper: np.ndarray,
    sweep_group: np.ndarray,
    sweep_group_complement: np.ndarray,
    T_mask: np.ndarray,
    coeffs: np.ndarray,
    value_sum_chunk_size: int,
    beta: float,
    num_samples: int,
    burn_in: int,
    thin: int,
    lazy_prob: float,
    init_pi: np.ndarray,
    chain_seed: int,
    report_every: int,
) -> Dict[str, object]:
    n = int(n)
    lam_code = paper_lambda_to_code_lam(lam_paper)
    raw_edges_unit = edges_to_unit_weight_tuples(edges)

    t0 = time.perf_counter()
    if method == METHOD_PASV:
        samples, acceptance_rate = mh_sampler_pasv(
            n=n,
            edges=edges,
            lam=lam_code.tolist(),
            alpha=1.0,
            seed=int(chain_seed),
            n_samples=int(num_samples),
            burn_in=int(burn_in),
            thin=int(thin),
            init_pi=init_pi,
            report_every=int(report_every),
            lazy_prob=float(lazy_prob),
        )
    elif method == METHOD_GPASV:
        samples, acceptance_rate = mh_sampler(
            n=n,
            edges=raw_edges_unit,
            lam=lam_code.tolist(),
            alpha=1.0,
            beta=float(beta),
            seed=int(chain_seed),
            n_samples=int(num_samples),
            burn_in=int(burn_in),
            thin=int(thin),
            init_pi=init_pi,
            report_every=int(report_every),
            lazy_prob=float(lazy_prob),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = float(time.perf_counter() - t0)

    samples_arr = np.asarray(samples, dtype=np.int64)
    summary = rank_summary_from_samples(samples_arr, sweep_group, sweep_group_complement)
    value_summary = value_sum_from_samples(
        samples_arr, sweep_group, T_mask, coeffs,
        chunk_size=int(value_sum_chunk_size),
    )
    summary.update(value_summary)
    summary.update({
        "acceptance_rate": float(acceptance_rate),
        "elapsed_seconds": float(elapsed),
        "num_samples": int(samples_arr.shape[0]),
        "chain_seed": int(chain_seed),
    })
    return summary


# -----------------------------
# Per-configuration runner (nreps replications; DAG/G/init all fixed)
# -----------------------------
def run_config(
    *,
    method: str,
    n: int,
    edges: Sequence[Tuple[int, int]],
    lam_paper: np.ndarray,
    sweep_group: np.ndarray,
    sweep_group_complement: np.ndarray,
    T_mask: np.ndarray,
    coeffs: np.ndarray,
    value_sum_chunk_size: int,
    beta: float,
    nreps: int,
    num_samples: int,
    burn_in: int,
    thin: int,
    lazy_prob: float,
    init_seed: int,
    chain_seed_base_for_config: int,
    report_every: int,
) -> Dict[str, object]:
    n = int(n)
    nreps = int(nreps)
    raw_edges_unit = edges_to_unit_weight_tuples(edges)
    init_pi = _build_init_pi(
        method=method, n=n, edges=edges, lam_paper=lam_paper,
        raw_edges_unit=raw_edges_unit, init_seed=int(init_seed),
    )

    per_rep: List[Dict[str, object]] = []
    for rep_idx in range(nreps):
        rep_chain_seed = _normalize_seed(int(chain_seed_base_for_config) + int(rep_idx))
        summary = run_one_chain(
            method=method, n=n, edges=edges, lam_paper=lam_paper,
            sweep_group=sweep_group, sweep_group_complement=sweep_group_complement,
            T_mask=T_mask, coeffs=coeffs,
            value_sum_chunk_size=int(value_sum_chunk_size),
            beta=beta, num_samples=num_samples, burn_in=burn_in, thin=thin,
            lazy_prob=lazy_prob, init_pi=init_pi, chain_seed=rep_chain_seed,
            report_every=report_every,
        )
        per_rep.append(summary)

    def _agg(keys):
        arr = np.asarray([float(r[keys]) for r in per_rep], dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")
        if arr.size == 1:
            return float(arr[0]), 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    # Scalars (mean and SD across reps).
    agg_scalars = {}
    for scalar_key in (
        "rbar_G_mean", "rbar_Gc_mean",
        "within_sd_G_mean", "within_sd_Gc_mean",
        "value_sum_unanimity_G_mean", "value_sum_race_G_mean",
        "acceptance_rate",
    ):
        m, s = _agg(scalar_key)
        # naming: `<key>_rep_mean` = average across reps of the per-rep scalar.
        #         `<key>_rep_sd`   = SD across reps of the per-rep scalar.
        agg_scalars[f"{scalar_key}_rep_mean"] = m
        agg_scalars[f"{scalar_key}_rep_sd"] = s

    # Per-player rank: mean across reps of per-rep mean vector; SD across reps.
    per_player_stack = np.vstack([
        np.asarray(r["per_player_rank_mean"], dtype=np.float64) for r in per_rep
    ])
    per_player_rep_mean = per_player_stack.mean(axis=0).astype(np.float64)
    if per_player_stack.shape[0] <= 1:
        per_player_rep_sd = np.zeros(n, dtype=np.float64)
    else:
        per_player_rep_sd = per_player_stack.std(axis=0, ddof=1).astype(np.float64)

    total_elapsed = float(sum(float(r["elapsed_seconds"]) for r in per_rep))
    rep_chain_seeds = [int(r["chain_seed"]) for r in per_rep]

    return {
        "nreps": int(nreps),
        "init_seed": int(init_seed),
        "chain_seed_base_for_config": int(chain_seed_base_for_config),
        "rep_chain_seeds": rep_chain_seeds,
        "rep_acceptance_rates": [float(r["acceptance_rate"]) for r in per_rep],
        "rep_elapsed_seconds": [float(r["elapsed_seconds"]) for r in per_rep],
        "total_elapsed_seconds": total_elapsed,
        "num_samples_per_rep": int(per_rep[0]["num_samples"]) if per_rep else 0,
        "per_player_rank_rep_mean": per_player_rep_mean,
        "per_player_rank_rep_sd": per_player_rep_sd,
        **agg_scalars,
    }


# -----------------------------
# Checkpoint bookkeeping
# -----------------------------
def config_key(method: str, p: float, beta: float, lam_bar: float) -> str:
    return f"method={method}|p={float(p):.4f}|beta={float(beta):.4f}|lam_bar={float(lam_bar):.4f}"


def build_config_list(
    *,
    p_grid: Sequence[float],
    lam_bar_grid: Sequence[float],
    beta_grid: Sequence[float],
) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for p in p_grid:
        for lam_bar in lam_bar_grid:
            configs.append({
                "method": METHOD_PASV,
                "p": float(p),
                "beta": float("nan"),
                "lam_bar": float(lam_bar),
            })
            for beta in beta_grid:
                configs.append({
                    "method": METHOD_GPASV,
                    "p": float(p),
                    "beta": float(beta),
                    "lam_bar": float(lam_bar),
                })
    return configs


def load_existing_results(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"Warning: could not load existing checkpoint {path}: {exc}")
        return {}
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    if isinstance(results, list):
        results = {
            config_key(r["method"], r["p"], r["beta"], r["lam_bar"]): r
            for r in results
            if "method" in r and "p" in r and "beta" in r and "lam_bar" in r
        }
    required_keys = (
        "value_sum_unanimity_G_mean_rep_mean",
        "value_sum_race_G_mean_rep_mean",
    )
    filtered: Dict[str, Dict[str, object]] = {}
    stale = 0
    for k, v in results.items():
        entry = dict(v)
        if all(key in entry for key in required_keys):
            filtered[str(k)] = entry
        else:
            stale += 1
    if stale > 0:
        print(
            f"Note: {stale} of {len(results)} cached configs in {path} predate "
            f"the value-sum schema and will be re-run.",
            flush=True,
        )
    return filtered


def save_results(
    path: Path,
    config_payload: Dict[str, object],
    setup_payload: Dict[str, object],
    results: Dict[str, Dict[str, object]],
) -> None:
    _save_pickle(
        path,
        {
            "config": dict(config_payload),
            "setup": dict(setup_payload),
            "results": dict(results),
        },
    )


# -----------------------------
# Main
# -----------------------------
def run_experiment(
    *,
    n: int,
    group_size: int,
    p_grid: Sequence[float],
    lam_bar_grid: Sequence[float],
    beta_grid: Sequence[float],
    nreps: int,
    dag_seed: int,
    group_seed: int,
    init_seed_base: int,
    chain_seed_base: int,
    num_samples: int,
    burn_in: int,
    thin: int,
    lazy_prob: float,
    report_every: int,
    num_carriers: int,
    carrier_seed: int,
    value_sum_chunk_size: int,
    save_path: Path,
    config_payload: Dict[str, object],
) -> Dict[str, object]:
    n = int(n)

    # DAGs (one per p; p-dependent seed)
    dags: Dict[float, List[Tuple[int, int]]] = {}
    for p in p_grid:
        p_seed = _normalize_seed(int(dag_seed) + int(round(float(p) * 10_000)))
        dags[float(p)] = build_random_dag(n, float(p), p_seed)

    # Sweep group (single seed, shared across all p)
    sweep_group = select_sweep_group(n, int(group_size), int(group_seed))
    sweep_group_complement = np.setdiff1d(
        np.arange(n, dtype=np.int64), sweep_group, assume_unique=True
    ).astype(np.int64)

    # Shared carrier sets T_1,...,T_d and coefficients c_k for both utilities.
    T_mask, carrier_coeffs = build_carriers(n, int(num_carriers), int(carrier_seed))
    carrier_sizes = T_mask.sum(axis=1).astype(np.int64)

    setup_payload: Dict[str, object] = {
        "n": int(n),
        "group_size": int(group_size),
        "p_grid": [float(x) for x in p_grid],
        "lam_bar_grid": [float(x) for x in lam_bar_grid],
        "beta_grid": [float(x) for x in beta_grid],
        "nreps": int(nreps),
        "dag_seed": int(dag_seed),
        "group_seed": int(group_seed),
        "init_seed_base": int(init_seed_base),
        "chain_seed_base": int(chain_seed_base),
        "sweep_group": sweep_group.tolist(),
        "sweep_group_complement": sweep_group_complement.tolist(),
        "dags": {
            float(p): [(int(u), int(v)) for (u, v) in edges] for p, edges in dags.items()
        },
        "num_carriers": int(num_carriers),
        "carrier_seed": int(carrier_seed),
        "carrier_T_mask": T_mask.copy(),
        "carrier_coeffs": carrier_coeffs.copy(),
        "carrier_sizes": carrier_sizes.copy(),
        "value_sum_chunk_size": int(value_sum_chunk_size),
    }

    configs = build_config_list(
        p_grid=p_grid, lam_bar_grid=lam_bar_grid, beta_grid=beta_grid,
    )
    total_configs = len(configs)

    results = load_existing_results(save_path)
    completed = 0
    for c in configs:
        if config_key(c["method"], c["p"], c["beta"], c["lam_bar"]) in results:
            completed += 1

    print(f"Total configurations: {total_configs}")
    print(f"Already completed (from checkpoint): {completed}")
    print(f"Remaining: {total_configs - completed}")
    print()

    cumulative_elapsed = 0.0
    runs_this_session = 0
    t_wall_start = time.perf_counter()

    for idx, c in enumerate(configs, start=1):
        method = str(c["method"])
        p = float(c["p"])
        beta = float(c["beta"])
        lam_bar = float(c["lam_bar"])
        key = config_key(method, p, beta, lam_bar)

        if key in results:
            print(
                f"[{idx}/{total_configs}] SKIP (done)  method={method}  p={p}  "
                f"beta={beta}  lam_bar={lam_bar}",
                flush=True,
            )
            continue

        beta_str = f"{beta:.4g}" if math.isfinite(beta) else "n/a"
        print(
            f"[{idx}/{total_configs}] START  method={method}  p={p}  "
            f"beta={beta_str}  lam_bar={lam_bar}",
            flush=True,
        )

        # Seeds: deterministic from (method, p, beta, lam_bar).
        # init_seed is shared across reps within a config, so init_pi is fixed.
        # chain_seed_base_for_config + rep_idx gives each rep its own MH seed.
        seed_tag = abs(hash((method, round(p, 6), round(beta, 6), round(lam_bar, 6)))) % (2**31)
        init_seed = _normalize_seed(int(init_seed_base) + int(seed_tag))
        chain_seed_base_for_config = _normalize_seed(int(chain_seed_base) + int(seed_tag))

        lam_paper = build_lambda_paper(n, sweep_group, lam_bar)

        summary = run_config(
            method=method,
            n=n,
            edges=dags[p],
            lam_paper=lam_paper,
            sweep_group=sweep_group,
            sweep_group_complement=sweep_group_complement,
            T_mask=T_mask,
            coeffs=carrier_coeffs,
            value_sum_chunk_size=int(value_sum_chunk_size),
            beta=beta,
            nreps=int(nreps),
            num_samples=int(num_samples),
            burn_in=int(burn_in),
            thin=int(thin),
            lazy_prob=float(lazy_prob),
            init_seed=int(init_seed),
            chain_seed_base_for_config=int(chain_seed_base_for_config),
            report_every=int(report_every),
        )

        result_entry: Dict[str, object] = {
            "method": method,
            "p": p,
            "beta": beta,
            "lam_bar": lam_bar,
            **summary,
        }
        results[key] = result_entry

        cumulative_elapsed += float(summary["total_elapsed_seconds"])
        runs_this_session += 1

        # Cumulative ETA: based on *this session's* average per-config time.
        avg_time = cumulative_elapsed / max(1, runs_this_session)
        remaining = total_configs - idx
        eta_sec = float(avg_time * remaining)

        acc_mean = float(summary["acceptance_rate_rep_mean"])
        rbar_G_m = float(summary["rbar_G_mean_rep_mean"])
        rbar_G_s = float(summary["rbar_G_mean_rep_sd"])
        wsd_G_m = float(summary["within_sd_G_mean_rep_mean"])
        wsd_G_s = float(summary["within_sd_G_mean_rep_sd"])
        vsu_m = float(summary["value_sum_unanimity_G_mean_rep_mean"])
        vsu_s = float(summary["value_sum_unanimity_G_mean_rep_sd"])
        vsr_m = float(summary["value_sum_race_G_mean_rep_mean"])
        vsr_s = float(summary["value_sum_race_G_mean_rep_sd"])
        print(
            f"[{idx}/{total_configs}] DONE   method={method}  p={p}  "
            f"beta={beta_str}  lam_bar={lam_bar}  nreps={nreps}  "
            f"acc={acc_mean:.4f}  "
            f"time={summary['total_elapsed_seconds']:.2f}s  "
            f"rbar_G={rbar_G_m:.3f}±{rbar_G_s:.3f}  "
            f"within_sd_G={wsd_G_m:.3f}±{wsd_G_s:.3f}  "
            f"V^U={vsu_m:.3f}±{vsu_s:.3f}  "
            f"V^R={vsr_m:.3f}±{vsr_s:.3f}  "
            f"ETA={eta_sec:.0f}s",
            flush=True,
        )

        save_results(save_path, config_payload, setup_payload, results)

    wall_elapsed = float(time.perf_counter() - t_wall_start)
    print()
    print(f"Session wall time: {wall_elapsed:.2f}s over {runs_this_session} new configs.")
    print(f"Saved results to {save_path}")

    return {
        "setup": setup_payload,
        "results": results,
        "total_configs": int(total_configs),
        "runs_this_session": int(runs_this_session),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--p_grid", type=str, default="0.2,0.5,0.8")
    parser.add_argument(
        "--lam_bar_grid", type=str, default="1,2,4,8,16,32,64,128",
    )
    parser.add_argument("--beta_grid", type=str, default="1,10,100")
    parser.add_argument("--nreps", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--burn_in", type=int, default=50_000)
    parser.add_argument("--thin", type=int, default=200)
    parser.add_argument("--lazy_prob", type=float, default=0.5)
    parser.add_argument("--report_every", type=int, default=0)
    parser.add_argument("--dag_seed", type=int, default=42)
    parser.add_argument("--group_seed", type=int, default=43)
    parser.add_argument("--init_seed_base", type=int, default=2024)
    parser.add_argument("--chain_seed_base", type=int, default=91_827_364)
    parser.add_argument(
        "--num_carriers", type=int, default=None,
        help="Number of shared carrier sets T_k (defaults to n**2).",
    )
    parser.add_argument("--carrier_seed", type=int, default=44)
    parser.add_argument("--value_sum_chunk_size", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_tag", type=str, default="default")
    args = parser.parse_args()

    p_grid = parse_float_list(args.p_grid)
    lam_bar_grid = parse_float_list(args.lam_bar_grid)
    beta_grid = parse_float_list(args.beta_grid)
    num_carriers = (
        int(args.num_carriers) if args.num_carriers is not None else int(args.n) ** 2
    )

    if args.save_dir is None:
        save_dir = Path(__file__).resolve().parent / "save" / "sweep"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sweep_{str(args.save_tag)}.pkl"

    print_system_info()

    config_payload: Dict[str, object] = {
        "n": int(args.n),
        "group_size": int(args.group_size),
        "p_grid": [float(x) for x in p_grid],
        "lam_bar_grid": [float(x) for x in lam_bar_grid],
        "beta_grid": [float(x) for x in beta_grid],
        "nreps": int(args.nreps),
        "num_samples": int(args.num_samples),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "lazy_prob": float(args.lazy_prob),
        "report_every": int(args.report_every),
        "dag_seed": int(args.dag_seed),
        "group_seed": int(args.group_seed),
        "init_seed_base": int(args.init_seed_base),
        "chain_seed_base": int(args.chain_seed_base),
        "num_carriers": int(num_carriers),
        "carrier_seed": int(args.carrier_seed),
        "value_sum_chunk_size": int(args.value_sum_chunk_size),
        "save_tag": str(args.save_tag),
    }

    print("Lambda-sweep comparison configuration")
    for key, value in config_payload.items():
        print(f"  {key}: {value}")
    print(f"  save_path: {save_path}")
    print()

    run_experiment(
        n=int(args.n),
        group_size=int(args.group_size),
        p_grid=p_grid,
        lam_bar_grid=lam_bar_grid,
        beta_grid=beta_grid,
        nreps=int(args.nreps),
        dag_seed=int(args.dag_seed),
        group_seed=int(args.group_seed),
        init_seed_base=int(args.init_seed_base),
        chain_seed_base=int(args.chain_seed_base),
        num_samples=int(args.num_samples),
        burn_in=int(args.burn_in),
        thin=int(args.thin),
        lazy_prob=float(args.lazy_prob),
        report_every=int(args.report_every),
        num_carriers=int(num_carriers),
        carrier_seed=int(args.carrier_seed),
        value_sum_chunk_size=int(args.value_sum_chunk_size),
        save_path=save_path,
        config_payload=config_payload,
    )


if __name__ == "__main__":
    main()
