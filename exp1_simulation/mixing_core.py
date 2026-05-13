import argparse
import hashlib
import math
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    from exact import MAX_EXACT_N, exact_pairwise_matrix_subset_dp
    from graphs import (
        attach_omega_iid,
        build_graph_edges,
        graph_case_grid,
        parse_float_grid,
        parse_int_grid,
        parse_interval_grid,
        sample_greedy_init_perms,
        sample_lambda_iid,
        sample_random_init_perms,
        to_raw_params,
    )
    from metrics import build_doubling_checkpoints, empirical_pairwise_matrix, exact_discrepancy
    from sampler import draw_independent_final_permutations
else:
    from .exact import MAX_EXACT_N, exact_pairwise_matrix_subset_dp
    from .graphs import (
        attach_omega_iid,
        build_graph_edges,
        graph_case_grid,
        parse_float_grid,
        parse_int_grid,
        parse_interval_grid,
        sample_greedy_init_perms,
        sample_lambda_iid,
        sample_random_init_perms,
        to_raw_params,
    )
    from .metrics import build_doubling_checkpoints, empirical_pairwise_matrix, exact_discrepancy
    from .sampler import draw_independent_final_permutations


SAVE_ROOT = Path(__file__).resolve().parent / "save" / "mixing"
DEFAULT_SMALL_N_GRID = "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
DEFAULT_U_LAMBDA_GRID = "1,16,256"
DEFAULT_OMEGA_INTERVALS = "0,0.5;0.5,1.0;1.0,1.0"


def _save_pickle(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _seed_from(*parts: object) -> int:
    text = "|".join(str(x) for x in parts)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _make_chain_seeds(num_chains: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    return rng.randint(0, np.iinfo(np.int32).max, size=int(num_chains), dtype=np.int64)


def _make_shared_init_perms(
    *,
    n: int,
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    mode: str,
    num_chains: int,
    seed: int,
) -> np.ndarray:
    if str(mode) == "random":
        init_perm = sample_random_init_perms(int(n), 1, int(seed))[0]
    else:
        init_perm = sample_greedy_init_perms(
            int(n),
            np.asarray(lambdas, dtype=np.float64),
            edges_with_omega,
            1,
            int(seed),
        )[0]
    return np.repeat(np.asarray(init_perm, dtype=np.int64)[None, :], int(num_chains), axis=0)


def _make_init_evaluator(
    *,
    n: int,
    raw_lambda: np.ndarray,
    raw_edges: Sequence[Tuple[int, int, float]],
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    init_mode: str,
    num_chains: int,
    seed_base: int,
    lazy_prob: float,
    shared_init_across_chains: bool = False,
) -> Tuple[Callable[[int], np.ndarray], Dict[int, float]]:
    mode = str(init_mode).strip().lower()
    if mode not in {"random", "greedy"}:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    chain_seeds = _make_chain_seeds(int(num_chains), int(_seed_from(seed_base, mode, "chains")))
    init_seed = int(_seed_from(seed_base, mode, "init"))
    if bool(shared_init_across_chains):
        init_perms = _make_shared_init_perms(
            n=int(n),
            lambdas=np.asarray(lambdas, dtype=np.float64),
            edges_with_omega=edges_with_omega,
            mode=mode,
            num_chains=int(num_chains),
            seed=init_seed,
        )
    elif mode == "random":
        init_perms = sample_random_init_perms(
            int(n),
            int(num_chains),
            init_seed,
        )
    else:
        init_perms = sample_greedy_init_perms(
            int(n),
            np.asarray(lambdas, dtype=np.float64),
            edges_with_omega,
            int(num_chains),
            init_seed,
        )

    matrices: Dict[int, np.ndarray] = {}
    accept_rate: Dict[int, float] = {}

    def evaluate_at(t: int) -> np.ndarray:
        t = int(t)
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")
        if t in matrices:
            return matrices[t]

        final_perms, mean_acc = draw_independent_final_permutations(
            n=int(n),
            edges=list(raw_edges),
            lam=np.asarray(raw_lambda, dtype=np.float64).tolist(),
            steps=int(t),
            num_chains=int(num_chains),
            chain_seeds=chain_seeds,
            init_perms=init_perms,
            alpha=1.0,
            beta=1.0,
            lazy_prob=float(lazy_prob),
        )
        matrices[t] = empirical_pairwise_matrix(final_perms)
        accept_rate[t] = float(mean_acc)
        return matrices[t]

    return evaluate_at, accept_rate


def _memoize_metric(metric_fn: Callable[[int], float]) -> Callable[[int], float]:
    cache: Dict[int, float] = {}

    def wrapped(t: int) -> float:
        t = int(t)
        if t not in cache:
            cache[t] = float(metric_fn(t))
        return float(cache[t])

    return wrapped


def _binary_search_first_time(
    *,
    metric_fn: Callable[[int], float],
    checkpoints: Sequence[int],
    checkpoint_values: Sequence[float],
    epsilon: float,
    epsilon0: float,
) -> Dict[str, object]:
    low = float(epsilon) - float(epsilon0)
    high = float(epsilon) + float(epsilon0)
    ckpts = [int(t) for t in checkpoints]
    vals = [float(v) for v in checkpoint_values]

    first_idx: Optional[int] = None
    for idx, value in enumerate(vals):
        if float(value) <= low:
            first_idx = int(idx)
            break

    if first_idx is None:
        return {
            "first_below_band_t": None,
            "used_binary_search": False,
            "binary_search_steps": 0,
            "band_hits_in_binary_search": 0,
            "bracket_t": None,
        }

    coarse_t = int(ckpts[first_idx])
    if first_idx == 0:
        return {
            "first_below_band_t": coarse_t,
            "used_binary_search": False,
            "binary_search_steps": 0,
            "band_hits_in_binary_search": 0,
            "bracket_t": None,
        }

    candidate_lo = [i for i in range(first_idx) if float(vals[i]) > high]
    if not candidate_lo:
        return {
            "first_below_band_t": coarse_t,
            "used_binary_search": False,
            "binary_search_steps": 0,
            "band_hits_in_binary_search": 0,
            "bracket_t": None,
        }

    lo_idx = int(candidate_lo[-1])
    lo_t = int(ckpts[lo_idx])
    hi_t = int(ckpts[first_idx])
    bracket = (int(lo_t), int(hi_t))
    steps = 0
    band_hits = 0

    while hi_t - lo_t > 1:
        mid = int((lo_t + hi_t) // 2)
        mid_val = float(metric_fn(mid))
        steps += 1
        if mid_val <= low:
            hi_t = int(mid)
        elif mid_val > high:
            lo_t = int(mid)
        else:
            # Ambiguity zone: move upper bound left for a conservative earliest-time rule.
            hi_t = int(mid)
            band_hits += 1

    return {
        "first_below_band_t": int(hi_t),
        "used_binary_search": True,
        "binary_search_steps": int(steps),
        "band_hits_in_binary_search": int(band_hits),
        "bracket_t": bracket,
    }


def _persist_k_info(
    *,
    values: Sequence[float],
    checkpoints: Sequence[int],
    low_threshold: float,
    persist_k: int,
) -> Dict[str, object]:
    vals = [float(v) for v in values]
    ckpts = [int(t) for t in checkpoints]
    k = max(1, int(persist_k))

    for start in range(0, max(0, len(vals) - k + 1)):
        window_vals = vals[start : start + k]
        if all(v <= float(low_threshold) for v in window_vals):
            return {
                "required_consecutive": int(k),
                "first_t": int(ckpts[start]),
                "window_t": [int(t) for t in ckpts[start : start + k]],
                "window_values": [float(v) for v in window_vals],
            }

    return {
        "required_consecutive": int(k),
        "first_t": None,
        "window_t": None,
        "window_values": None,
    }


def _compute_curve_and_first_time(
    *,
    metric_fn: Callable[[int], float],
    checkpoints: Sequence[int],
    epsilon: float,
    epsilon0: float,
    persist_k: int,
) -> Dict[str, object]:
    ckpts = [int(t) for t in checkpoints]
    vals = [float(metric_fn(t)) for t in ckpts]
    low = float(epsilon) - float(epsilon0)
    out = {
        "curve": vals,
        "last_value": (None if not vals else float(vals[-1])),
        "coarse_first_below_low_t": next((int(t) for t, v in zip(ckpts, vals) if float(v) <= low), None),
    }
    out.update(
        _binary_search_first_time(
            metric_fn=metric_fn,
            checkpoints=ckpts,
            checkpoint_values=vals,
            epsilon=float(epsilon),
            epsilon0=float(epsilon0),
        )
    )
    out["persist"] = _persist_k_info(
        values=vals,
        checkpoints=ckpts,
        low_threshold=low,
        persist_k=int(persist_k),
    )
    out["band"] = {
        "low": low,
        "high": float(epsilon) + float(epsilon0),
    }
    return out


def _sorted_float_dict(d: Dict[int, float]) -> Dict[int, float]:
    return {int(k): float(d[k]) for k in sorted(d.keys())}


def _summary_stats(values: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "max": float(arr.max()),
    }


def _binary_entropy_base2(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    out = np.zeros_like(probs, dtype=np.float64)
    mask = (probs > 0.0) & (probs < 1.0)
    p = probs[mask]
    out[mask] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return out


def _categorical_entropy_base2(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    mask = probs > 0.0
    if not np.any(mask):
        return 0.0
    return float(-(probs[mask] * np.log2(probs[mask])).sum())


def _pack_perm_array(perms: np.ndarray) -> np.ndarray:
    perms = np.asarray(perms)
    max_value = int(perms.max()) if perms.size else 0
    if max_value <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.int64
    return perms.astype(dtype, copy=True)


def _compute_position_probs(samples: np.ndarray, n: int) -> np.ndarray:
    counts = np.zeros((int(n), int(n)), dtype=np.int64)
    positions = np.arange(int(n), dtype=np.int64)
    for sample in np.asarray(samples, dtype=np.int64):
        counts[sample, positions] += 1
    return counts.astype(np.float32) / float(samples.shape[0])


def _pair_order_signatures(samples: np.ndarray, n: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.int64)
    num_samples = int(samples.shape[0])
    n = int(n)
    pos_dtype = np.int16 if n <= np.iinfo(np.int16).max else np.int32
    positions = np.empty((num_samples, n), dtype=pos_dtype)
    rows = np.arange(num_samples, dtype=np.int64)[:, None]
    positions[rows, samples] = np.arange(n, dtype=np.int64)[None, :]
    i_idx, j_idx = np.triu_indices(n, k=1)
    return (positions[:, i_idx] < positions[:, j_idx]).astype(np.uint8)


def _kendall_analysis_from_samples(samples: np.ndarray, n: int, mode_permutation: np.ndarray) -> Dict[str, object]:
    samples = np.asarray(samples, dtype=np.int64)
    num_samples = int(samples.shape[0])
    n = int(n)
    max_pairs = int(n * (n - 1) // 2)

    if num_samples <= 1:
        return {
            "storage": "condensed_upper_tri",
            "num_samples": int(num_samples),
            "max_pairs": int(max_pairs),
            "distance_condensed": np.zeros(0, dtype=np.uint16),
            "distance_condensed_dtype": "uint16",
            "distance_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
            "distance_normalized_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
            "row_mean_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
            "row_mean_normalized_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
            "row_std_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
            "to_mode_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
            "to_mode_normalized_stats": _summary_stats(np.zeros(0, dtype=np.float64)),
        }

    signatures = _pair_order_signatures(samples, int(n)).astype(np.int32, copy=False)
    row_sums = signatures.sum(axis=1, dtype=np.int32)
    common = signatures @ signatures.T
    distances = row_sums[:, None] + row_sums[None, :] - (2 * common)
    distances = np.asarray(distances, dtype=np.int32)

    tri_idx = np.triu_indices(num_samples, k=1)
    condensed = distances[tri_idx].astype(np.uint16)
    condensed_float = condensed.astype(np.float64)
    normalized = condensed_float / float(max_pairs) if max_pairs > 0 else condensed_float

    row_mean = distances.sum(axis=1, dtype=np.float64) / float(num_samples - 1)
    row_sq_mean = (distances.astype(np.float64) ** 2).sum(axis=1) / float(num_samples - 1)
    row_var = np.maximum(0.0, row_sq_mean - row_mean ** 2)
    row_std = np.sqrt(row_var)

    mode_signature = _pair_order_signatures(np.asarray(mode_permutation, dtype=np.int64)[None, :], int(n)).reshape(-1)
    mode_signature = mode_signature.astype(np.int32, copy=False)
    to_mode = row_sums + int(mode_signature.sum(dtype=np.int32)) - (2 * (signatures @ mode_signature))
    to_mode = np.asarray(to_mode, dtype=np.float64)

    return {
        "storage": "condensed_upper_tri",
        "num_samples": int(num_samples),
        "max_pairs": int(max_pairs),
        "distance_condensed": condensed,
        "distance_condensed_dtype": str(condensed.dtype),
        "distance_stats": _summary_stats(condensed_float),
        "distance_normalized_stats": _summary_stats(normalized),
        "row_mean_stats": _summary_stats(row_mean),
        "row_mean_normalized_stats": _summary_stats(
            row_mean / float(max_pairs) if max_pairs > 0 else row_mean
        ),
        "row_std_stats": _summary_stats(row_std),
        "to_mode_stats": _summary_stats(to_mode),
        "to_mode_normalized_stats": _summary_stats(
            to_mode / float(max_pairs) if max_pairs > 0 else to_mode
        ),
    }


def _compute_greedy_init_analysis(
    *,
    n: int,
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    num_samples: int,
    seed: int,
) -> Dict[str, object]:
    num_samples = int(num_samples)
    if num_samples <= 0:
        return {"num_samples": 0}

    greedy_samples = sample_greedy_init_perms(
        int(n),
        np.asarray(lambdas, dtype=np.float64),
        edges_with_omega,
        int(num_samples),
        int(seed),
    )
    packed_samples = _pack_perm_array(greedy_samples)
    unique_perms, counts = np.unique(packed_samples, axis=0, return_counts=True)
    mode_idx = int(np.argmax(counts))

    pairwise_probs = empirical_pairwise_matrix(greedy_samples).astype(np.float32)
    tri_mask = np.triu(np.ones((int(n), int(n)), dtype=bool), k=1)
    tri_probs = pairwise_probs[tri_mask].astype(np.float64)
    pairwise_margin = np.abs(tri_probs - 0.5)
    pairwise_variance = tri_probs * (1.0 - tri_probs)
    pairwise_entropy = _binary_entropy_base2(tri_probs)

    position_probs = _compute_position_probs(greedy_samples, int(n))
    pos_idx = np.arange(int(n), dtype=np.float64)
    item_mean_pos = position_probs.astype(np.float64) @ pos_idx
    item_second_moment = position_probs.astype(np.float64) @ (pos_idx ** 2)
    item_var_pos = np.maximum(0.0, item_second_moment - item_mean_pos ** 2)
    item_std_pos = np.sqrt(item_var_pos)
    item_position_entropy = np.array(
        [_categorical_entropy_base2(position_probs[i, :]) for i in range(int(n))],
        dtype=np.float64,
    )

    return {
        "num_samples": int(num_samples),
        "samples": packed_samples,
        "samples_dtype": str(packed_samples.dtype),
        "unique_count": int(unique_perms.shape[0]),
        "mode_count": int(counts[mode_idx]),
        "mode_fraction": float(counts[mode_idx] / float(num_samples)),
        "mode_permutation": unique_perms[mode_idx].copy(),
        "pairwise_preference_probs": pairwise_probs,
        "pairwise_margin_to_half_stats": _summary_stats(pairwise_margin),
        "pairwise_variance_stats": _summary_stats(pairwise_variance),
        "pairwise_entropy_stats": _summary_stats(pairwise_entropy),
        "position_probs": position_probs,
        "item_mean_position": item_mean_pos.astype(np.float32),
        "item_position_std_stats": _summary_stats(item_std_pos),
        "item_position_entropy_stats": _summary_stats(item_position_entropy),
        "top1_entropy": float(_categorical_entropy_base2(position_probs[:, 0])),
        "top1_mode_mass": float(np.max(position_probs[:, 0])),
        "last1_entropy": float(_categorical_entropy_base2(position_probs[:, -1])),
        "last1_mode_mass": float(np.max(position_probs[:, -1])),
        "kendall": _kendall_analysis_from_samples(
            packed_samples,
            int(n),
            unique_perms[mode_idx].copy(),
        ),
    }


def _exact_horizon(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return max(1, int(math.ceil((float(n) ** 3) * math.log(float(n)))))


def _build_checkpoints_with_exact_horizon(horizon: int) -> List[int]:
    horizon = int(horizon)
    checkpoints = build_doubling_checkpoints(max_t=int(horizon), t0=1)
    if int(horizon) not in checkpoints:
        checkpoints.append(int(horizon))
    return sorted({int(t) for t in checkpoints})


def _build_small_tasks(
    *,
    n_grid: Sequence[int],
    u_lambda_grid: Sequence[float],
    omega_intervals: Sequence[Tuple[float, float]],
    num_chains: int,
    graph_reps: int,
    target_graph_reps: int,
    epsilon: float,
    epsilon0: float,
    lazy_prob: float,
    seed_base: int,
    persist_k: int,
    greedy_init_samples: int,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    cases = graph_case_grid()
    case_index = 0

    for n in n_grid:
        horizon = _exact_horizon(int(n))
        checkpoints = _build_checkpoints_with_exact_horizon(int(horizon))
        for case in cases:
            family = str(case["family"])
            p = case["p"]
            reps_run = 1 if family == "total_order" else int(graph_reps)
            reps_target = 1 if family == "total_order" else int(target_graph_reps)
            if reps_run > reps_target:
                raise ValueError(
                    f"graph_reps={reps_run} cannot exceed target_graph_reps={reps_target} "
                    f"for family={family}"
                )

            for rep in range(reps_run):
                graph_seed = _seed_from(seed_base, "small", n, family, p, rep, "graph")
                for u_lambda in u_lambda_grid:
                    lambda_seed = _seed_from(seed_base, "small", n, family, p, rep, u_lambda, "lambda")
                    for omega_interval in omega_intervals:
                        omega_seed = _seed_from(
                            seed_base,
                            "small",
                            n,
                            family,
                            p,
                            rep,
                            u_lambda,
                            omega_interval[0],
                            omega_interval[1],
                            "omega",
                        )
                        case_index += 1
                        tasks.append(
                            {
                                "case_index": int(case_index),
                                "mode": "small",
                                "n": int(n),
                                "graph_family": family,
                                "graph_p": (None if p is None else float(p)),
                                "graph_rep": int(rep),
                                "graph_reps_run_for_family": int(reps_run),
                                "target_graph_reps_for_family": int(reps_target),
                                "u_lambda": float(u_lambda),
                                "omega_interval": (float(omega_interval[0]), float(omega_interval[1])),
                                "graph_seed": int(graph_seed),
                                "lambda_seed": int(lambda_seed),
                                "omega_seed": int(omega_seed),
                                "horizon": int(horizon),
                                "checkpoints": [int(t) for t in checkpoints],
                                "num_chains": int(num_chains),
                                "epsilon": float(epsilon),
                                "epsilon0": float(epsilon0),
                                "lazy_prob": float(lazy_prob),
                                "persist_k": int(persist_k),
                                "greedy_init_samples": int(greedy_init_samples),
                            }
                        )
    return tasks


def _format_task_brief(task: Dict[str, object], total_cases: int) -> str:
    return (
        f"[small] case {int(task['case_index'])}/{int(total_cases)} "
        f"n={int(task['n'])} graph={task['graph_family']} p={task['graph_p']} "
        f"rep={int(task['graph_rep']) + 1}/{int(task['target_graph_reps_for_family'])} "
        f"U_lambda={float(task['u_lambda'])} omega={task['omega_interval']}"
    )


def _format_metric_log(name: str, metric_payload: Dict[str, object]) -> str:
    last_value = metric_payload.get("last_value")
    first_t = metric_payload.get("first_below_band_t")
    persist_t = metric_payload.get("persist", {}).get("first_t")
    used_binary_search = bool(metric_payload.get("used_binary_search", False))
    first_t_text = "None" if first_t is None else str(int(first_t))
    persist_text = "None" if persist_t is None else str(int(persist_t))
    last_text = "None" if last_value is None else f"{float(last_value):.4f}"
    bs_text = "Y" if used_binary_search else "N"
    return f"{name}(last={last_text},t*={first_t_text},p3={persist_text},bs={bs_text})"


def _format_greedy_init_log(greedy_analysis: Dict[str, object]) -> str:
    num_samples = int(greedy_analysis.get("num_samples", 0))
    if num_samples <= 0:
        return "GI(ns=0)"

    unique_count = int(greedy_analysis.get("unique_count", 0))
    mode_fraction = greedy_analysis.get("mode_fraction")
    top1_entropy = greedy_analysis.get("top1_entropy")
    top1_mode_mass = greedy_analysis.get("top1_mode_mass")

    kendall = greedy_analysis.get("kendall", {})
    k_stats = kendall.get("distance_normalized_stats", {})
    k_mean = k_stats.get("mean")
    k_std = k_stats.get("std")

    mode_text = "None" if mode_fraction is None else f"{float(mode_fraction):.3f}"
    k_text = "None" if k_mean is None else f"{float(k_mean):.3f}"
    ks_text = "None" if k_std is None else f"{float(k_std):.3f}"
    top1h_text = "None" if top1_entropy is None else f"{float(top1_entropy):.2f}"
    top1m_text = "None" if top1_mode_mass is None else f"{float(top1_mode_mass):.3f}"

    return (
        f"GI(uniq={unique_count},mode={mode_text},"
        f"kd={k_text}±{ks_text},top1H={top1h_text},top1M={top1m_text})"
    )


def _summarize_result_for_log(result: Dict[str, object]) -> str:
    return " ".join(
        [
            _format_metric_log("Dr", result["random"]["D"]),
            _format_metric_log("Dg", result["greedy"]["D"]),
            _format_greedy_init_log(result["greedy_init_analysis"]),
        ]
    )


def _format_case_done(
    task: Dict[str, object],
    total_cases: int,
    done: int,
    result: Dict[str, object],
) -> str:
    return (
        f"{_format_task_brief(task, total_cases)} done "
        f"({done}/{total_cases} completed, "
        f"elapsed={float(result['elapsed_seconds']):.2f}s; "
        f"{_summarize_result_for_log(result)})"
    )


def _run_condition(
    *,
    n: int,
    graph_family: str,
    graph_p: Optional[float],
    graph_rep: int,
    u_lambda: float,
    omega_interval: Tuple[float, float],
    graph_seed: int,
    lambda_seed: int,
    omega_seed: int,
    graph_reps_run_for_family: int,
    target_graph_reps_for_family: int,
    horizon: int,
    checkpoints: Sequence[int],
    num_chains: int,
    epsilon: float,
    epsilon0: float,
    lazy_prob: float,
    persist_k: int,
    greedy_init_samples: int,
    init_rep: int = 0,
    shared_init_across_chains: bool = False,
) -> Dict[str, object]:
    graph_edges = build_graph_edges(
        n=int(n),
        family=str(graph_family),
        p=(None if graph_p is None else float(graph_p)),
        seed=int(graph_seed),
    )
    lambdas = sample_lambda_iid(
        int(n),
        float(u_lambda),
        np.random.RandomState(int(lambda_seed)),
    )
    edges_with_omega = attach_omega_iid(
        graph_edges,
        omega_interval,
        np.random.RandomState(int(omega_seed)),
    )
    raw_lambda, raw_edges = to_raw_params(lambdas, edges_with_omega)
    checkpoints = [int(t) for t in checkpoints]

    eval_seed = _seed_from(
        "eval",
        "small",
        n,
        graph_family,
        graph_p,
        graph_rep,
        init_rep,
        u_lambda,
        omega_interval[0],
        omega_interval[1],
    )
    rand_eval, rand_acc = _make_init_evaluator(
        n=int(n),
        raw_lambda=np.asarray(raw_lambda, dtype=np.float64),
        raw_edges=list(raw_edges),
        lambdas=lambdas,
        edges_with_omega=edges_with_omega,
        init_mode="random",
        num_chains=int(num_chains),
        seed_base=int(_seed_from(eval_seed, "random")),
        lazy_prob=float(lazy_prob),
        shared_init_across_chains=bool(shared_init_across_chains),
    )
    greedy_eval, greedy_acc = _make_init_evaluator(
        n=int(n),
        raw_lambda=np.asarray(raw_lambda, dtype=np.float64),
        raw_edges=list(raw_edges),
        lambdas=lambdas,
        edges_with_omega=edges_with_omega,
        init_mode="greedy",
        num_chains=int(num_chains),
        seed_base=int(_seed_from(eval_seed, "greedy")),
        lazy_prob=float(lazy_prob),
        shared_init_across_chains=bool(shared_init_across_chains),
    )

    m_exact, z = exact_pairwise_matrix_subset_dp(
        np.asarray(lambdas, dtype=np.float64),
        edges_with_omega,
        check_symmetry=True,
    )
    rand_d_metric = _memoize_metric(lambda t: exact_discrepancy(rand_eval(int(t)), m_exact))
    greedy_d_metric = _memoize_metric(lambda t: exact_discrepancy(greedy_eval(int(t)), m_exact))

    payload: Dict[str, object] = {
        "mode": "small",
        "n": int(n),
        "graph_family": str(graph_family),
        "graph_p": (None if graph_p is None else float(graph_p)),
        "graph_rep": int(graph_rep),
        "init_rep": int(init_rep),
        "graph_reps_run_for_family": int(graph_reps_run_for_family),
        "target_graph_reps_for_family": int(target_graph_reps_for_family),
        "u_lambda": float(u_lambda),
        "omega_interval": (float(omega_interval[0]), float(omega_interval[1])),
        "graph_seed": int(graph_seed),
        "lambda_seed": int(lambda_seed),
        "omega_seed": int(omega_seed),
        "num_edges": int(len(graph_edges)),
        "horizon": int(horizon),
        "checkpoints": [int(t) for t in checkpoints],
        "shared_init_across_chains": bool(shared_init_across_chains),
        "random": {
            "D": _compute_curve_and_first_time(
                metric_fn=rand_d_metric,
                checkpoints=checkpoints,
                epsilon=float(epsilon),
                epsilon0=float(epsilon0),
                persist_k=int(persist_k),
            ),
        },
        "greedy": {
            "D": _compute_curve_and_first_time(
                metric_fn=greedy_d_metric,
                checkpoints=checkpoints,
                epsilon=float(epsilon),
                epsilon0=float(epsilon0),
                persist_k=int(persist_k),
            ),
        },
        "exact": {
            "partition_z": float(z),
        },
        "greedy_init_analysis": _compute_greedy_init_analysis(
            n=int(n),
            lambdas=lambdas,
            edges_with_omega=edges_with_omega,
            num_samples=int(greedy_init_samples),
            seed=int(_seed_from(eval_seed, "greedy_init_analysis")),
        ),
    }
    payload["random"]["accept_rate_at_t"] = _sorted_float_dict(rand_acc)
    payload["greedy"]["accept_rate_at_t"] = _sorted_float_dict(greedy_acc)
    return payload


def _run_condition_from_task(task: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    result = _run_condition(
        n=int(task["n"]),
        graph_family=str(task["graph_family"]),
        graph_p=task["graph_p"],
        graph_rep=int(task["graph_rep"]),
        u_lambda=float(task["u_lambda"]),
        omega_interval=task["omega_interval"],
        graph_seed=int(task["graph_seed"]),
        lambda_seed=int(task["lambda_seed"]),
        omega_seed=int(task["omega_seed"]),
        graph_reps_run_for_family=int(task["graph_reps_run_for_family"]),
        target_graph_reps_for_family=int(task["target_graph_reps_for_family"]),
        horizon=int(task["horizon"]),
        checkpoints=task["checkpoints"],
        num_chains=int(task["num_chains"]),
        epsilon=float(task["epsilon"]),
        epsilon0=float(task["epsilon0"]),
        lazy_prob=float(task["lazy_prob"]),
        persist_k=int(task["persist_k"]),
        greedy_init_samples=int(task["greedy_init_samples"]),
        init_rep=int(task.get("init_rep", 0)),
        shared_init_across_chains=bool(task.get("shared_init_across_chains", False)),
    )
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return {"case_index": int(task["case_index"]), "result": result}


def _ordered_results(results_by_index: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    return [results_by_index[idx] for idx in sorted(results_by_index.keys())]


def _run_small(
    *,
    n_grid: Sequence[int],
    u_lambda_grid: Sequence[float],
    omega_intervals: Sequence[Tuple[float, float]],
    num_chains: int,
    graph_reps: int,
    target_graph_reps: int,
    epsilon: float,
    epsilon0: float,
    lazy_prob: float,
    save_dir: Path,
    seed_base: int,
    num_workers: int,
    persist_k: int,
    greedy_init_samples: int,
) -> List[Dict[str, object]]:
    tasks = _build_small_tasks(
        n_grid=n_grid,
        u_lambda_grid=u_lambda_grid,
        omega_intervals=omega_intervals,
        num_chains=int(num_chains),
        graph_reps=int(graph_reps),
        target_graph_reps=int(target_graph_reps),
        epsilon=float(epsilon),
        epsilon0=float(epsilon0),
        lazy_prob=float(lazy_prob),
        seed_base=int(seed_base),
        persist_k=int(persist_k),
        greedy_init_samples=int(greedy_init_samples),
    )
    total_cases = len(tasks)
    results_by_index: Dict[int, Dict[str, object]] = {}
    done = 0
    t_start = time.perf_counter()
    partial_path = save_dir / "mixing_partial.pkl"

    if int(num_workers) <= 1:
        for task in tasks:
            print(_format_task_brief(task, total_cases), flush=True)
            payload = _run_condition_from_task(task)
            done += 1
            results_by_index[int(payload["case_index"])] = payload["result"]
            print(_format_case_done(task, total_cases, done, payload["result"]), flush=True)
            if done % 5 == 0:
                results = _ordered_results(results_by_index)
                _save_pickle(
                    partial_path,
                    {"mode": "small", "num_results": len(results), "results": results},
                )
                print(f"  saved partial: {partial_path}", flush=True)
    else:
        max_workers = min(int(num_workers), max(1, total_cases))
        print(
            f"[small] running case-level multiprocessing with num_workers={max_workers}",
            flush=True,
        )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            future_to_task = {executor.submit(_run_condition_from_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                payload = future.result()
                done += 1
                results_by_index[int(payload["case_index"])] = payload["result"]
                print(_format_case_done(task, total_cases, done, payload["result"]), flush=True)
                if done % 5 == 0:
                    results = _ordered_results(results_by_index)
                    _save_pickle(
                        partial_path,
                        {"mode": "small", "num_results": len(results), "results": results},
                    )
                    print(f"  saved partial: {partial_path}", flush=True)

    results = _ordered_results(results_by_index)
    print(
        f"[small] done in {time.perf_counter() - t_start:.2f}s with {len(results)} results",
        flush=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1 small-mode exact-D mixing experiment runner")
    parser.add_argument("--save_dir", type=str, default=str(SAVE_ROOT))
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--small_n_grid", type=str, default=DEFAULT_SMALL_N_GRID)
    parser.add_argument("--u_lambda_grid", type=str, default=DEFAULT_U_LAMBDA_GRID)
    parser.add_argument("--omega_intervals", type=str, default=DEFAULT_OMEGA_INTERVALS)
    parser.add_argument("--num_chains", type=int, default=10000)
    parser.add_argument("--graph_reps", type=int, default=1)
    parser.add_argument("--target_graph_reps", type=int, default=1)
    parser.add_argument("--greedy_init_samples", type=int, default=1000)
    parser.add_argument("--persist_k", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--epsilon0", type=float, default=0.02)
    parser.add_argument("--lazy_prob", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    if int(args.graph_reps) <= 0:
        raise ValueError(f"graph_reps must be positive, got {args.graph_reps}")
    if int(args.target_graph_reps) <= 0:
        raise ValueError(f"target_graph_reps must be positive, got {args.target_graph_reps}")
    if int(args.graph_reps) > int(args.target_graph_reps):
        raise ValueError(
            f"graph_reps ({args.graph_reps}) cannot exceed target_graph_reps ({args.target_graph_reps})"
        )
    if int(args.greedy_init_samples) < 0:
        raise ValueError(f"greedy_init_samples must be non-negative, got {args.greedy_init_samples}")
    if int(args.persist_k) <= 0:
        raise ValueError(f"persist_k must be positive, got {args.persist_k}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    small_n_grid = parse_int_grid(args.small_n_grid)
    if any(int(n) > int(MAX_EXACT_N) for n in small_n_grid):
        raise ValueError(
            f"mixing_core.py only supports n <= {MAX_EXACT_N} because exact subset-DP "
            f"is configured for that range, got n_grid={small_n_grid}"
        )

    u_lambda_grid = parse_float_grid(args.u_lambda_grid)
    omega_intervals = parse_interval_grid(args.omega_intervals)

    config = {
        "mode": "small",
        "seed_base": int(args.seed_base),
        "small_n_grid": [int(n) for n in small_n_grid],
        "u_lambda_grid": [float(x) for x in u_lambda_grid],
        "omega_intervals": [(float(lo), float(hi)) for lo, hi in omega_intervals],
        "num_chains": int(args.num_chains),
        "graph_reps": int(args.graph_reps),
        "target_graph_reps": int(args.target_graph_reps),
        "greedy_init_samples": int(args.greedy_init_samples),
        "persist_k": int(args.persist_k),
        "horizon_rule": "T(n) = ceil(n^3 log n)",
        "checkpoint_rule": "t_k = 2^k up to the largest power of 2 <= T(n), plus T(n) itself if needed",
        "persist_rule": "persist_k consecutive checkpoints with D_t <= epsilon - epsilon0",
        "epsilon": float(args.epsilon),
        "epsilon0": float(args.epsilon0),
        "lazy_prob": float(args.lazy_prob),
        "num_workers": int(args.num_workers),
        "max_exact_n": int(MAX_EXACT_N),
        "graph_cases": graph_case_grid(),
        "timestamp_unix": float(time.time()),
    }

    print("Mixing-small experiment configuration")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    results = _run_small(
        n_grid=small_n_grid,
        u_lambda_grid=u_lambda_grid,
        omega_intervals=omega_intervals,
        num_chains=int(args.num_chains),
        graph_reps=int(args.graph_reps),
        target_graph_reps=int(args.target_graph_reps),
        epsilon=float(args.epsilon),
        epsilon0=float(args.epsilon0),
        lazy_prob=float(args.lazy_prob),
        save_dir=save_dir,
        seed_base=int(_seed_from(args.seed_base, "small")),
        num_workers=int(args.num_workers),
        persist_k=int(args.persist_k),
        greedy_init_samples=int(args.greedy_init_samples),
    )

    out_path = save_dir / "mixing.pkl"
    _save_pickle(out_path, {"config": config, "small": results})
    print(f"Saved mixing-small results to {out_path}")


if __name__ == "__main__":
    main()
