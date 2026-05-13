import argparse
import math
import os
import pickle
import resource
import tracemalloc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

if __package__ in (None, ""):
    from sampler import mh_sampler
else:
    from .sampler import mh_sampler

try:
    from numba import njit, prange
except Exception:

    def njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def deco(func):
            return func

        return deco

    prange = range


# ---------------------------------------------------------------------------
# Import everything we can reuse verbatim from surrogate.py
# ---------------------------------------------------------------------------
from surrogate import (
    # constants
    EPS_OMEGA,
    EPS_LAMBDA,
    PROXY_LSMR_COL_THRESHOLD,
    PROXY_LSMR_ATOL,
    PROXY_LSMR_BTOL,
    PROXY_LSMR_MAXITER,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_NUM_PAIR_SAMPLES,
    # helpers
    parse_n_grid,
    parse_list_arg,
    parse_float_list_arg,
    summarize_scalar,
    summarize_curve,
    get_cpu_model_name,
    get_total_memory_gb,
    warmup_numba,
    print_system_info,
    resolve_burnin_steps,
    normalize_init_mode,
    parse_init_modes,
    sample_greedy_order,
    base_params_to_raw,
    # memory helpers
    _memory_status_mb,
    _finite_mb_delta,
    _mb_to_gb,
    # numba kernels
    _bit_index_uint64,
    _build_dense_design_numba_uint64,
    _build_dense_design_numba_chunks,
    _count_sparse_design_nnz_and_active,
    _fill_sparse_design_csr,
    _build_sparse_design_csr,
    _build_pair_lookup,
    _pack_subset_masks_to_words,
    _proxy_values_words_numba,
    _estimate_pairwise_probs_for_pairs_numba,
    _proxy_value_mask_numba,
    _residual_loop_numba,
    # proxy helpers
    estimate_pairwise_probs_for_pairs,
    fit_quadratic_proxy,
    proxy_value_from_pair_probs,
    proxy_value_mask,
    build_proxy_incremental_updates,
    # data generation
    segment_value,
    make_blocks,
    make_random_block_dag,
    sample_uniform_intervals,
    build_interval_weight_matrix,
    build_segment_weight_table,
    exact_target_block_cycle,
    exact_target_total_order,
    make_lambda_block_cycle,
    make_lambda_total_order,
    omega_case_bounds,
    build_block_cycle_graph,
    build_block_cycle_graph_from_template,
    build_shared_scenario2_graph_templates,
    build_total_order_graph,
    # oracles & stats
    UtilityStats,
    CaseData,
    TotalOrderOracle,
    BlockCycleOracle,
    get_cached_utility,
    # sampling
    draw_permutations,
    resolve_num_pair_samples,
    resolve_pair_permutation_count,
    make_random_pair_list,
    # rho support
    build_rho_support,
    sample_q_indices,
    gamma_scale,
    # pickling
    _save_pickle,
)


# ---------------------------------------------------------------------------
# Checkpoint grid
# ---------------------------------------------------------------------------

def make_checkpoint_grid(num_samples: int, checkpoint_every: int) -> List[int]:
    return [ckpt for ckpt in range(int(checkpoint_every), int(num_samples) + 1, int(checkpoint_every))]


def make_unique_checkpoint_grid(unique_budget: int, num_checkpoints: int) -> List[int]:
    if int(unique_budget) <= 0 or int(num_checkpoints) <= 0:
        return []
    raw = np.linspace(
        float(unique_budget) / float(num_checkpoints),
        float(unique_budget),
        int(num_checkpoints),
        dtype=np.float64,
    )
    checkpoints = sorted({int(math.ceil(float(x))) for x in raw})
    return [min(max(int(x), 1), int(unique_budget)) for x in checkpoints]


def relative_are(phi_hat: np.ndarray, exact_target: np.ndarray) -> float:
    exact_norm = max(float(np.linalg.norm(exact_target)), EPS_LAMBDA)
    return float(np.linalg.norm(phi_hat - exact_target) / exact_norm)


# ---------------------------------------------------------------------------
# Method: permutation  (unchanged from surrogate.py)
# ---------------------------------------------------------------------------

def run_permutation_method(
    case: CaseData,
    *,
    samples: np.ndarray,
    sampling_seconds: float,
    checkpoints: Sequence[int],
) -> Dict[str, object]:
    t0 = time.perf_counter()
    n = int(case.n)
    phi_sum = np.zeros(n, dtype=np.float64)
    cache: Dict[int, float] = {0: 0.0}
    stats = UtilityStats()
    phi_hats: List[np.ndarray] = []
    time_curve: List[float] = []
    utility_time_curve: List[float] = []
    total_eval_curve: List[int] = []
    unique_eval_curve: List[int] = []
    checkpoint_set = set(int(x) for x in checkpoints)

    for sample_idx, perm in enumerate(samples, start=1):
        for player, _mask, current_u, prev_u in case.oracle.iter_prefix_values(perm, cache, stats):
            phi_sum[int(player)] += float(current_u - prev_u)
        if sample_idx in checkpoint_set:
            phi_hats.append(phi_sum / float(sample_idx))
            elapsed = float(sampling_seconds + (time.perf_counter() - t0))
            time_curve.append(elapsed)
            utility_time_curve.append(float(stats.utility_seconds))
            total_eval_curve.append(int(stats.total_evals))
            unique_eval_curve.append(int(stats.unique_evals))

    out = summarize_trial_curve(
        exact_target=case.exact_target,
        checkpoints=checkpoints,
        phi_hats=phi_hats,
        time_curve=time_curve,
        utility_time_curve=utility_time_curve,
        total_eval_curve=total_eval_curve,
        unique_eval_curve=unique_eval_curve,
    )
    out.update(
        {
            "method": "permutation",
            "num_samples": int(samples.shape[0]),
            "sampling_seconds": float(sampling_seconds),
            "num_cached_subsets": int(len(cache)),
        }
    )
    return out


def replay_permutation_unique_checkpoints(
    case: CaseData,
    *,
    samples: np.ndarray,
    unique_checkpoints: Sequence[int],
) -> Dict[str, object]:
    """Replay already sampled permutations and record ARE at unique-call checkpoints."""
    checkpoints = [int(x) for x in sorted({int(c) for c in unique_checkpoints if int(c) > 0})]
    if not checkpoints:
        return {
            "unique_checkpoint_target_curve": [],
            "unique_checkpoint_curve": [],
            "unique_checkpoint_actual_curve": [],
            "are_by_unique_curve": [],
            "checkpoint_kind_curve": [],
            "unique_checkpoint_sample_curve": [],
        }

    n = int(case.n)
    phi_sum = np.zeros(n, dtype=np.float64)
    cache: Dict[int, float] = {0: 0.0}
    stats = UtilityStats()
    target_pos = 0
    target_curve: List[int] = []
    actual_unique_curve: List[int] = []
    are_curve: List[float] = []
    sample_curve: List[int] = []

    for sample_idx, perm in enumerate(samples, start=1):
        for player, _mask, current_u, prev_u in case.oracle.iter_prefix_values(perm, cache, stats):
            phi_sum[int(player)] += float(current_u - prev_u)

        while target_pos < len(checkpoints) and int(stats.unique_evals) >= int(checkpoints[target_pos]):
            phi_hat = phi_sum / float(sample_idx)
            target_curve.append(int(checkpoints[target_pos]))
            actual_unique_curve.append(int(stats.unique_evals))
            are_curve.append(relative_are(phi_hat, case.exact_target))
            sample_curve.append(int(sample_idx))
            target_pos += 1

        if target_pos >= len(checkpoints):
            break

    return {
        "unique_checkpoint_target_curve": target_curve,
        "unique_checkpoint_curve": target_curve,
        "unique_checkpoint_actual_curve": actual_unique_curve,
        "are_by_unique_curve": are_curve,
        "checkpoint_kind_curve": ["common"] * len(are_curve),
        "unique_checkpoint_sample_curve": sample_curve,
    }


# ---------------------------------------------------------------------------
# summarize_trial_curve  (unchanged from surrogate.py)
# ---------------------------------------------------------------------------

def summarize_trial_curve(
    *,
    exact_target: np.ndarray,
    checkpoints: Sequence[int],
    phi_hats: Sequence[np.ndarray],
    time_curve: Sequence[float],
    utility_time_curve: Sequence[float],
    total_eval_curve: Sequence[int],
    unique_eval_curve: Sequence[int],
) -> Dict[str, object]:
    exact_norm = max(float(np.linalg.norm(exact_target)), EPS_LAMBDA)
    are_curve = [float(np.linalg.norm(phi - exact_target) / exact_norm) for phi in phi_hats]
    return {
        "checkpoints": [int(x) for x in checkpoints],
        "utility_evals_per_player_curve": [float(x) / float(len(exact_target)) for x in total_eval_curve],
        "are_curve": are_curve,
        "time_curve": [float(x) for x in time_curve],
        "utility_time_curve": [float(x) for x in utility_time_curve],
        "non_utility_time_curve": [float(t - u) for t, u in zip(time_curve, utility_time_curve)],
        "total_utility_eval_curve": [int(x) for x in total_eval_curve],
        "unique_utility_eval_curve": [int(x) for x in unique_eval_curve],
        "reused_utility_eval_curve": [int(t - u) for t, u in zip(total_eval_curve, unique_eval_curve)],
        "final_are": float(are_curve[-1]) if are_curve else float("nan"),
        "elapsed_seconds": float(time_curve[-1]) if time_curve else float("nan"),
        "utility_eval_seconds": float(utility_time_curve[-1]) if utility_time_curve else float("nan"),
        "non_utility_seconds": float(time_curve[-1] - utility_time_curve[-1]) if time_curve else float("nan"),
        "total_utility_evals": int(total_eval_curve[-1]) if total_eval_curve else 0,
        "unique_utility_evals": int(unique_eval_curve[-1]) if unique_eval_curve else 0,
        "reused_utility_evals": int(total_eval_curve[-1] - unique_eval_curve[-1]) if total_eval_curve else 0,
    }


def collect_q_training_data_until_unique_budget(
    case: CaseData,
    *,
    rho_support: Dict[str, object],
    rng: np.random.RandomState,
    train_unique_budget: int,
    chunk_size: int,
) -> Tuple[Dict[int, float], Dict[int, float], UtilityStats, Dict[int, float], int, bool]:
    """Collect q-training raw samples until the requested unique budget is reached."""
    n = int(case.n)
    masks: List[int] = rho_support["masks"]
    probs: np.ndarray = rho_support["probs"]
    abs_counts: Dict[int, int] = rho_support["abs_counts"]
    square_counts: Dict[int, int] = rho_support["square_counts"]

    cache: Dict[int, float] = {0: 0.0}
    stats = UtilityStats()
    value_by_subset: Dict[int, float] = {}
    multiplicity: Dict[int, float] = {}
    total_samples = 0
    max_new_support = sum(1 for mask in masks if int(mask) not in cache)
    target_unique = min(int(train_unique_budget), int(max_new_support))

    while int(stats.unique_evals) < int(target_unique):
        remaining = int(target_unique) - int(stats.unique_evals)
        draw_count = min(int(chunk_size), max(1, remaining) * 4)
        idx_chunk = rng.choice(int(probs.shape[0]), size=int(draw_count), replace=True, p=probs)
        for idx_raw in idx_chunk:
            idx = int(idx_raw)
            mask = int(masks[idx])
            value = get_cached_utility(case.oracle, cache, mask, stats)
            value_by_subset.setdefault(mask, float(value))
            multiplicity[mask] = float(multiplicity.get(mask, 0.0)) + 1.0
            total_samples += 1
            if int(stats.unique_evals) >= int(target_unique):
                break

    weight_by_subset: Dict[int, float] = {}
    rho_perms = max(int(rho_support["rho_permutations"]), 1)
    for mask, count in multiplicity.items():
        rho_weight = float(square_counts[int(mask)]) * float(2 * n) / (
            float(rho_perms) * max(float(abs_counts[int(mask)]), 1.0)
        )
        weight_by_subset[int(mask)] = max(float(count) * max(rho_weight, 1e-12), 1e-12)

    exhausted = int(target_unique) < int(train_unique_budget)
    return value_by_subset, weight_by_subset, stats, cache, int(total_samples), bool(exhausted)


# ---------------------------------------------------------------------------
# Method: surrogate_subset with unique-budget alignment
# ---------------------------------------------------------------------------

def run_surrogate_subset_budget_aligned(
    case: CaseData,
    *,
    rho_support: Dict[str, object],
    pair_samples: np.ndarray,
    pair_sampling_seconds: float,
    pair_list: Sequence[Tuple[int, int]],
    unique_budget: int,
    seed: int,
    chunk_size: int,
    interaction_percent: float,
    method_name: str,
    fit_label: str,
    unique_checkpoints: Sequence[int],
    train_frac: float,
    train_cap: int,
) -> Dict[str, object]:
    """Run surrogate_subset with unique-budget train split and raw q adjustment."""
    t0 = time.perf_counter()
    n = int(case.n)
    rng = np.random.RandomState(int(seed))

    if int(unique_budget) > 1:
        requested_train = int(math.floor(float(train_frac) * float(int(unique_budget))))
        requested_train = min(int(train_cap), max(1, requested_train))
        train_unique_budget = min(int(requested_train), int(unique_budget) - 1)
    else:
        train_unique_budget = max(0, int(unique_budget))

    # --- Train phase: raw q samples until the train unique budget is reached ---
    value_by_subset, weight_by_subset, stats, cache, train_total_samples, train_exhausted = (
        collect_q_training_data_until_unique_budget(
            case,
            rho_support=rho_support,
            rng=rng,
            train_unique_budget=int(train_unique_budget),
            chunk_size=int(chunk_size),
        )
    )
    B_train = int(stats.unique_evals)
    if train_exhausted:
        print(
            f"      NOTE: rho support exhausted during train "
            f"(unique={B_train}, train_budget={train_unique_budget}).",
            flush=True,
        )

    # --- Fit proxy ---
    prob_start = time.perf_counter()
    pair_probs = estimate_pairwise_probs_for_pairs(pair_samples, pair_list, n)
    prob_seconds = float(time.perf_counter() - prob_start)

    fit_start = time.perf_counter()
    a_hat, b_hat, fit_diag = fit_quadratic_proxy(
        value_by_subset,
        weight_by_subset,
        n=n,
        empty_value=float(case.oracle.empty_value),
        pair_list=pair_list,
        print_label=fit_label,
        return_diagnostics=True,
    )
    fit_seconds = float(time.perf_counter() - fit_start)
    proxy_phi = proxy_value_from_pair_probs(a_hat, b_hat, pair_list, pair_probs)

    # --- Prepare residual (bias-adjustment) phase ---
    masks: List[int] = rho_support["masks"]
    probs: np.ndarray = rho_support["probs"]
    rho_abs_counts: np.ndarray = rho_support["rho_abs_counts"]
    rho_indptr: np.ndarray = rho_support["rho_indptr"]
    rho_players: np.ndarray = rho_support["rho_players"]
    rho_signed_counts: np.ndarray = rho_support["rho_signed_counts"]
    num_support = int(len(masks))
    max_new_support = sum(1 for mask in masks if int(mask) != 0)

    empty_value = float(case.oracle.empty_value)
    support_utility = np.full(num_support, np.nan, dtype=np.float64)
    support_proxy = np.full(num_support, np.nan, dtype=np.float64)
    pair_i = np.asarray([int(i) for i, _ in pair_list], dtype=np.int64)
    pair_j = np.asarray([int(j) for _, j in pair_list], dtype=np.int64)
    pair_coefs = np.asarray([float(b_hat[int(i), int(j)]) for i, j in pair_list], dtype=np.float64)
    proxy_eval_seconds = 0.0

    residual_sum = np.zeros(n, dtype=np.float64)
    adjustment_total_samples = 0
    checkpoint_targets: List[int] = []
    checkpoint_actual: List[int] = []
    checkpoint_are: List[float] = []
    checkpoint_kind: List[str] = []
    checkpoint_adjust_samples: List[int] = []

    if B_train > 0:
        checkpoint_targets.append(int(B_train))
        checkpoint_actual.append(int(B_train))
        checkpoint_are.append(relative_are(proxy_phi, case.exact_target))
        checkpoint_kind.append("fit")
        checkpoint_adjust_samples.append(0)

    common_checkpoints = [int(c) for c in sorted({int(c) for c in unique_checkpoints if int(c) > int(B_train)})]
    checkpoint_pos = 0
    budget_reached = int(stats.unique_evals) >= int(unique_budget)

    while (
        int(stats.unique_evals) < int(unique_budget)
        and int(stats.unique_evals) < int(max_new_support)
        and not budget_reached
    ):
        idx_chunk = rng.choice(
            int(probs.shape[0]),
            size=min(int(chunk_size), max(1, int(unique_budget) - int(stats.unique_evals)) * 4),
            replace=True,
            p=probs,
        )

        accepted_indices: List[int] = []
        record_marks: List[Tuple[int, int, int]] = []
        for idx_raw in idx_chunk:
            idx = int(idx_raw)
            mask = int(masks[idx])
            stats.total_evals += 1
            value = cache.get(mask)
            if value is None:
                t_util = time.perf_counter()
                value = float(case.oracle.utility_from_mask(mask))
                stats.utility_seconds += float(time.perf_counter() - t_util)
                stats.unique_evals += 1
                cache[mask] = float(value)
            support_utility[idx] = float(value)
            accepted_indices.append(idx)

            while (
                checkpoint_pos < len(common_checkpoints)
                and int(stats.unique_evals) >= int(common_checkpoints[checkpoint_pos])
            ):
                record_marks.append(
                    (
                        int(common_checkpoints[checkpoint_pos]),
                        int(len(accepted_indices)),
                        int(stats.unique_evals),
                    )
                )
                checkpoint_pos += 1

            if int(stats.unique_evals) >= int(unique_budget):
                budget_reached = True
                break

        if not accepted_indices:
            continue

        proxy_missing_indices: List[int] = []
        proxy_missing_masks: List[int] = []
        seen_missing = set()
        for idx in accepted_indices:
            idx_int = int(idx)
            if idx_int not in seen_missing and np.isnan(support_proxy[idx_int]):
                seen_missing.add(idx_int)
                proxy_missing_indices.append(idx_int)
                proxy_missing_masks.append(int(masks[idx_int]))
        if proxy_missing_indices:
            t_proxy = time.perf_counter()
            proxy_words = _pack_subset_masks_to_words(proxy_missing_masks, int(n))
            proxy_values = _proxy_values_words_numba(
                proxy_words,
                np.float64(empty_value),
                a_hat,
                pair_i,
                pair_j,
                pair_coefs,
                int(n),
            )
            for pos, idx in enumerate(proxy_missing_indices):
                support_proxy[int(idx)] = float(proxy_values[int(pos)])
            proxy_eval_seconds += float(time.perf_counter() - t_proxy)

        accepted_arr = np.asarray(accepted_indices, dtype=np.int64)
        prev_pos = 0
        for checkpoint, mark_pos, actual_unique in record_marks:
            if int(mark_pos) > int(prev_pos):
                residual_sum += _residual_loop_numba(
                    accepted_arr[int(prev_pos):int(mark_pos)],
                    support_utility,
                    support_proxy,
                    rho_abs_counts,
                    rho_indptr,
                    rho_players,
                    rho_signed_counts,
                    n,
                )
                adjustment_total_samples += int(mark_pos) - int(prev_pos)
                prev_pos = int(mark_pos)
            residual_hat = residual_sum / float(max(adjustment_total_samples, 1))
            phi_current = proxy_phi + residual_hat
            checkpoint_targets.append(int(checkpoint))
            checkpoint_actual.append(int(actual_unique))
            checkpoint_are.append(relative_are(phi_current, case.exact_target))
            checkpoint_kind.append("common")
            checkpoint_adjust_samples.append(int(adjustment_total_samples))

        if int(prev_pos) < int(accepted_arr.size):
            residual_sum += _residual_loop_numba(
                accepted_arr[int(prev_pos):],
                support_utility,
                support_proxy,
                rho_abs_counts,
                rho_indptr,
                rho_players,
                rho_signed_counts,
                n,
            )
            adjustment_total_samples += int(accepted_arr.size) - int(prev_pos)

    if int(stats.unique_evals) < int(unique_budget):
        print(
            f"      NOTE: all {max_new_support} rho-support masks evaluated "
            f"(unique={stats.unique_evals}, budget={unique_budget}). "
            f"Cannot reach budget — stopping adjustment early.",
            flush=True,
        )

    B_adjust_unique = max(int(stats.unique_evals) - int(B_train), 0)

    residual_hat = (
        residual_sum / float(adjustment_total_samples)
        if adjustment_total_samples > 0
        else np.zeros(n, dtype=np.float64)
    )
    phi_hat = proxy_phi + residual_hat

    elapsed = float(
        float(rho_support["rho_estimation_seconds"])
        + pair_sampling_seconds
        + (time.perf_counter() - t0)
    )

    final_are = relative_are(phi_hat, case.exact_target)

    out = {
        "method": str(method_name),
        "interaction_percent": float(interaction_percent),
        "final_are": float(final_are),
        "phi_hat": phi_hat,
        "elapsed_seconds": float(elapsed),
        "utility_eval_seconds": float(stats.utility_seconds),
        "non_utility_seconds": float(elapsed - stats.utility_seconds),
        "total_utility_evals": int(stats.total_evals),
        "unique_utility_evals": int(stats.unique_evals),
        "reused_utility_evals": int(stats.total_evals - stats.unique_evals),
        "B": int(unique_budget),
        "B_pool": int(stats.unique_evals),
        "B_train": int(B_train),
        "B_adjust": int(B_adjust_unique),
        "train_subset_budget": int(train_total_samples),
        "train_unique_budget": int(train_unique_budget),
        "surrogate_train_frac": float(train_frac),
        "surrogate_train_cap": int(train_cap),
        "adjustment_total_samples": int(adjustment_total_samples),
        "unique_checkpoint_target_curve": checkpoint_targets,
        "unique_checkpoint_curve": checkpoint_targets,
        "unique_checkpoint_actual_curve": checkpoint_actual,
        "are_by_unique_curve": checkpoint_are,
        "checkpoint_kind_curve": checkpoint_kind,
        "checkpoint_adjustment_sample_curve": checkpoint_adjust_samples,
        "num_proxy_pairs": int(len(pair_list)),
        "pair_permutations": int(pair_samples.shape[0]),
        "rho_permutations": int(rho_support["rho_permutations"]),
        "rho_support_size": int(rho_support["num_support"]),
        "rho_estimation_seconds": float(rho_support["rho_estimation_seconds"]),
        "pair_sampling_seconds": float(pair_sampling_seconds),
        "prob_estimate_seconds": float(prob_seconds),
        "surrogate_fit_seconds": float(fit_seconds),
        "proxy_eval_seconds": float(proxy_eval_seconds),
        "num_cached_subsets": int(len(cache)),
        "proxy_fit_rows": int(len(value_by_subset)),
        "proxy_fit_cols": int(n + len(pair_list)),
        "proxy_fit_active_cols": int(fit_diag.get("proxy_active_cols", 0)),
        "fit_diagnostics": fit_diag,
        "sampling_seconds": float(pair_sampling_seconds),
    }
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_method_summary(label: str, result: Dict[str, object]) -> None:
    are = float(result.get("final_are", float("nan")))
    unique_calls = int(result.get("unique_utility_evals", 0))
    total_calls = int(result.get("total_utility_evals", 0))
    non_utility_seconds = float(result.get("non_utility_seconds", float("nan")))
    utility_seconds = float(result.get("utility_eval_seconds", float("nan")))
    elapsed_seconds = float(result.get("elapsed_seconds", float("nan")))
    lines = (
        f"\n"
        f"      [{label}]\n"
        f"        ARE                 : {are:.6f}\n"
        f"        unique utility calls: {unique_calls}\n"
        f"        total utility calls : {total_calls}\n"
        f"        runtime for others  : {non_utility_seconds:.2f}s\n"
        f"        utility runtime     : {utility_seconds:.2f}s\n"
        f"        total runtime       : {elapsed_seconds:.2f}s"
    )
    # Budget info for surrogate methods
    B = result.get("B")
    if B is not None:
        B_pool = result.get("B_pool")
        B_train = int(result.get("B_train", 0))
        B_adjust = int(result.get("B_adjust", 0))
        adj_samples = int(result.get("adjustment_total_samples", 0))
        lines += (
            f"\n        B (unique budget)   : {int(B)}"
            f"\n        B_train             : {B_train}"
            f"\n        B_adjust            : {B_adjust}"
            f"\n        adjustment samples  : {adj_samples}"
        )
        if B_pool is not None and int(B_pool) != int(B):
            lines += f"\n        B_pool              : {int(B_pool)}"
    for key, label_str in [
        ("sampling_seconds",       "sampling time      "),
        ("pair_sampling_seconds",  "pair sampling time "),
        ("rho_estimation_seconds", "rho estimation time"),
        ("prob_estimate_seconds",  "prob estimate time "),
        ("surrogate_fit_seconds",  "proxy fit time     "),
        ("proxy_eval_seconds",     "proxy eval time    "),
    ]:
        val = result.get(key)
        if val is not None:
            lines += f"\n        {label_str}: {float(val):.2f}s"
    fit_diag = result.get("fit_diagnostics")
    if fit_diag:
        solver = str(fit_diag.get("proxy_solver", "unknown"))
        lsmr_iter = int(fit_diag.get("proxy_lsmr_iterations", 0))
        tracemalloc_mb = fit_diag.get("proxy_tracemalloc_peak_mb", float("nan"))
        peak_delta_mb = fit_diag.get("proxy_fit_peak_delta_mb", float("nan"))
        tm_str = f"{tracemalloc_mb:.1f} MB" if np.isfinite(tracemalloc_mb) else "unknown"
        delta_str = f"{peak_delta_mb:.1f} MB" if np.isfinite(peak_delta_mb) else "unknown"
        solver_line = f"{solver}"
        if solver == "lsmr":
            solver_line += f" ({lsmr_iter} iters)"
        lines += (
            f"\n        proxy solver        : {solver_line}"
            f"\n        proxy fit mem (tm)  : {tm_str}\n"
            f"        proxy fit mem (rss) : {delta_str}"
        )
    print(lines, flush=True)


def print_rep_comparison_table(rep_methods: Sequence[Dict[str, object]]) -> None:
    if not rep_methods:
        return

    rows = []
    for result in rep_methods:
        fit_diag = result.get("fit_diagnostics")
        tm_mb = float(fit_diag["proxy_tracemalloc_peak_mb"]) if fit_diag else float("nan")
        rows.append(
            {
                "method": str(result.get("method", "unknown")),
                "are": float(result.get("final_are", float("nan"))),
                "unique": int(result.get("unique_utility_evals", 0)),
                "total": int(result.get("total_utility_evals", 0)),
                "B_train": int(result.get("B_train", 0)) if "B_train" in result else None,
                "B_adjust": int(result.get("B_adjust", 0)) if "B_adjust" in result else None,
                "elapsed": float(result.get("elapsed_seconds", float("nan"))),
                "fit_mem_mb": tm_mb,
            }
        )

    method_width = max(28, max(len(row["method"]) for row in rows))
    print("\n      [comparison table for this rep]", flush=True)
    print(
        "      "
        + f"{'method':<{method_width}}  "
        + f"{'ARE':>10}  "
        + f"{'unique_U':>10}  "
        + f"{'total_U':>10}  "
        + f"{'B_train':>10}  "
        + f"{'B_adjust':>10}  "
        + f"{'total_time':>13}  "
        + f"{'fit_mem':>10}",
        flush=True,
    )
    print(
        "      "
        + "-" * method_width
        + "  " + "-" * 10
        + "  " + "-" * 10
        + "  " + "-" * 10
        + "  " + "-" * 10
        + "  " + "-" * 10
        + "  " + "-" * 13
        + "  " + "-" * 10,
        flush=True,
    )
    for row in sorted(rows, key=lambda x: (not np.isfinite(x["are"]), x["are"])):
        are_str = f"{row['are']:.6f}" if np.isfinite(row["are"]) else "nan"
        elapsed_str = f"{row['elapsed']:.2f}s" if np.isfinite(row["elapsed"]) else "nan"
        mem_str = f"{row['fit_mem_mb']:.1f}MB" if np.isfinite(row["fit_mem_mb"]) else "none"
        bt_str = str(row["B_train"]) if row["B_train"] is not None else "-"
        ba_str = str(row["B_adjust"]) if row["B_adjust"] is not None else "-"
        print(
            "      "
            + f"{row['method']:<{method_width}}  "
            + f"{are_str:>10}  "
            + f"{row['unique']:>10}  "
            + f"{row['total']:>10}  "
            + f"{bt_str:>10}  "
            + f"{ba_str:>10}  "
            + f"{elapsed_str:>13}  "
            + f"{mem_str:>10}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Summarisation across reps
# ---------------------------------------------------------------------------

def summarize_method_reps(rep_results: List[Dict[str, object]]) -> Dict[str, object]:
    if not rep_results:
        return {}
    summary: Dict[str, object] = {}
    scalar_keys = [
        "final_are",
        "elapsed_seconds",
        "utility_eval_seconds",
        "non_utility_seconds",
        "total_utility_evals",
        "unique_utility_evals",
        "reused_utility_evals",
        "B",
        "B_pool",
        "B_train",
        "B_adjust",
        "adjustment_total_samples",
        "train_subset_budget",
        "surrogate_train_frac",
        "surrogate_train_cap",
        "sampling_seconds",
        "pair_sampling_seconds",
        "prob_estimate_seconds",
        "surrogate_fit_seconds",
        "rho_estimation_seconds",
        "proxy_eval_seconds",
        "num_cached_subsets",
        "proxy_fit_rows",
        "proxy_fit_cols",
        "proxy_fit_active_cols",
    ]
    for key in scalar_keys:
        vals = [float(rep[key]) for rep in rep_results if key in rep and np.isfinite(float(rep[key]))]
        if vals:
            summary[key] = summarize_scalar(vals)
    return summary


def summarize_case_results(rep_results: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for rep in rep_results:
        for method_result in rep["methods"]:
            key = str(method_result["method"])
            grouped.setdefault(key, []).append(method_result)
    return {key: summarize_method_reps(value) for key, value in grouped.items()}


# ---------------------------------------------------------------------------
# Case building (reuse from surrogate.py)
# ---------------------------------------------------------------------------

def build_cases_for_n(
    *,
    n: int,
    args,
    random_state: int,
    shared_scenario2_templates: Optional[Dict[Tuple[str, str], Dict[str, object]]] = None,
) -> List[CaseData]:
    cases: List[CaseData] = []

    if args.scenario in {"all", "scenario1"}:
        for lambda_case in parse_list_arg(args.scenario1_lambda_cases):
            for omega_bar in parse_float_list_arg(args.scenario1_omega_values):
                case_seed = int(random_state) + 200_000 * int(n) + 10_000 * len(cases)
                lambdas = make_lambda_total_order(str(lambda_case), int(n), case_seed + 13)
                interval_rng = np.random.RandomState(case_seed + 29)
                starts, ends, coeffs = sample_uniform_intervals(int(n), int(n) * int(n), interval_rng)
                interval_weights = build_interval_weight_matrix(int(n), starts, ends, coeffs)
                segment_table = build_segment_weight_table(interval_weights)
                exact_target = exact_target_total_order(interval_weights, lambdas, float(omega_bar))
                edges = build_total_order_graph(int(n), float(omega_bar))
                cases.append(
                    CaseData(
                        scenario="scenario1",
                        n=int(n),
                        case_label=f"scenario1_lambda={lambda_case}_omega={omega_bar}",
                        lambdas=lambdas,
                        edges_with_omega=edges,
                        exact_target=exact_target,
                        oracle=TotalOrderOracle(segment_table),
                        meta={
                            "lambda_case": str(lambda_case),
                            "omega_bar": float(omega_bar),
                            "num_terms": int(n) * int(n),
                            "interval_weights": interval_weights,
                        },
                    )
                )

    if args.scenario in {"all", "scenario2"}:
        for lambda_case in parse_list_arg(args.scenario2_lambda_cases):
            for omega_case in parse_list_arg(args.scenario2_omega_cases):
                case_seed = int(random_state) + 300_000 * int(n) + 10_000 * len(cases)
                template_key = (str(lambda_case), str(omega_case))
                shared_template = None
                if shared_scenario2_templates is not None:
                    shared_template = shared_scenario2_templates.get(template_key)
                    if shared_template is None:
                        raise KeyError(
                            f"Missing shared scenario2 graph template for key={template_key}. "
                            "This indicates inconsistent scenario2 case lists."
                        )
                if shared_template is not None:
                    graph = build_block_cycle_graph_from_template(
                        n=int(n),
                        block_size=int(args.scenario2_block_size),
                        block_edges_master=shared_template["block_edges_master"],
                        block_edge_weights_master=shared_template["block_edge_weights_master"],
                        internal_cycle_weights_master=shared_template["internal_cycle_weights_master"],
                    )
                else:
                    graph = build_block_cycle_graph(
                        n=int(n),
                        block_size=int(args.scenario2_block_size),
                        edge_prob=float(args.scenario2_edge_prob),
                        dag_seed=int(args.dag_seed) + int(n),
                        omega_case=str(omega_case),
                        omega_seed=case_seed + 11,
                    )
                coeff_rng = np.random.RandomState(case_seed + 17)
                block_coeffs = coeff_rng.uniform(0.5, 1.5, size=graph["num_blocks"]).astype(np.float64)
                lambdas = make_lambda_block_cycle(str(lambda_case), graph["block_members"], case_seed + 23)
                exact_target = exact_target_block_cycle(block_coeffs, graph["block_members"])
                meta = {
                    "lambda_case": str(lambda_case),
                    "omega_case": str(omega_case),
                    "omega_bounds": omega_case_bounds(str(omega_case)),
                    "num_blocks": int(graph["num_blocks"]),
                    "block_size": int(args.scenario2_block_size),
                    "block_coeffs": block_coeffs,
                    "block_edges": graph["block_edges"],
                    "block_edge_weights": graph["block_edge_weights"],
                    "internal_cycle_weights": graph["internal_cycle_weights"],
                }
                if shared_template is not None:
                    meta["shared_graph_across_n"] = True
                    meta["shared_graph_reference_n"] = int(shared_template["reference_n"])
                    meta["shared_graph_reference_case_seed"] = int(shared_template["reference_case_seed"])
                    meta["shared_graph_reference_dag_seed"] = int(shared_template["reference_dag_seed"])
                cases.append(
                    CaseData(
                        scenario="scenario2",
                        n=int(n),
                        case_label=f"scenario2_lambda={lambda_case}_omega={omega_case}",
                        lambdas=lambdas,
                        edges_with_omega=graph["edges"],
                        exact_target=exact_target,
                        oracle=BlockCycleOracle(graph["block_of"], graph["block_members"], block_coeffs),
                        meta=meta,
                    )
                )

    return cases


# ---------------------------------------------------------------------------
# Main per-case loop
# ---------------------------------------------------------------------------

def run_case_reps(
    case: CaseData,
    *,
    args,
    methods: List[str],
    init_mode: str,
) -> List[Dict[str, object]]:
    n = int(case.n)
    burnin_steps = resolve_burnin_steps(n, float(args.burnin_multiplier), float(args.burnin_exponent))
    num_pair_samples = resolve_num_pair_samples(args, n)
    num_perm_samples = int(args.num_samples)

    print(
        f"  {case.case_label}, init={init_mode}, burnin={burnin_steps}, "
        f"perm_samples={num_perm_samples}, num_pair_samples={num_pair_samples}",
        flush=True,
    )

    results: List[Dict[str, object]] = []

    # Build pair lists for surrogate methods
    pair_list_linear = make_random_pair_list(n, 0.0, int(args.random_state) + 91_000 + n)
    pair_list_quadratic = make_random_pair_list(n, 0.1, int(args.random_state) + 91_000 + n)

    for rep_idx in range(int(args.nreps)):
        rep_seed = int(args.random_state) + 1_000_000 * n + 10_000 * rep_idx + len(results)
        print(f"    rep {rep_idx + 1}/{args.nreps}, seed={rep_seed}", flush=True)

        # ---- Step 1: draw permutation samples ----
        value_samples, acceptance_rate, sampling_seconds = draw_permutations(
            case,
            seed=rep_seed,
            num_samples=int(num_perm_samples),
            burnin_steps=int(burnin_steps),
            steps_between=int(args.steps_between),
            init_mode=str(init_mode),
            report_every=int(args.print_every),
        )
        value_samples = np.asarray(value_samples, dtype=np.int64)

        # ---- Step 2: run permutation method & record B ----
        method_start = time.perf_counter()
        perm_out = run_permutation_method(
            case,
            samples=value_samples,
            sampling_seconds=float(sampling_seconds),
            checkpoints=[num_perm_samples],  # single checkpoint at the end
        )
        perm_out["method_seconds"] = float(time.perf_counter() - method_start)
        B = int(perm_out["unique_utility_evals"])
        unique_checkpoints = make_unique_checkpoint_grid(B, int(args.num_unique_checkpoints))
        perm_out.update(
            replay_permutation_unique_checkpoints(
                case,
                samples=value_samples,
                unique_checkpoints=unique_checkpoints,
            )
        )
        print_method_summary("method permutation", perm_out)
        print(f"      >>> B (permutation unique budget) = {B}", flush=True)

        rep_methods: List[Dict[str, object]] = [perm_out]

        # ---- Step 3: draw pair samples (needed for surrogate methods) ----
        needs_surrogate = ("linear_surrogate_subset" in methods or "quadratic_surrogate_subset" in methods)
        if needs_surrogate:
            pair_initial_permutation = value_samples[-1] if int(value_samples.shape[0]) > 0 else None
            pair_burnin_steps = 0 if pair_initial_permutation is not None else int(burnin_steps)
            if pair_initial_permutation is not None:
                print("      continuing pair/rho sampler from value chain (burnin=0)", flush=True)
            pair_samples, pair_acceptance_rate, pair_sampling_seconds = draw_permutations(
                case,
                seed=rep_seed + 5_000_000,
                num_samples=int(num_pair_samples),
                burnin_steps=int(pair_burnin_steps),
                steps_between=int(args.steps_between),
                init_mode=str(init_mode),
                report_every=int(args.print_every),
                initial_permutation=pair_initial_permutation,
            )
            pair_samples = np.asarray(pair_samples, dtype=np.int64)

            print("      building rho support", flush=True)
            rho_support = build_rho_support(pair_samples, n)
            print(
                f"      rho support={rho_support['num_support']} "
                f"events={rho_support['num_events']} "
                f"time={rho_support['rho_estimation_seconds']:.2f}s",
                flush=True,
            )
        else:
            pair_samples = np.empty((0, n), dtype=np.int64)
            pair_acceptance_rate = float("nan")
            pair_sampling_seconds = 0.0
            rho_support = None

        # ---- Step 4: run linear_surrogate_subset ----
        if "linear_surrogate_subset" in methods and rho_support is not None:
            method_start = time.perf_counter()
            out = run_surrogate_subset_budget_aligned(
                case,
                rho_support=rho_support,
                pair_samples=pair_samples,
                pair_sampling_seconds=float(pair_sampling_seconds),
                pair_list=pair_list_linear,
                unique_budget=B,
                seed=rep_seed + 8_000_000,
                chunk_size=int(args.subset_chunk_size),
                interaction_percent=0.0,
                method_name="linear_surrogate_subset",
                fit_label="linear_surrogate_subset (ip=0)",
                unique_checkpoints=unique_checkpoints,
                train_frac=float(args.surrogate_train_frac),
                train_cap=int(args.surrogate_train_cap),
            )
            out["method_seconds"] = float(time.perf_counter() - method_start)
            rep_methods.append(out)
            print_method_summary("method linear_surrogate_subset", out)

        # ---- Step 5: run quadratic_surrogate_subset ----
        if "quadratic_surrogate_subset" in methods and rho_support is not None and n < int(args.quadratic_max_n):
            method_start = time.perf_counter()
            out = run_surrogate_subset_budget_aligned(
                case,
                rho_support=rho_support,
                pair_samples=pair_samples,
                pair_sampling_seconds=float(pair_sampling_seconds),
                pair_list=pair_list_quadratic,
                unique_budget=B,
                seed=rep_seed + 9_000_000,
                chunk_size=int(args.subset_chunk_size),
                interaction_percent=0.1,
                method_name="quadratic_surrogate_subset",
                fit_label="quadratic_surrogate_subset (ip=0.1)",
                unique_checkpoints=unique_checkpoints,
                train_frac=float(args.surrogate_train_frac),
                train_cap=int(args.surrogate_train_cap),
            )
            out["method_seconds"] = float(time.perf_counter() - method_start)
            rep_methods.append(out)
            print_method_summary("method quadratic_surrogate_subset", out)

        print_rep_comparison_table(rep_methods)

        results.append(
            {
                "rep_idx": int(rep_idx),
                "seed": int(rep_seed),
                "acceptance_rate": float(acceptance_rate),
                "pair_acceptance_rate": float(pair_acceptance_rate) if needs_surrogate else float("nan"),
                "sampling_seconds": float(sampling_seconds),
                "pair_sampling_seconds": float(pair_sampling_seconds) if needs_surrogate else 0.0,
                "B": int(B),
                "unique_checkpoints": [int(x) for x in unique_checkpoints],
                "methods": rep_methods,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHOD_CHOICES = ["permutation", "linear_surrogate_subset", "quadratic_surrogate_subset"]


def parse_methods_new(arg: str) -> List[str]:
    if arg.strip().lower() == "all":
        return list(METHOD_CHOICES)
    methods: List[str] = []
    for token in parse_list_arg(arg):
        key = token.strip().lower()
        if key not in METHOD_CHOICES:
            raise ValueError(f"Unknown method: {token}. Must be one of {METHOD_CHOICES}")
        if key not in methods:
            methods.append(key)
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unique-budget-aligned comparison: permutation vs surrogate_subset."
    )
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "scenario1", "scenario2"])
    parser.add_argument("--n_grid", type=str, default="32,128,512,2048")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument("--nreps", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--burnin_multiplier", type=float, default=1.0)
    parser.add_argument("--burnin_exponent", type=float, default=2.5)
    parser.add_argument("--steps_between", type=int, default=1000)
    parser.add_argument("--init_mode", type=str, default="greedy", choices=["all", "random", "greedy", "topological"])
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--num_pair_samples", type=int, default=None)
    parser.add_argument("--pair_permutation_multiplier", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--pair_permutation_scale",
        type=str,
        default="fixed",
        choices=["fixed", "sqrt_n", "n"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--subset_chunk_size", type=int, default=10000)
    parser.add_argument("--surrogate_train_frac", type=float, default=0.3)
    parser.add_argument("--surrogate_train_cap", type=int, default=50000)
    parser.add_argument("--num_unique_checkpoints", type=int, default=10)
    parser.add_argument("--quadratic_max_n", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--scenario1_lambda_cases", type=str, default="ones,uniform_1_10")
    parser.add_argument("--scenario1_omega_values", type=str, default="0.3,0.7")
    parser.add_argument("--scenario2_block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--scenario2_edge_prob", type=float, default=0.8)
    parser.add_argument("--dag_seed", type=int, default=42)
    parser.add_argument("--scenario2_lambda_cases", type=str, default="ones,block_uniform_1_10")
    parser.add_argument("--scenario2_omega_cases", type=str, default="uniform_0_0.5,uniform_0.5_1")
    parser.add_argument(
        "--scenario2_shared_graph_across_n",
        action="store_true",
        help=(
            "Use one shared scenario2 block-DAG template (at max n in n_grid) and "
            "slice prefix blocks for each n to reduce cross-n structural variation."
        ),
    )
    args = parser.parse_args()
    if args.num_pair_samples is None and args.pair_permutation_multiplier is None:
        args.num_pair_samples = int(DEFAULT_NUM_PAIR_SAMPLES)

    print_system_info()
    warmup_numba()
    n_grid = parse_n_grid(args.n_grid)
    methods = parse_methods_new(args.methods)
    init_modes = parse_init_modes(args.init_mode)
    save_dir = (
        Path(args.save_dir)
        if args.save_dir is not None
        else Path(__file__).resolve().parent / "save" / "surrogate"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "scenario": str(args.scenario),
        "n_grid": list(n_grid),
        "num_samples": int(args.num_samples),
        "print_every": int(args.print_every),
        "nreps": int(args.nreps),
        "random_state": int(args.random_state),
        "burnin_multiplier": float(args.burnin_multiplier),
        "burnin_exponent": float(args.burnin_exponent),
        "steps_between": int(args.steps_between),
        "init_mode": str(args.init_mode),
        "methods": list(methods),
        "num_pair_samples": int(args.num_pair_samples) if args.num_pair_samples is not None else None,
        "subset_chunk_size": int(args.subset_chunk_size),
        "surrogate_train_frac": float(args.surrogate_train_frac),
        "surrogate_train_cap": int(args.surrogate_train_cap),
        "num_unique_checkpoints": int(args.num_unique_checkpoints),
        "quadratic_max_n": int(args.quadratic_max_n),
        "scenario1_lambda_cases": parse_list_arg(args.scenario1_lambda_cases),
        "scenario1_omega_values": parse_float_list_arg(args.scenario1_omega_values),
        "scenario2_block_size": int(args.scenario2_block_size),
        "scenario2_edge_prob": float(args.scenario2_edge_prob),
        "dag_seed": int(args.dag_seed),
        "scenario2_lambda_cases": parse_list_arg(args.scenario2_lambda_cases),
        "scenario2_omega_cases": parse_list_arg(args.scenario2_omega_cases),
        "scenario2_shared_graph_across_n": bool(args.scenario2_shared_graph_across_n),
    }

    print("Surrogate-new experiment configuration (unique-budget-aligned)")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    payload: Dict[str, object] = {"config": config, "results": []}
    case_counter = 0
    shared_scenario2_templates: Optional[Dict[Tuple[str, str], Dict[str, object]]] = None
    if bool(args.scenario2_shared_graph_across_n) and args.scenario in {"all", "scenario2"}:
        shared_scenario2_templates = build_shared_scenario2_graph_templates(
            n_grid=n_grid,
            args=args,
            random_state=int(args.random_state),
        )

    for n in n_grid:
        cases = build_cases_for_n(
            n=int(n),
            args=args,
            random_state=int(args.random_state),
            shared_scenario2_templates=shared_scenario2_templates,
        )
        for case in cases:
            for init_mode in init_modes:
                case_counter += 1
                print(f"[case {case_counter}] n={n} {case.case_label} init={init_mode}", flush=True)
                rep_results = run_case_reps(
                    case,
                    args=args,
                    methods=methods,
                    init_mode=str(init_mode),
                )
                result = {
                    "scenario": case.scenario,
                    "n": int(case.n),
                    "case_label": case.case_label,
                    "init_mode": str(init_mode),
                    "meta": case.meta,
                    "exact_target": case.exact_target,
                    "rep_results": rep_results,
                    "summary": summarize_case_results(rep_results),
                }
                payload["results"].append(result)
                partial_path = save_dir / f"surrogate_partial_n{int(n)}.pkl"
                _save_pickle(partial_path, payload)
                print(f"  saved partial: {partial_path}", flush=True)

    out_path = save_dir / "surrogate.pkl"
    _save_pickle(out_path, payload)
    print()
    print(f"Saved surrogate results to {out_path}")


if __name__ == "__main__":
    main()
