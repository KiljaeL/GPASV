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


DEFAULT_BLOCK_SIZE = 16
DEFAULT_N_GRID = (32, 128, 512, 2048)
DEFAULT_CHECKPOINT_GRID = tuple(range(1000, 20001, 1000))
EPS_OMEGA = 1e-8
EPS_LAMBDA = 1e-12
PROXY_LSMR_COL_THRESHOLD = 4096
PROXY_LSMR_ATOL = 1e-4
PROXY_LSMR_BTOL = 1e-4
PROXY_LSMR_MAXITER = 200
DEFAULT_NUM_PAIR_SAMPLES = 100
METHOD_ALIASES = {
    "all": ("perm", "surrogate_perm", "subset", "surrogate_subset"),
    "perm": ("perm",),
    "surrogate_perm": ("surrogate_perm",),
    "subset": ("subset",),
    "surrogate_subset": ("surrogate_subset",),
}


def parse_n_grid(n_grid_str: str) -> List[int]:
    return [int(x.strip()) for x in str(n_grid_str).split(",") if x.strip()]


def parse_list_arg(arg: str) -> List[str]:
    return [x.strip() for x in str(arg).split(",") if x.strip()]


def parse_float_list_arg(arg: str) -> List[float]:
    return [float(x.strip()) for x in str(arg).split(",") if x.strip()]


def summarize_scalar(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def summarize_curve(curves: np.ndarray) -> Dict[str, List[float]]:
    if curves.ndim != 2:
        raise ValueError(f"Expected 2D curve array, got shape {curves.shape}")
    std = np.zeros(curves.shape[1], dtype=np.float64) if curves.shape[0] == 1 else curves.std(axis=0, ddof=1)
    return {
        "mean": curves.mean(axis=0).astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
    }


def get_cpu_model_name() -> str:
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            with cpuinfo_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return "unknown"


def get_total_memory_gb() -> float:
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            with meminfo_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / (1024.0**2)
        except OSError:
            pass
    return float("nan")


def warmup_numba() -> None:
    """Trigger Numba JIT compilation before any timed experiment code runs."""
    print("Warming up Numba JIT...", flush=True)
    t0 = time.perf_counter()
    _n = 8
    pair_lookup = _build_pair_lookup(_n, [(0, 1), (2, 3)])
    total_cols = _n + 2
    dummy_masks = np.asarray([0b00000011, 0b00001100, 0b00110000], dtype=np.uint64)
    dummy_weights = np.ones(3, dtype=np.float64)
    _build_dense_design_numba_uint64(dummy_masks, dummy_weights, pair_lookup, total_cols)
    dummy_words = _pack_subset_masks_to_words([0b00000011, 0b00001100], _n)
    _build_dense_design_numba_chunks(dummy_words, dummy_weights[:2], pair_lookup, total_cols, _n)
    dummy_pair_i = np.asarray([0, 2], dtype=np.int64)
    dummy_pair_j = np.asarray([1, 3], dtype=np.int64)
    dummy_counts, dummy_active = _count_sparse_design_nnz_and_active(
        dummy_words,
        dummy_pair_i,
        dummy_pair_j,
        _n,
        total_cols,
    )
    dummy_active_cols = np.flatnonzero(dummy_active).astype(np.int32)
    dummy_col_lookup = np.full(total_cols, -1, dtype=np.int32)
    dummy_col_lookup[dummy_active_cols] = np.arange(dummy_active_cols.size, dtype=np.int32)
    dummy_indptr = np.zeros(dummy_words.shape[0] + 1, dtype=np.int64)
    dummy_indptr[1:] = np.cumsum(dummy_counts, dtype=np.int64)
    dummy_indices = np.empty(int(dummy_indptr[-1]), dtype=np.int32)
    dummy_data = np.empty(int(dummy_indptr[-1]), dtype=np.float64)
    _fill_sparse_design_csr(
        dummy_words,
        dummy_weights[:2],
        dummy_pair_i,
        dummy_pair_j,
        dummy_col_lookup,
        dummy_indptr,
        dummy_indices,
        dummy_data,
        _n,
    )
    dummy_a = np.ones(_n, dtype=np.float64)
    _proxy_values_words_numba(
        dummy_words,
        np.float64(0.0),
        dummy_a,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.5], dtype=np.float64),
        _n,
    )
    dummy_b = np.zeros((_n, _n), dtype=np.float64)
    _proxy_value_mask_numba(np.int64(0b00000011), np.float64(0.0), dummy_a, dummy_b, np.int64(_n))
    _residual_loop_numba(
        np.zeros(2, dtype=np.int64),
        np.ones(4, dtype=np.float64),
        np.ones(4, dtype=np.float64),
        np.ones(4, dtype=np.float64),
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.float64),
        _n,
    )
    print(f"  Numba JIT warmup done ({time.perf_counter() - t0:.1f}s)", flush=True)


def print_system_info() -> None:
    cpu_model = get_cpu_model_name()
    total_memory_gb = get_total_memory_gb()
    logical_cores = os.cpu_count()
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    print("System information")
    print(f"  cpu: {cpu_model}")
    print(f"  memory_gb: {total_memory_gb:.2f}" if np.isfinite(total_memory_gb) else "  memory_gb: unknown")
    print(f"  logical_cores: {logical_cores}" if logical_cores is not None else "  logical_cores: unknown")
    if slurm_cpus is not None:
        print(f"  slurm_cpus_per_task: {slurm_cpus}")
    print()


def resolve_burnin_steps(n: int, burnin_multiplier: float, burnin_exponent: float = 2.0) -> int:
    return int(float(burnin_multiplier) * (float(n) ** float(burnin_exponent)))


def normalize_init_mode(init_mode: str) -> str:
    mode = str(init_mode).strip().lower()
    if mode == "topological":
        return "greedy"
    if mode in {"random", "greedy"}:
        return mode
    raise ValueError(f"Unknown init_mode: {init_mode}")


def parse_init_modes(init_mode: str) -> List[str]:
    mode = str(init_mode).strip().lower()
    if mode == "all":
        return ["random", "greedy"]
    return [normalize_init_mode(mode)]


def sample_greedy_order(
    n: int,
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    seed: int,
) -> np.ndarray:
    n = int(n)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    edge_weights: Dict[Tuple[int, int], float] = {}
    incoming: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    scores = np.zeros(n, dtype=np.float64)

    for u, v, omega in edges_with_omega:
        raw_weight = float(-math.log(max(min(float(omega), 1.0), EPS_OMEGA)))
        if raw_weight <= 0.0:
            continue
        edge = (int(u), int(v))
        edge_weights[edge] = edge_weights.get(edge, 0.0) + raw_weight

    for (u, v), weight in edge_weights.items():
        scores[int(u)] += float(weight)
        incoming[int(v)].append((int(u), float(weight)))

    rng = np.random.RandomState(int(seed))
    remaining = np.ones(n, dtype=np.bool_)
    ordering = np.empty(n, dtype=np.int64)

    for pos in range(n - 1, -1, -1):
        active = np.flatnonzero(remaining)
        active_scores = scores[active].astype(np.float64)
        active_lambdas = np.maximum(lambdas[active], EPS_LAMBDA).astype(np.float64)
        log_weights = np.log(active_lambdas) - active_scores
        log_weights -= float(np.max(log_weights))
        choice_weights = np.exp(log_weights)
        weight_sum = float(choice_weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            chosen = int(active[int(rng.randint(0, len(active)))])
        else:
            chosen = int(rng.choice(active, p=choice_weights / weight_sum))
        ordering[pos] = chosen
        remaining[chosen] = False
        for pred, weight in incoming[chosen]:
            if remaining[int(pred)]:
                scores[int(pred)] -= float(weight)

    return ordering


def segment_value(table: np.ndarray, left: int, right: int) -> float:
    if int(left) > int(right):
        return 0.0
    return float(table[int(left), int(right)])


def make_blocks(n: int, block_size: int) -> Tuple[List[np.ndarray], np.ndarray]:
    if int(n) % int(block_size) != 0:
        raise ValueError(f"n={n} must be divisible by block_size={block_size}")
    num_blocks = int(n) // int(block_size)
    members = []
    block_of = np.empty(int(n), dtype=np.int64)
    for block_id in range(num_blocks):
        start = block_id * int(block_size)
        stop = start + int(block_size)
        arr = np.arange(start, stop, dtype=np.int64)
        members.append(arr)
        block_of[start:stop] = int(block_id)
    return members, block_of


def make_random_block_dag(num_blocks: int, edge_prob: float, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.RandomState(int(seed))
    edges: List[Tuple[int, int]] = []
    for src in range(int(num_blocks)):
        for dst in range(src + 1, int(num_blocks)):
            if rng.rand() < float(edge_prob):
                edges.append((int(src), int(dst)))
    return edges


def sample_uniform_intervals(n: int, num_terms: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(int(n), dtype=np.int64)
    counts = (int(n) - starts).astype(np.int64)
    cumulative = np.cumsum(counts)
    total_intervals = int(cumulative[-1])
    draws = rng.randint(0, total_intervals, size=int(num_terms))
    a = np.searchsorted(cumulative, draws, side="right").astype(np.int64)
    prev = np.where(a == 0, 0, cumulative[a - 1])
    offset = draws - prev
    b = a + offset
    coeffs = rng.uniform(0.5, 1.5, size=int(num_terms)).astype(np.float64)
    return a, b.astype(np.int64), coeffs


def build_interval_weight_matrix(n: int, starts: np.ndarray, ends: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    weights = np.zeros((int(n), int(n)), dtype=np.float64)
    for a, b, c in zip(starts, ends, coeffs):
        weights[int(a), int(b)] += float(c)
    return weights


def build_segment_weight_table(interval_weights: np.ndarray) -> np.ndarray:
    n = int(interval_weights.shape[0])
    table = np.zeros((n, n), dtype=np.float64)
    for length in range(1, n + 1):
        for left in range(0, n - length + 1):
            right = left + length - 1
            value = float(interval_weights[left, right])
            if left + 1 <= right:
                value += float(table[left + 1, right])
            if left <= right - 1:
                value += float(table[left, right - 1])
            if left + 1 <= right - 1:
                value -= float(table[left + 1, right - 1])
            table[left, right] = value
    return table


def exact_target_block_cycle(coeffs: np.ndarray, block_members: Sequence[np.ndarray]) -> np.ndarray:
    n = sum(len(m) for m in block_members)
    out = np.zeros(int(n), dtype=np.float64)
    for block_id, members in enumerate(block_members):
        out[np.asarray(members, dtype=np.int64)] = float(coeffs[int(block_id)]) / float(len(members))
    return out


def exact_target_total_order(interval_weights: np.ndarray, lambdas: np.ndarray, omega_bar: float) -> np.ndarray:
    n = int(lambdas.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for a in range(n):
        for b in range(a, n):
            coeff = float(interval_weights[a, b])
            if coeff == 0.0:
                continue
            denom = 0.0
            for r in range(a, b + 1):
                denom += float(lambdas[r]) * (float(omega_bar) ** (b - r))
            scale = coeff / max(denom, EPS_LAMBDA)
            for i in range(a, b + 1):
                out[i] += scale * float(lambdas[i]) * (float(omega_bar) ** (b - i))
    return out


def make_lambda_block_cycle(case: str, block_members: Sequence[np.ndarray], seed: int) -> np.ndarray:
    n = sum(len(m) for m in block_members)
    if case == "ones":
        return np.ones(int(n), dtype=np.float64)
    if case == "block_uniform_1_10":
        rng = np.random.RandomState(int(seed))
        values = rng.uniform(1.0, 10.0, size=len(block_members)).astype(np.float64)
        out = np.zeros(int(n), dtype=np.float64)
        for block_id, members in enumerate(block_members):
            out[np.asarray(members, dtype=np.int64)] = float(values[block_id])
        return out
    raise ValueError(f"Unknown block-cycle lambda case: {case}")


def make_lambda_total_order(case: str, n: int, seed: int) -> np.ndarray:
    if case == "ones":
        return np.ones(int(n), dtype=np.float64)
    if case == "uniform_1_10":
        rng = np.random.RandomState(int(seed))
        return rng.uniform(1.0, 10.0, size=int(n)).astype(np.float64)
    raise ValueError(f"Unknown total-order lambda case: {case}")


def omega_case_bounds(case: str) -> Tuple[float, float]:
    if case == "uniform_0.5_1":
        return 0.5, 1.0
    if case == "uniform_0_0.5":
        return EPS_OMEGA, 0.5
    raise ValueError(f"Unknown omega case: {case}")


def build_block_cycle_graph(
    *,
    n: int,
    block_size: int,
    edge_prob: float,
    dag_seed: int,
    omega_case: str,
    omega_seed: int,
) -> Dict[str, object]:
    block_members, block_of = make_blocks(int(n), int(block_size))
    num_blocks = len(block_members)
    block_edges = make_random_block_dag(num_blocks, float(edge_prob), int(dag_seed))
    omega_low, omega_high = omega_case_bounds(omega_case)
    rng = np.random.RandomState(int(omega_seed))
    block_edge_weights: Dict[Tuple[int, int], float] = {}
    internal_cycle_weights: Dict[int, float] = {}
    for src, dst in block_edges:
        block_edge_weights[(int(src), int(dst))] = float(max(rng.uniform(omega_low, omega_high), EPS_OMEGA))
    for block_id in range(int(num_blocks)):
        internal_cycle_weights[int(block_id)] = float(max(rng.uniform(omega_low, omega_high), EPS_OMEGA))

    edges = []
    internal_cycle_edges: List[Tuple[int, int]] = []
    for block_id, members in enumerate(block_members):
        members = np.asarray(members, dtype=np.int64)
        if int(members.size) <= 1:
            continue
        omega_in = float(internal_cycle_weights[int(block_id)])
        for offset in range(int(members.size)):
            u = int(members[offset])
            v = int(members[(offset + 1) % int(members.size)])
            internal_cycle_edges.append((u, v))
            edges.append((u, v, omega_in))

    for (src, dst), omega in block_edge_weights.items():
        for u in block_members[int(src)]:
            for v in block_members[int(dst)]:
                edges.append((int(u), int(v), float(omega)))

    return {
        "block_members": block_members,
        "block_of": block_of,
        "num_blocks": int(num_blocks),
        "block_edges": block_edges,
        "block_edge_weights": block_edge_weights,
        "internal_cycle_edges": internal_cycle_edges,
        "internal_cycle_weights": internal_cycle_weights,
        "edges": edges,
    }


def build_block_cycle_graph_from_template(
    *,
    n: int,
    block_size: int,
    block_edges_master: Sequence[Tuple[int, int]],
    block_edge_weights_master: Dict[Tuple[int, int], float],
    internal_cycle_weights_master: Dict[int, float],
) -> Dict[str, object]:
    block_members, block_of = make_blocks(int(n), int(block_size))
    num_blocks = len(block_members)
    block_edges = [
        (int(src), int(dst))
        for src, dst in block_edges_master
        if int(src) < int(num_blocks) and int(dst) < int(num_blocks)
    ]
    block_edge_weights: Dict[Tuple[int, int], float] = {}
    internal_cycle_weights: Dict[int, float] = {}
    for src, dst in block_edges:
        block_edge_weights[(int(src), int(dst))] = float(block_edge_weights_master[(int(src), int(dst))])
    for block_id in range(int(num_blocks)):
        internal_cycle_weights[int(block_id)] = float(internal_cycle_weights_master[int(block_id)])

    edges = []
    internal_cycle_edges: List[Tuple[int, int]] = []
    for block_id, members in enumerate(block_members):
        members = np.asarray(members, dtype=np.int64)
        if int(members.size) <= 1:
            continue
        omega_in = float(internal_cycle_weights[int(block_id)])
        for offset in range(int(members.size)):
            u = int(members[offset])
            v = int(members[(offset + 1) % int(members.size)])
            internal_cycle_edges.append((u, v))
            edges.append((u, v, omega_in))

    for (src, dst), omega in block_edge_weights.items():
        for u in block_members[int(src)]:
            for v in block_members[int(dst)]:
                edges.append((int(u), int(v), float(omega)))

    return {
        "block_members": block_members,
        "block_of": block_of,
        "num_blocks": int(num_blocks),
        "block_edges": block_edges,
        "block_edge_weights": block_edge_weights,
        "internal_cycle_edges": internal_cycle_edges,
        "internal_cycle_weights": internal_cycle_weights,
        "edges": edges,
    }


def build_shared_scenario2_graph_templates(
    *,
    n_grid: Sequence[int],
    args,
    random_state: int,
) -> Dict[Tuple[str, str], Dict[str, object]]:
    if not n_grid:
        return {}
    max_n = int(max(int(n) for n in n_grid))
    block_size = int(args.scenario2_block_size)
    if max_n % block_size != 0:
        raise ValueError(
            "scenario2_shared_graph_across_n requires max(n_grid) "
            f"to be divisible by block_size; got max_n={max_n}, block_size={block_size}"
        )

    lambda_cases = parse_list_arg(args.scenario2_lambda_cases)
    omega_cases = parse_list_arg(args.scenario2_omega_cases)
    scenario1_case_count = 0
    if args.scenario in {"all", "scenario1"}:
        scenario1_case_count = (
            len(parse_list_arg(args.scenario1_lambda_cases))
            * len(parse_float_list_arg(args.scenario1_omega_values))
        )

    max_num_blocks = int(max_n) // int(block_size)
    dag_seed_shared = int(args.dag_seed) + int(max_n)
    block_edges_master = make_random_block_dag(max_num_blocks, float(args.scenario2_edge_prob), dag_seed_shared)
    templates: Dict[Tuple[str, str], Dict[str, object]] = {}

    scenario2_pairs = [(str(lc), str(oc)) for lc in lambda_cases for oc in omega_cases]
    for idx, (lambda_case, omega_case) in enumerate(scenario2_pairs):
        case_position = int(scenario1_case_count) + int(idx)
        case_seed = int(random_state) + 300_000 * int(max_n) + 10_000 * int(case_position)
        omega_low, omega_high = omega_case_bounds(str(omega_case))
        rng = np.random.RandomState(int(case_seed) + 11)
        block_edge_weights_master: Dict[Tuple[int, int], float] = {}
        internal_cycle_weights_master: Dict[int, float] = {}
        for src, dst in block_edges_master:
            block_edge_weights_master[(int(src), int(dst))] = float(max(rng.uniform(omega_low, omega_high), EPS_OMEGA))
        for block_id in range(int(max_num_blocks)):
            internal_cycle_weights_master[int(block_id)] = float(max(rng.uniform(omega_low, omega_high), EPS_OMEGA))

        templates[(str(lambda_case), str(omega_case))] = {
            "block_edges_master": list(block_edges_master),
            "block_edge_weights_master": block_edge_weights_master,
            "internal_cycle_weights_master": internal_cycle_weights_master,
            "reference_n": int(max_n),
            "reference_case_seed": int(case_seed),
            "reference_dag_seed": int(dag_seed_shared),
        }

    return templates


def build_total_order_graph(n: int, omega_bar: float) -> List[Tuple[int, int, float]]:
    omega_bar = float(max(omega_bar, EPS_OMEGA))
    return [(int(i), int(j), omega_bar) for i in range(int(n)) for j in range(i + 1, int(n))]


def base_params_to_raw(
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    lambdas = np.asarray(lambdas, dtype=np.float64)
    raw_lambda = -np.log(np.maximum(lambdas, EPS_LAMBDA))
    raw_edges: List[Tuple[int, int, float]] = []
    for u, v, omega in edges_with_omega:
        omega = float(min(max(omega, EPS_OMEGA), 1.0))
        if omega < 1.0:
            raw_edges.append((int(u), int(v), float(-math.log(omega))))
    return raw_lambda.astype(np.float64), raw_edges


def _memory_status_mb() -> Dict[str, float]:
    rss_mb = float("nan")
    peak_rss_mb = float("nan")
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_mb = float(parts[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        peak_rss_mb = float(parts[1]) / 1024.0
    except OSError:
        pass
    if not np.isfinite(peak_rss_mb):
        try:
            peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception:
            peak_rss_mb = float("nan")
    return {"rss_mb": float(rss_mb), "peak_rss_mb": float(peak_rss_mb)}


def _finite_mb_delta(after_mb: float, before_mb: float) -> float:
    if not np.isfinite(after_mb) or not np.isfinite(before_mb):
        return float("nan")
    return float(max(after_mb - before_mb, 0.0))


def _mb_to_gb(value_mb: float) -> float:
    if not np.isfinite(value_mb):
        return float("nan")
    return float(value_mb / 1024.0)


@njit
def _bit_index_uint64(bitmask):
    idx = 0
    while bitmask > 1:
        bitmask = bitmask >> 1
        idx += 1
    return idx


@njit(parallel=True)
def _build_dense_design_numba_uint64(subset_masks, weights, pair_lookup, total_global_cols):
    n = pair_lookup.shape[0]
    num_rows = subset_masks.shape[0]
    X = np.zeros((num_rows, total_global_cols), dtype=np.float32)
    y_scale = np.sqrt(weights)

    for row_idx in prange(num_rows):
        bitmask = subset_masks[row_idx]
        num_singletons = 0
        row_scale = np.float32(y_scale[row_idx])
        singleton_buf = np.empty(n, dtype=np.int64)

        while bitmask != 0:
            lsb = bitmask & (np.uint64(0) - bitmask)
            idx = _bit_index_uint64(lsb)
            singleton_buf[num_singletons] = idx
            num_singletons += 1
            X[row_idx, idx] = row_scale
            bitmask ^= lsb

        for pos_i in range(num_singletons):
            i = singleton_buf[pos_i]
            for pos_j in range(pos_i + 1, num_singletons):
                j = singleton_buf[pos_j]
                global_idx = pair_lookup[i, j]
                if global_idx >= 0:
                    X[row_idx, global_idx] = row_scale
    return X


@njit(parallel=True)
def _build_dense_design_numba_chunks(subset_words, weights, pair_lookup, total_global_cols, n):
    num_rows = subset_words.shape[0]
    num_words = subset_words.shape[1]
    X = np.zeros((num_rows, total_global_cols), dtype=np.float32)
    y_scale = np.sqrt(weights)

    for row_idx in prange(num_rows):
        num_singletons = 0
        row_scale = np.float32(y_scale[row_idx])
        singleton_buf = np.empty(int(n), dtype=np.int64)

        for word_idx in range(num_words):
            word = subset_words[row_idx, word_idx]
            base_idx = 64 * word_idx
            while word != 0:
                lsb = word & (np.uint64(0) - word)
                idx = base_idx + _bit_index_uint64(lsb)
                singleton_buf[num_singletons] = idx
                num_singletons += 1
                X[row_idx, idx] = row_scale
                word ^= lsb

        for pos_i in range(num_singletons):
            i = singleton_buf[pos_i]
            for pos_j in range(pos_i + 1, num_singletons):
                j = singleton_buf[pos_j]
                global_idx = pair_lookup[i, j]
                if global_idx >= 0:
                    X[row_idx, global_idx] = row_scale
    return X


@njit(parallel=True)
def _count_sparse_design_nnz_and_active(subset_words, pair_i, pair_j, n, total_global_cols):
    num_rows = subset_words.shape[0]
    num_words = subset_words.shape[1]
    row_counts = np.zeros(num_rows, dtype=np.int64)
    active_mask = np.zeros(total_global_cols, dtype=np.bool_)

    for row_idx in prange(num_rows):
        count = 0
        for word_idx in range(num_words):
            word = subset_words[row_idx, word_idx]
            base_idx = 64 * word_idx
            while word != 0:
                lsb = word & (np.uint64(0) - word)
                idx = base_idx + _bit_index_uint64(lsb)
                if idx < int(n):
                    count += 1
                    active_mask[idx] = True
                word ^= lsb

        for pair_idx in range(pair_i.shape[0]):
            i = pair_i[pair_idx]
            j = pair_j[pair_idx]
            i_word = i // 64
            j_word = j // 64
            i_bit = i - 64 * i_word
            j_bit = j - 64 * j_word
            i_mask = np.uint64(1) << np.uint64(i_bit)
            j_mask = np.uint64(1) << np.uint64(j_bit)
            if (subset_words[row_idx, i_word] & i_mask) != 0 and (subset_words[row_idx, j_word] & j_mask) != 0:
                count += 1
                active_mask[int(n) + pair_idx] = True

        row_counts[row_idx] = count

    return row_counts, active_mask


@njit(parallel=True)
def _fill_sparse_design_csr(subset_words, weights, pair_i, pair_j, col_lookup, indptr, indices, data, n):
    num_rows = subset_words.shape[0]
    num_words = subset_words.shape[1]
    y_scale = np.sqrt(weights)

    for row_idx in prange(num_rows):
        cursor = indptr[row_idx]
        row_scale = float(y_scale[row_idx])

        for word_idx in range(num_words):
            word = subset_words[row_idx, word_idx]
            base_idx = 64 * word_idx
            while word != 0:
                lsb = word & (np.uint64(0) - word)
                idx = base_idx + _bit_index_uint64(lsb)
                if idx < int(n):
                    local_idx = col_lookup[idx]
                    if local_idx >= 0:
                        indices[cursor] = local_idx
                        data[cursor] = row_scale
                        cursor += 1
                word ^= lsb

        for pair_idx in range(pair_i.shape[0]):
            i = pair_i[pair_idx]
            j = pair_j[pair_idx]
            i_word = i // 64
            j_word = j // 64
            i_bit = i - 64 * i_word
            j_bit = j - 64 * j_word
            i_mask = np.uint64(1) << np.uint64(i_bit)
            j_mask = np.uint64(1) << np.uint64(j_bit)
            if (subset_words[row_idx, i_word] & i_mask) != 0 and (subset_words[row_idx, j_word] & j_mask) != 0:
                local_idx = col_lookup[int(n) + pair_idx]
                if local_idx >= 0:
                    indices[cursor] = local_idx
                    data[cursor] = row_scale
                    cursor += 1


def _build_sparse_design_csr(
    subset_mask_list: Sequence[int],
    weights: np.ndarray,
    pair_list: Sequence[Tuple[int, int]],
    *,
    n: int,
    total_global_cols: int,
) -> Tuple[sps.csr_matrix, np.ndarray]:
    subset_words = _pack_subset_masks_to_words(subset_mask_list, int(n))
    pair_i = np.asarray([int(i) for i, _ in pair_list], dtype=np.int64)
    pair_j = np.asarray([int(j) for _, j in pair_list], dtype=np.int64)

    row_counts, active_mask = _count_sparse_design_nnz_and_active(
        subset_words,
        pair_i,
        pair_j,
        int(n),
        int(total_global_cols),
    )
    active_global_cols = np.flatnonzero(active_mask).astype(np.int32, copy=False)
    col_lookup = np.full(int(total_global_cols), -1, dtype=np.int32)
    col_lookup[active_global_cols.astype(np.int64)] = np.arange(active_global_cols.size, dtype=np.int32)

    indptr = np.zeros(subset_words.shape[0] + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_counts, dtype=np.int64)
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    _fill_sparse_design_csr(
        subset_words,
        weights,
        pair_i,
        pair_j,
        col_lookup,
        indptr,
        indices,
        data,
        int(n),
    )
    X_sparse = sps.csr_matrix(
        (data, indices, indptr),
        shape=(subset_words.shape[0], int(active_global_cols.size)),
        copy=False,
    )
    return X_sparse, active_global_cols


def _build_pair_lookup(n: int, pair_list: Sequence[Tuple[int, int]]) -> np.ndarray:
    pair_lookup = np.full((int(n), int(n)), -1, dtype=np.int32)
    for offset, (i, j) in enumerate(pair_list):
        global_idx = int(n) + int(offset)
        pair_lookup[int(i), int(j)] = global_idx
        pair_lookup[int(j), int(i)] = global_idx
    return pair_lookup


def _pack_subset_masks_to_words(subset_masks: Sequence[int], n: int) -> np.ndarray:
    num_words = (int(n) + 63) // 64
    packed = np.zeros((len(subset_masks), num_words), dtype=np.uint64)
    full_mask = (1 << 64) - 1
    for row_idx, subset_mask in enumerate(subset_masks):
        bitmask = int(subset_mask)
        for word_idx in range(num_words):
            packed[row_idx, word_idx] = np.uint64(bitmask & full_mask)
            bitmask >>= 64
    return packed


@njit(parallel=True)
def _proxy_values_words_numba(subset_words, empty_value, a_hat, pair_i, pair_j, pair_coefs, n):
    num_rows = subset_words.shape[0]
    num_words = subset_words.shape[1]
    out = np.empty(num_rows, dtype=np.float64)
    for row_idx in prange(num_rows):
        total = float(empty_value)
        for word_idx in range(num_words):
            word = subset_words[row_idx, word_idx]
            base_idx = 64 * word_idx
            while word != 0:
                lsb = word & (np.uint64(0) - word)
                idx = base_idx + _bit_index_uint64(lsb)
                if idx < int(n):
                    total += a_hat[idx]
                word ^= lsb

        for pair_idx in range(pair_i.shape[0]):
            i = pair_i[pair_idx]
            j = pair_j[pair_idx]
            i_word = i // 64
            j_word = j // 64
            i_bit = i - 64 * i_word
            j_bit = j - 64 * j_word
            i_mask = np.uint64(1) << np.uint64(i_bit)
            j_mask = np.uint64(1) << np.uint64(j_bit)
            if (subset_words[row_idx, i_word] & i_mask) != 0 and (subset_words[row_idx, j_word] & j_mask) != 0:
                total += pair_coefs[pair_idx]
        out[row_idx] = total
    return out


@njit
def _estimate_pairwise_probs_for_pairs_numba(
    perms: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n: int,
) -> np.ndarray:
    probs = np.zeros(pair_i.shape[0], dtype=np.float64)
    pos = np.empty(int(n), dtype=np.int64)

    for row_idx in range(perms.shape[0]):
        perm = perms[row_idx]
        for k in range(int(n)):
            pos[perm[k]] = k
        for idx in range(pair_i.shape[0]):
            if pos[pair_i[idx]] > pos[pair_j[idx]]:
                probs[idx] += 1.0

    if perms.shape[0] > 0:
        probs /= float(perms.shape[0])
    return probs


def estimate_pairwise_probs_for_pairs(
    perms: np.ndarray,
    pair_list: Sequence[Tuple[int, int]],
    n: int,
) -> np.ndarray:
    if len(pair_list) == 0:
        return np.zeros(0, dtype=np.float64)
    pair_i = np.asarray([int(i) for i, _ in pair_list], dtype=np.int64)
    pair_j = np.asarray([int(j) for _, j in pair_list], dtype=np.int64)
    return _estimate_pairwise_probs_for_pairs_numba(perms, pair_i, pair_j, int(n))


def fit_quadratic_proxy(
    value_by_subset: Dict[int, float],
    weight_by_subset: Dict[int, float],
    *,
    n: int,
    empty_value: float,
    pair_list: Sequence[Tuple[int, int]],
    print_label: str = "",
    return_diagnostics: bool = False,
):
    mem_start = _memory_status_mb()
    if return_diagnostics:
        tracemalloc.start()
    t_build_start = time.perf_counter()
    total_global_cols = int(n) + len(pair_list)

    subset_mask_list = list(value_by_subset.keys())
    y = np.asarray(
        [float(value_by_subset[int(mask)] - empty_value) for mask in subset_mask_list],
        dtype=np.float64,
    )
    weights = np.asarray(
        [max(float(weight_by_subset[int(mask)]), 1e-12) for mask in subset_mask_list],
        dtype=np.float64,
    )
    num_rows = int(len(subset_mask_list))
    y_scaled = np.sqrt(weights) * y
    del y

    X_sparse, active_global_cols = _build_sparse_design_csr(
        subset_mask_list,
        weights,
        pair_list,
        n=int(n),
        total_global_cols=int(total_global_cols),
    )
    del weights
    del subset_mask_list

    num_active_cols = int(active_global_cols.size)
    proxy_solver = "lsmr"
    mem_after_build = _memory_status_mb()
    build_seconds = float(time.perf_counter() - t_build_start)

    if print_label:
        print(f"      [{print_label}] rows={num_rows} cols={num_active_cols} solver={proxy_solver}", flush=True)

    system_seconds = 0.0
    solve_seconds = 0.0
    mem_after_system = dict(mem_after_build)
    mem_after_solve = dict(mem_after_build)
    ridge = 0.0
    lsmr_info = None
    if num_active_cols == 0:
        coef_active = np.zeros(0, dtype=np.float64)
        del X_sparse
    else:
        t_system_start = time.perf_counter()
        col_norm_sq = np.asarray(X_sparse.power(2).sum(axis=0)).ravel()
        diag_scale = float(np.mean(col_norm_sq)) if col_norm_sq.size > 0 else 1.0
        ridge = max(1e-10, 1e-10 * diag_scale)
        mem_after_system = _memory_status_mb()
        system_seconds += float(time.perf_counter() - t_system_start)

        t_solve_start = time.perf_counter()
        lsmr_info = spsl.lsmr(
            X_sparse,
            y_scaled,
            damp=float(math.sqrt(ridge)),
            atol=PROXY_LSMR_ATOL,
            btol=PROXY_LSMR_BTOL,
            maxiter=PROXY_LSMR_MAXITER,
        )
        coef_active = np.asarray(lsmr_info[0], dtype=np.float64)
        solve_seconds += float(time.perf_counter() - t_solve_start)
        mem_after_solve = _memory_status_mb()
        del X_sparse

    t_unpack_start = time.perf_counter()
    coef = np.zeros(total_global_cols, dtype=np.float64)
    coef[active_global_cols.astype(np.int64)] = coef_active
    a_hat = coef[: int(n)].astype(np.float64)
    b_hat = np.zeros((int(n), int(n)), dtype=np.float64)
    for offset, (i, j) in enumerate(pair_list):
        val = float(coef[int(n) + offset])
        b_hat[int(i), int(j)] = val
        b_hat[int(j), int(i)] = val
    unpack_seconds = float(time.perf_counter() - t_unpack_start)
    mem_after_unpack = _memory_status_mb()
    if return_diagnostics:
        _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        tracemalloc_peak_bytes = 0

    if not return_diagnostics:
        return a_hat, b_hat
    return a_hat, b_hat, {
        "proxy_build_seconds": float(build_seconds),
        "proxy_system_seconds": float(system_seconds),
        "proxy_solve_seconds": float(solve_seconds),
        "proxy_unpack_seconds": float(unpack_seconds),
        "proxy_build_rss_mb": float(mem_after_build["rss_mb"]),
        "proxy_system_rss_mb": float(mem_after_system["rss_mb"]),
        "proxy_solve_rss_mb": float(mem_after_solve["rss_mb"]),
        "proxy_unpack_rss_mb": float(mem_after_unpack["rss_mb"]),
        "proxy_build_peak_rss_mb": float(mem_after_build["peak_rss_mb"]),
        "proxy_system_peak_rss_mb": float(mem_after_system["peak_rss_mb"]),
        "proxy_solve_peak_rss_mb": float(mem_after_solve["peak_rss_mb"]),
        "proxy_unpack_peak_rss_mb": float(mem_after_unpack["peak_rss_mb"]),
        "proxy_build_peak_delta_mb": _finite_mb_delta(
            float(mem_after_build["peak_rss_mb"]),
            float(mem_start["peak_rss_mb"]),
        ),
        "proxy_system_peak_delta_mb": _finite_mb_delta(
            float(mem_after_system["peak_rss_mb"]),
            float(mem_start["peak_rss_mb"]),
        ),
        "proxy_solve_peak_delta_mb": _finite_mb_delta(
            float(mem_after_solve["peak_rss_mb"]),
            float(mem_start["peak_rss_mb"]),
        ),
        "proxy_unpack_peak_delta_mb": _finite_mb_delta(
            float(mem_after_unpack["peak_rss_mb"]),
            float(mem_start["peak_rss_mb"]),
        ),
        "proxy_fit_rss_mb": float(mem_after_unpack["rss_mb"]),
        "proxy_fit_peak_rss_mb": float(mem_after_unpack["peak_rss_mb"]),
        "proxy_fit_peak_delta_mb": _finite_mb_delta(
            float(mem_after_unpack["peak_rss_mb"]),
            float(mem_start["peak_rss_mb"]),
        ),
        "proxy_build_rss_gb": _mb_to_gb(float(mem_after_build["rss_mb"])),
        "proxy_system_rss_gb": _mb_to_gb(float(mem_after_system["rss_mb"])),
        "proxy_solve_rss_gb": _mb_to_gb(float(mem_after_solve["rss_mb"])),
        "proxy_unpack_rss_gb": _mb_to_gb(float(mem_after_unpack["rss_mb"])),
        "proxy_fit_rss_gb": _mb_to_gb(float(mem_after_unpack["rss_mb"])),
        "proxy_active_cols": int(num_active_cols),
        "proxy_solver": proxy_solver,
        "proxy_ridge": float(ridge),
        "proxy_lsmr_iterations": int(lsmr_info[2]) if lsmr_info is not None else 0,
        "proxy_lsmr_stop_code": int(lsmr_info[1]) if lsmr_info is not None else 0,
        "proxy_lsmr_normr": float(lsmr_info[3]) if lsmr_info is not None else float("nan"),
        "proxy_lsmr_normar": float(lsmr_info[4]) if lsmr_info is not None else float("nan"),
        "proxy_tracemalloc_peak_mb": float(tracemalloc_peak_bytes) / (1024.0 ** 2),
        "proxy_tracemalloc_peak_gb": float(tracemalloc_peak_bytes) / (1024.0 ** 3),
    }


def proxy_value_from_pair_probs(
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    pair_list: Sequence[Tuple[int, int]],
    pair_probs: np.ndarray,
) -> np.ndarray:
    phi_hat = np.asarray(a_hat, dtype=np.float64).copy()
    for idx, (i, j) in enumerate(pair_list):
        b_val = float(b_hat[int(i), int(j)])
        p_ij = float(pair_probs[idx])
        phi_hat[int(i)] += p_ij * b_val
        phi_hat[int(j)] += (1.0 - p_ij) * b_val
    return phi_hat


def build_proxy_incremental_updates(
    n: int,
    pair_list: Sequence[Tuple[int, int]],
    b_hat: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    neighbor_indices: List[List[int]] = [[] for _ in range(int(n))]
    neighbor_weights: List[List[float]] = [[] for _ in range(int(n))]
    for i_raw, j_raw in pair_list:
        i = int(i_raw)
        j = int(j_raw)
        weight = float(b_hat[i, j])
        if weight == 0.0:
            continue
        neighbor_indices[i].append(j)
        neighbor_weights[i].append(weight)
        neighbor_indices[j].append(i)
        neighbor_weights[j].append(weight)
    return (
        [np.asarray(indices, dtype=np.int32) for indices in neighbor_indices],
        [np.asarray(weights, dtype=np.float64) for weights in neighbor_weights],
    )


@dataclass
class UtilityStats:
    utility_seconds: float = 0.0
    total_evals: int = 0
    unique_evals: int = 0


@dataclass
class CaseData:
    scenario: str
    n: int
    case_label: str
    lambdas: np.ndarray
    edges_with_omega: Sequence[Tuple[int, int, float]]
    exact_target: np.ndarray
    meta: Dict[str, object]
    oracle: object


class TotalOrderOracle:
    def __init__(self, segment_table: np.ndarray):
        self.segment_table = np.asarray(segment_table, dtype=np.float64)
        self.empty_value = 0.0
        self.n = int(self.segment_table.shape[0])

    def utility_from_mask(self, subset_mask: int) -> float:
        bitmask = int(subset_mask)
        total = 0.0
        idx = 0
        while idx < self.n:
            if (bitmask >> idx) & 1:
                left = idx
                while idx + 1 < self.n and ((bitmask >> (idx + 1)) & 1):
                    idx += 1
                right = idx
                total += segment_value(self.segment_table, left, right)
            idx += 1
        return float(total)

    def iter_prefix_values(
        self,
        perm: np.ndarray,
        cache: Dict[int, float],
        stats: UtilityStats,
    ) -> Iterable[Tuple[int, int, float, float]]:
        seen = np.zeros(self.n, dtype=np.bool_)
        run_left = np.arange(self.n, dtype=np.int64)
        run_right = np.arange(self.n, dtype=np.int64)
        subset_mask = 0
        prev_u = 0.0

        for player_raw in perm:
            player = int(player_raw)
            left = player
            right = player
            if player > 0 and seen[player - 1]:
                left = int(run_left[player - 1])
            if player + 1 < self.n and seen[player + 1]:
                right = int(run_right[player + 1])

            subset_mask |= 1 << player
            stats.total_evals += 1
            current_u = cache.get(int(subset_mask))
            if current_u is None:
                t_util = time.perf_counter()
                marginal = segment_value(self.segment_table, left, right)
                marginal -= segment_value(self.segment_table, left, player - 1)
                marginal -= segment_value(self.segment_table, player + 1, right)
                current_u = float(prev_u + marginal)
                stats.utility_seconds += float(time.perf_counter() - t_util)
                stats.unique_evals += 1
                cache[int(subset_mask)] = float(current_u)
            seen[player] = True
            run_left[player] = left
            run_right[player] = right
            run_right[left] = right
            run_left[right] = left
            yield player, int(subset_mask), float(current_u), float(prev_u)
            prev_u = float(current_u)


class BlockCycleOracle:
    def __init__(self, block_of: np.ndarray, block_members: Sequence[np.ndarray], block_coeffs: np.ndarray):
        self.block_of = np.asarray(block_of, dtype=np.int64)
        self.block_members = [np.asarray(m, dtype=np.int64) for m in block_members]
        self.block_coeffs = np.asarray(block_coeffs, dtype=np.float64)
        self.num_blocks = len(self.block_members)
        self.block_size = int(len(self.block_members[0]))
        self.block_masks = []
        for members in self.block_members:
            mask = 0
            for player in members:
                mask |= 1 << int(player)
            self.block_masks.append(int(mask))
        self.empty_value = 0.0

    def utility_from_mask(self, subset_mask: int) -> float:
        mask = int(subset_mask)
        total = 0.0
        for block_id, block_mask in enumerate(self.block_masks):
            if (mask & block_mask) == block_mask:
                total += float(self.block_coeffs[block_id])
        return float(total)

    def iter_prefix_values(
        self,
        perm: np.ndarray,
        cache: Dict[int, float],
        stats: UtilityStats,
    ) -> Iterable[Tuple[int, int, float, float]]:
        counts = np.zeros(self.num_blocks, dtype=np.int64)
        subset_mask = 0
        prev_u = 0.0
        for player_raw in perm:
            player = int(player_raw)
            subset_mask |= 1 << player
            block_id = int(self.block_of[player])
            counts[block_id] += 1
            stats.total_evals += 1
            current_u = cache.get(int(subset_mask))
            if current_u is None:
                t_util = time.perf_counter()
                current_u = float(prev_u)
                if counts[block_id] == self.block_size:
                    current_u += float(self.block_coeffs[block_id])
                stats.utility_seconds += float(time.perf_counter() - t_util)
                stats.unique_evals += 1
                cache[int(subset_mask)] = float(current_u)
            yield player, int(subset_mask), float(current_u), float(prev_u)
            prev_u = float(current_u)


def get_cached_utility(oracle, cache: Dict[int, float], subset_mask: int, stats: UtilityStats) -> float:
    stats.total_evals += 1
    key = int(subset_mask)
    value = cache.get(key)
    if value is not None:
        return float(value)
    t0 = time.perf_counter()
    value = float(oracle.utility_from_mask(key))
    stats.utility_seconds += float(time.perf_counter() - t0)
    stats.unique_evals += 1
    cache[key] = float(value)
    return float(value)


def _save_pickle(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def parse_methods(arg: str) -> List[str]:
    methods: List[str] = []
    for token in parse_list_arg(arg):
        key = token.lower()
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unknown method: {token}")
        for method in METHOD_ALIASES[key]:
            if method not in methods:
                methods.append(method)
    return methods


def make_checkpoint_grid(num_samples: int, checkpoint_every: int) -> List[int]:
    return [ckpt for ckpt in range(int(checkpoint_every), int(num_samples) + 1, int(checkpoint_every))]


def resolve_pair_count(n: int, interaction_percent: float) -> int:
    total_pairs = int(n) * (int(n) - 1) // 2
    if float(interaction_percent) <= 0.0:
        return 0
    if float(interaction_percent) >= 1.0:
        return total_pairs
    return int(math.ceil(float(interaction_percent) * float(total_pairs)))


def pair_rank_to_pairs(n: int, ranks: np.ndarray) -> List[Tuple[int, int]]:
    ranks = np.sort(np.asarray(ranks, dtype=np.int64))
    out: List[Tuple[int, int]] = []
    i = 0
    start = 0
    count = int(n) - 1
    for rank_raw in ranks:
        rank = int(rank_raw)
        while count > 0 and rank >= start + count:
            start += count
            i += 1
            count = int(n) - i - 1
        j = i + 1 + (rank - start)
        out.append((int(i), int(j)))
    return out


def make_random_pair_list(n: int, interaction_percent: float, seed: int) -> List[Tuple[int, int]]:
    pair_count = resolve_pair_count(int(n), float(interaction_percent))
    total_pairs = int(n) * (int(n) - 1) // 2
    if pair_count <= 0:
        return []
    if pair_count >= total_pairs:
        ranks = np.arange(total_pairs, dtype=np.int64)
    else:
        rng = np.random.RandomState(int(seed))
        ranks = rng.choice(total_pairs, size=pair_count, replace=False).astype(np.int64)
    return pair_rank_to_pairs(int(n), ranks)


def resolve_pair_permutation_count(n: int, multiplier: float, scale: str) -> int:
    multiplier = float(multiplier)
    if multiplier <= 0.0:
        return 0
    if scale == "n":
        base = float(n)
    elif scale == "sqrt_n":
        base = math.sqrt(float(n))
    elif scale == "fixed":
        base = 1.0
    else:
        raise ValueError(f"Unknown pair_permutation_scale: {scale}")
    return max(1, int(math.ceil(multiplier * base)))


def resolve_num_pair_samples(args, n: int) -> int:
    if args.num_pair_samples is not None:
        return max(0, int(args.num_pair_samples))
    if args.pair_permutation_multiplier is not None:
        return resolve_pair_permutation_count(
            int(n),
            float(args.pair_permutation_multiplier),
            str(args.pair_permutation_scale),
        )
    return int(DEFAULT_NUM_PAIR_SAMPLES)


def draw_permutations(
    case: CaseData,
    *,
    seed: int,
    num_samples: int,
    burnin_steps: int,
    steps_between: int,
    init_mode: str,
    report_every: int,
    initial_permutation=None,
) -> Tuple[np.ndarray, float, float]:
    raw_lambda, raw_edges = base_params_to_raw(case.lambdas, case.edges_with_omega)
    if initial_permutation is not None:
        init_pi = np.asarray(initial_permutation, dtype=np.int64)
    elif normalize_init_mode(init_mode) == "greedy":
        init_pi = sample_greedy_order(
            int(case.n),
            np.asarray(case.lambdas, dtype=np.float64),
            case.edges_with_omega,
            int(seed),
        )
    else:
        init_pi = None
    t0 = time.perf_counter()
    samples, acceptance_rate = mh_sampler(
        n=int(case.n),
        edges=raw_edges,
        lam=raw_lambda.tolist(),
        alpha=1.0,
        beta=1.0,
        seed=int(seed),
        n_samples=int(num_samples),
        burn_in=int(burnin_steps),
        thin=int(steps_between),
        init_pi=init_pi,
        report_every=int(report_every),
    )
    return samples, float(acceptance_rate), float(time.perf_counter() - t0)


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
            "method": "perm",
            "num_samples": int(samples.shape[0]),
            "sampling_seconds": float(sampling_seconds),
            "num_cached_subsets": int(len(cache)),
        }
    )
    return out


def collect_prefix_training_data(
    case: CaseData,
    *,
    train_samples: np.ndarray,
    cache: Dict[int, float],
    stats: UtilityStats,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    value_by_subset: Dict[int, float] = {}
    weight_by_subset: Dict[int, float] = {}
    for perm in train_samples:
        for _player, mask, current_u, _prev_u in case.oracle.iter_prefix_values(perm, cache, stats):
            if mask not in value_by_subset:
                value_by_subset[int(mask)] = float(current_u)
                weight_by_subset[int(mask)] = 1.0
            else:
                weight_by_subset[int(mask)] += 1.0
    return value_by_subset, weight_by_subset


@njit
def _proxy_value_mask_numba(subset_mask, empty_value, a_hat, b_hat, n):
    total = empty_value
    num_active = 0
    active_buf = np.empty(n, dtype=np.int64)
    bitmask = subset_mask
    while bitmask != np.int64(0):
        lsb = bitmask & -bitmask
        idx = np.int64(0)
        tmp = lsb
        while tmp > np.int64(1):
            tmp >>= np.int64(1)
            idx += np.int64(1)
        active_buf[num_active] = idx
        total += a_hat[idx]
        num_active += 1
        bitmask ^= lsb
    for pi in range(num_active):
        i = active_buf[pi]
        for pj in range(pi + 1, num_active):
            j = active_buf[pj]
            total += b_hat[i, j]
    return total


def proxy_value_mask(
    subset_mask: int,
    *,
    empty_value: float,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    pair_list: Sequence[Tuple[int, int]],
) -> float:
    return float(_proxy_value_mask_numba(
        np.int64(subset_mask),
        np.float64(empty_value),
        a_hat,
        b_hat,
        np.int64(len(a_hat)),
    ))


def run_surrogate_permutation_method(
    case: CaseData,
    *,
    train_samples: np.ndarray,
    residual_samples: np.ndarray,
    pair_samples: np.ndarray,
    pair_list: Sequence[Tuple[int, int]],
    sampling_seconds: float,
    pair_sampling_seconds: float,
    checkpoints: Sequence[int],
    train_permutation_budget: int,
    interaction_percent: float,
    train_multiplier: float,
    fit_label: str,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    n = int(case.n)
    cache: Dict[int, float] = {0: 0.0}
    stats = UtilityStats()
    value_by_subset, weight_by_subset = collect_prefix_training_data(
        case,
        train_samples=train_samples,
        cache=cache,
        stats=stats,
    )

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
    neighbor_indices, neighbor_weights = build_proxy_incremental_updates(n, pair_list, b_hat)

    residual_sum = np.zeros(n, dtype=np.float64)
    contrib = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=np.bool_)
    phi_hats: List[np.ndarray] = []
    time_curve: List[float] = []
    utility_time_curve: List[float] = []
    total_eval_curve: List[int] = []
    unique_eval_curve: List[int] = []
    checkpoint_set = set(int(x) for x in checkpoints)

    if int(train_permutation_budget) in checkpoint_set:
        phi_hats.append(proxy_phi.copy())
        elapsed = float(sampling_seconds + pair_sampling_seconds + (time.perf_counter() - t0))
        time_curve.append(elapsed)
        utility_time_curve.append(float(stats.utility_seconds))
        total_eval_curve.append(int(stats.total_evals))
        unique_eval_curve.append(int(stats.unique_evals))

    for sample_idx, perm in enumerate(residual_samples, start=1):
        contrib.fill(0.0)
        seen.fill(False)
        for player, _mask, current_u, prev_u in case.oracle.iter_prefix_values(perm, cache, stats):
            proxy_marg = float(a_hat[int(player)]) + float(contrib[int(player)])
            residual_sum[int(player)] += float(current_u - prev_u - proxy_marg)

            seen[int(player)] = True
            nbr_idx = neighbor_indices[int(player)]
            if nbr_idx.size > 0:
                unseen_mask = ~seen[nbr_idx]
                if np.any(unseen_mask):
                    contrib[nbr_idx[unseen_mask]] += neighbor_weights[int(player)][unseen_mask]

        used_permutation_budget = int(train_permutation_budget + sample_idx)
        if used_permutation_budget in checkpoint_set:
            residual_hat = residual_sum / float(max(sample_idx, 1))
            phi_hats.append(proxy_phi + residual_hat)
            elapsed = float(sampling_seconds + pair_sampling_seconds + (time.perf_counter() - t0))
            time_curve.append(elapsed)
            utility_time_curve.append(float(stats.utility_seconds))
            total_eval_curve.append(int(stats.total_evals))
            unique_eval_curve.append(int(stats.unique_evals))

    used_checkpoints = [ckpt for ckpt in checkpoints if ckpt >= int(train_permutation_budget)]
    out = summarize_trial_curve(
        exact_target=case.exact_target,
        checkpoints=used_checkpoints,
        phi_hats=phi_hats,
        time_curve=time_curve,
        utility_time_curve=utility_time_curve,
        total_eval_curve=total_eval_curve,
        unique_eval_curve=unique_eval_curve,
    )
    out.update(
        {
            "method": "surrogate_perm",
            "train_multiplier": float(train_multiplier),
            "interaction_percent": float(interaction_percent),
            "num_proxy_pairs": int(len(pair_list)),
            "train_permutation_budget": int(train_permutation_budget),
            "train_raw_utility_budget": int(train_permutation_budget * n),
            "pair_permutations": int(pair_samples.shape[0]),
            "train_permutations": int(train_samples.shape[0]),
            "residual_permutations": int(residual_samples.shape[0]),
            "sampling_seconds": float(sampling_seconds),
            "pair_sampling_seconds": float(pair_sampling_seconds),
            "prob_estimate_seconds": float(prob_seconds),
            "surrogate_fit_seconds": float(fit_seconds),
            "proxy_fit_rows": int(len(value_by_subset)),
            "proxy_fit_cols": int(n + len(pair_list)),
            "proxy_fit_active_cols": int(fit_diag.get("proxy_active_cols", 0)),
            "fit_diagnostics": fit_diag,
            "num_cached_subsets": int(len(cache)),
        }
    )
    return out


def build_rho_support(
    perms: np.ndarray,
    n: int,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    rho_counts: Dict[int, Dict[int, int]] = {}
    abs_counts: Dict[int, int] = {}
    square_counts: Dict[int, int] = {}
    player_bits = [1 << player for player in range(int(n))]

    for perm in perms:
        mask = 0
        for player_raw in perm:
            player = int(player_raw)
            before = int(mask)
            d_before = rho_counts.get(before)
            if d_before is None:
                d_before = {}
                rho_counts[before] = d_before
            old = int(d_before.get(player, 0))
            d_before[player] = old - 1
            abs_counts[before] = int(abs_counts.get(before, 0)) + 1
            square_counts[before] = int(square_counts.get(before, 0)) - 2 * old + 1

            mask |= player_bits[player]
            after = int(mask)
            d_after = rho_counts.get(after)
            if d_after is None:
                d_after = {}
                rho_counts[after] = d_after
            old = int(d_after.get(player, 0))
            d_after[player] = old + 1
            abs_counts[after] = int(abs_counts.get(after, 0)) + 1
            square_counts[after] = int(square_counts.get(after, 0)) + 2 * old + 1

    masks = list(rho_counts.keys())
    weights = np.asarray([float(abs_counts[m]) for m in masks], dtype=np.float64)
    total_weight = float(weights.sum())
    probs = weights / max(total_weight, 1.0)

    # CSR 형태로 rho_counts 변환 → Numba residual 루프에서 사용
    # rho_indptr[i] ~ rho_indptr[i+1]: mask i의 (player, signed_count) 범위
    num_masks = len(masks)
    rho_abs_counts = np.asarray([float(abs_counts[m]) for m in masks], dtype=np.float64)
    rho_indptr = np.zeros(num_masks + 1, dtype=np.int64)
    for i, m in enumerate(masks):
        rho_indptr[i + 1] = rho_indptr[i] + len(rho_counts[m])
    total_entries = int(rho_indptr[-1])
    rho_players = np.empty(total_entries, dtype=np.int64)
    rho_signed_counts = np.empty(total_entries, dtype=np.float64)
    for i, m in enumerate(masks):
        start = int(rho_indptr[i])
        for k, (player, sc) in enumerate(rho_counts[m].items()):
            rho_players[start + k] = int(player)
            rho_signed_counts[start + k] = float(sc)

    return {
        "rho_counts": rho_counts,
        "abs_counts": abs_counts,
        "square_counts": square_counts,
        "masks": masks,
        "probs": probs,
        "num_support": int(len(masks)),
        "num_events": int(2 * int(n) * int(perms.shape[0])),
        "rho_permutations": int(perms.shape[0]),
        "rho_estimation_seconds": float(time.perf_counter() - t0),
        # CSR arrays for Numba
        "rho_abs_counts": rho_abs_counts,
        "rho_indptr": rho_indptr,
        "rho_players": rho_players,
        "rho_signed_counts": rho_signed_counts,
    }


def sample_q_indices(rng: np.random.RandomState, probs: np.ndarray, count: int, chunk_size: int) -> Iterable[np.ndarray]:
    remaining = int(count)
    while remaining > 0:
        size = min(int(chunk_size), remaining)
        yield rng.choice(int(probs.shape[0]), size=size, replace=True, p=probs)
        remaining -= size


def gamma_scale(n: int, abs_count: int) -> float:
    return float(2.0 * float(n) / max(float(abs_count), 1.0))


def materialize_lazy_support_values(
    case: CaseData,
    *,
    n: int,
    idx_chunk: np.ndarray,
    masks: Sequence[int],
    cache: Dict[int, float],
    stats: UtilityStats,
    support_utility: np.ndarray,
    support_proxy: np.ndarray,
    empty_value: float,
    a_hat: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_coefs: np.ndarray,
) -> float:
    stats.total_evals += int(idx_chunk.size)
    proxy_seconds = 0.0
    proxy_missing_indices: List[int] = []
    proxy_missing_masks: List[int] = []
    for idx_raw in np.unique(idx_chunk):
        idx = int(idx_raw)
        mask = int(masks[idx])

        if np.isnan(support_utility[idx]):
            value = cache.get(mask)
            if value is None:
                t_util = time.perf_counter()
                value = float(case.oracle.utility_from_mask(mask))
                stats.utility_seconds += float(time.perf_counter() - t_util)
                stats.unique_evals += 1
                cache[mask] = float(value)
            support_utility[idx] = float(value)

        if np.isnan(support_proxy[idx]):
            proxy_missing_indices.append(idx)
            proxy_missing_masks.append(mask)

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
        proxy_seconds += float(time.perf_counter() - t_proxy)
    return float(proxy_seconds)


@njit(parallel=False)
def _residual_loop_numba(
    sampled_indices: np.ndarray,   # (total_samples,) int64 — rho support index
    utility_values: np.ndarray,    # (num_support,) float64 — cached utility per mask
    proxy_values: np.ndarray,      # (num_support,) float64 — proxy v(S) per mask
    rho_abs_counts: np.ndarray,    # (num_support,) float64
    rho_indptr: np.ndarray,        # (num_support+1,) int64
    rho_players: np.ndarray,       # (total_entries,) int64
    rho_signed_counts: np.ndarray, # (total_entries,) float64
    n: int,
) -> np.ndarray:
    residual_sum = np.zeros(n, dtype=np.float64)
    for k in range(sampled_indices.shape[0]):
        idx = sampled_indices[k]
        residual = utility_values[idx] - proxy_values[idx]
        scale = 2.0 * float(n) / max(rho_abs_counts[idx], 1.0)
        start = rho_indptr[idx]
        end = rho_indptr[idx + 1]
        for e in range(start, end):
            residual_sum[rho_players[e]] += scale * rho_signed_counts[e] * residual
    return residual_sum


def run_subset_method(
    case: CaseData,
    *,
    rho_support: Dict[str, object],
    pair_sampling_seconds: float,
    checkpoints: Sequence[int],
    seed: int,
    chunk_size: int,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    n = int(case.n)
    rng = np.random.RandomState(int(seed))
    rho_counts: Dict[int, Dict[int, int]] = rho_support["rho_counts"]
    abs_counts: Dict[int, int] = rho_support["abs_counts"]
    masks: List[int] = rho_support["masks"]
    probs: np.ndarray = rho_support["probs"]

    cache: Dict[int, float] = {0: 0.0}
    stats = UtilityStats()
    phi_sum = np.zeros(n, dtype=np.float64)
    sampled_done = 0
    phi_hats: List[np.ndarray] = []
    time_curve: List[float] = []
    utility_time_curve: List[float] = []
    total_eval_curve: List[int] = []
    unique_eval_curve: List[int] = []
    prev_checkpoint = 0

    for checkpoint in checkpoints:
        target_subset_calls = int(checkpoint) * n
        delta = int(target_subset_calls - prev_checkpoint)
        for idx_chunk in sample_q_indices(rng, probs, delta, int(chunk_size)):
            for idx in idx_chunk:
                mask = int(masks[int(idx)])
                value = get_cached_utility(case.oracle, cache, mask, stats)
                scale = gamma_scale(n, int(abs_counts[mask]))
                for player, signed_count in rho_counts[mask].items():
                    phi_sum[int(player)] += scale * float(signed_count) * float(value)
        sampled_done += delta
        prev_checkpoint = int(target_subset_calls)
        phi_hats.append(phi_sum / float(max(sampled_done, 1)))
        elapsed = float(
            pair_sampling_seconds
            + float(rho_support["rho_estimation_seconds"])
            + (time.perf_counter() - t0)
        )
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
            "method": "subset",
            "rho_permutations": int(rho_support["rho_permutations"]),
            "rho_support_size": int(rho_support["num_support"]),
            "rho_estimation_seconds": float(rho_support["rho_estimation_seconds"]),
            "pair_sampling_seconds": float(pair_sampling_seconds),
            "num_cached_subsets": int(len(cache)),
        }
    )
    return out


def collect_q_training_data(
    case: CaseData,
    *,
    rho_support: Dict[str, object],
    rng: np.random.RandomState,
    train_subset_budget: int,
    chunk_size: int,
) -> Tuple[Dict[int, float], Dict[int, float], UtilityStats, Dict[int, float]]:
    n = int(case.n)
    masks: List[int] = rho_support["masks"]
    probs: np.ndarray = rho_support["probs"]
    abs_counts: Dict[int, int] = rho_support["abs_counts"]
    square_counts: Dict[int, int] = rho_support["square_counts"]

    cache: Dict[int, float] = {0: 0.0}
    stats = UtilityStats()
    value_by_subset: Dict[int, float] = {}
    multiplicity: Dict[int, float] = {}
    for idx_chunk in sample_q_indices(rng, probs, int(train_subset_budget), int(chunk_size)):
        for idx in idx_chunk:
            mask = int(masks[int(idx)])
            value = get_cached_utility(case.oracle, cache, mask, stats)
            value_by_subset.setdefault(mask, float(value))
            multiplicity[mask] = float(multiplicity.get(mask, 0.0)) + 1.0

    weight_by_subset: Dict[int, float] = {}
    rho_perms = max(int(rho_support["rho_permutations"]), 1)
    for mask, count in multiplicity.items():
        rho_weight = float(square_counts[int(mask)]) * float(2 * n) / (
            float(rho_perms) * max(float(abs_counts[int(mask)]), 1.0)
        )
        weight_by_subset[int(mask)] = max(float(count) * max(rho_weight, 1e-12), 1e-12)
    return value_by_subset, weight_by_subset, stats, cache


def run_surrogate_subset_method(
    case: CaseData,
    *,
    rho_support: Dict[str, object],
    pair_samples: np.ndarray,
    pair_sampling_seconds: float,
    pair_list: Sequence[Tuple[int, int]],
    checkpoints: Sequence[int],
    seed: int,
    chunk_size: int,
    train_multiplier: float,
    interaction_percent: float,
    fit_label: str,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    n = int(case.n)
    rng = np.random.RandomState(int(seed))
    train_subset_budget = int(math.ceil(float(train_multiplier) * float(n) * float(n)))

    value_by_subset, weight_by_subset, stats, cache = collect_q_training_data(
        case,
        rho_support=rho_support,
        rng=rng,
        train_subset_budget=train_subset_budget,
        chunk_size=int(chunk_size),
    )

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

    masks: List[int] = rho_support["masks"]
    probs: np.ndarray = rho_support["probs"]
    rho_abs_counts: np.ndarray = rho_support["rho_abs_counts"]
    rho_indptr: np.ndarray = rho_support["rho_indptr"]
    rho_players: np.ndarray = rho_support["rho_players"]
    rho_signed_counts: np.ndarray = rho_support["rho_signed_counts"]
    num_support = int(len(masks))

    empty_value = float(case.oracle.empty_value)
    support_utility = np.full(num_support, np.nan, dtype=np.float64)
    support_proxy = np.full(num_support, np.nan, dtype=np.float64)
    pair_i = np.asarray([int(i) for i, _ in pair_list], dtype=np.int64)
    pair_j = np.asarray([int(j) for _, j in pair_list], dtype=np.int64)
    pair_coefs = np.asarray([float(b_hat[int(i), int(j)]) for i, j in pair_list], dtype=np.float64)
    proxy_eval_seconds = 0.0

    residual_sum = np.zeros(n, dtype=np.float64)
    residual_done = 0
    phi_hats: List[np.ndarray] = []
    time_curve: List[float] = []
    utility_time_curve: List[float] = []
    total_eval_curve: List[int] = []
    unique_eval_curve: List[int] = []
    prev_residual_budget = 0
    used_checkpoints: List[int] = []

    for checkpoint in checkpoints:
        target_raw_calls = int(checkpoint) * n
        if target_raw_calls < train_subset_budget:
            continue
        target_residual_budget = int(target_raw_calls - train_subset_budget)
        delta = int(target_residual_budget - prev_residual_budget)
        for idx_chunk in sample_q_indices(rng, probs, delta, int(chunk_size)):
            idx_chunk = idx_chunk.astype(np.int64, copy=False)
            proxy_eval_seconds += materialize_lazy_support_values(
                case,
                n=n,
                idx_chunk=idx_chunk,
                masks=masks,
                cache=cache,
                stats=stats,
                support_utility=support_utility,
                support_proxy=support_proxy,
                empty_value=empty_value,
                a_hat=a_hat,
                pair_i=pair_i,
                pair_j=pair_j,
                pair_coefs=pair_coefs,
            )
            residual_sum += _residual_loop_numba(
                idx_chunk,
                support_utility,
                support_proxy,
                rho_abs_counts,
                rho_indptr,
                rho_players,
                rho_signed_counts,
                n,
            )
        residual_done += delta
        prev_residual_budget = int(target_residual_budget)
        residual_hat = residual_sum / float(max(residual_done, 1)) if residual_done > 0 else np.zeros(n)
        phi_hats.append(proxy_phi + residual_hat)
        used_checkpoints.append(int(checkpoint))
        elapsed = float(
            float(rho_support["rho_estimation_seconds"])
            + pair_sampling_seconds
            + (time.perf_counter() - t0)
        )
        time_curve.append(elapsed)
        utility_time_curve.append(float(stats.utility_seconds))
        total_eval_curve.append(int(stats.total_evals))
        unique_eval_curve.append(int(stats.unique_evals))

    out = summarize_trial_curve(
        exact_target=case.exact_target,
        checkpoints=used_checkpoints,
        phi_hats=phi_hats,
        time_curve=time_curve,
        utility_time_curve=utility_time_curve,
        total_eval_curve=total_eval_curve,
        unique_eval_curve=unique_eval_curve,
    )
    out.update(
        {
            "method": "surrogate_subset",
            "train_multiplier": float(train_multiplier),
            "interaction_percent": float(interaction_percent),
            "num_proxy_pairs": int(len(pair_list)),
            "train_subset_budget": int(train_subset_budget),
            "pair_permutations": int(pair_samples.shape[0]),
            "rho_permutations": int(rho_support["rho_permutations"]),
            "rho_support_size": int(rho_support["num_support"]),
            "rho_estimation_seconds": float(rho_support["rho_estimation_seconds"]),
            "pair_sampling_seconds": float(pair_sampling_seconds),
            "prob_estimate_seconds": float(prob_seconds),
            "surrogate_fit_seconds": float(fit_seconds),
            "proxy_fit_rows": int(len(value_by_subset)),
            "proxy_fit_cols": int(n + len(pair_list)),
            "proxy_fit_active_cols": int(fit_diag.get("proxy_active_cols", 0)),
            "fit_diagnostics": fit_diag,
            "proxy_eval_seconds": float(proxy_eval_seconds),
            "num_cached_subsets": int(len(cache)),
        }
    )
    return out


def summarize_method_reps(rep_results: List[Dict[str, object]]) -> Dict[str, object]:
    if not rep_results:
        return {}
    keys = [
        "are_curve",
        "time_curve",
        "utility_time_curve",
        "non_utility_time_curve",
        "total_utility_eval_curve",
        "unique_utility_eval_curve",
        "reused_utility_eval_curve",
        "utility_evals_per_player_curve",
    ]
    summary: Dict[str, object] = {"checkpoints": rep_results[0].get("checkpoints", [])}
    for key in keys:
        curves = [rep.get(key, []) for rep in rep_results]
        min_len = min(len(curve) for curve in curves)
        if min_len == 0:
            summary[key] = {"mean": [], "std": []}
            continue
        arr = np.asarray([curve[:min_len] for curve in curves], dtype=np.float64)
        summary[key] = summarize_curve(arr)
    scalar_keys = [
        "final_are",
        "elapsed_seconds",
        "utility_eval_seconds",
        "non_utility_seconds",
        "total_utility_evals",
        "unique_utility_evals",
        "reused_utility_evals",
        "sampling_seconds",
        "pair_sampling_seconds",
        "prob_estimate_seconds",
        "surrogate_fit_seconds",
        "rho_estimation_seconds",
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


def print_method_summary(label: str, result: Dict[str, object]) -> None:
    are = float(result.get("final_are", float("nan")))
    unique_calls = int(result.get("unique_utility_evals", 0))
    non_utility_seconds = float(result.get("non_utility_seconds", float("nan")))
    total_calls = int(result.get("total_utility_evals", 0))
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
    for key, label_str in [
        ("sampling_seconds",      "sampling time      "),
        ("pair_sampling_seconds", "pair sampling time "),
        ("rho_estimation_seconds","rho estimation time"),
        ("prob_estimate_seconds", "prob estimate time "),
        ("surrogate_fit_seconds", "proxy fit time     "),
        ("proxy_eval_seconds",    "proxy eval time    "),
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
                "method": method_key(result),
                "are": float(result.get("final_are", float("nan"))),
                "unique": int(result.get("unique_utility_evals", 0)),
                "total": int(result.get("total_utility_evals", 0)),
                "others": float(result.get("non_utility_seconds", float("nan"))),
                "elapsed": float(result.get("elapsed_seconds", float("nan"))),
                "fit_mem_mb": tm_mb,
            }
        )

    method_width = max(16, max(len(row["method"]) for row in rows))
    print("\n      [comparison table for this rep]", flush=True)
    print(
        "      "
        + f"{'method':<{method_width}}  "
        + f"{'ARE':>10}  "
        + f"{'unique_U':>10}  "
        + f"{'total_U':>10}  "
        + f"{'runtime_others':>15}  "
        + f"{'total_runtime':>13}  "
        + f"{'fit_mem':>10}",
        flush=True,
    )
    print(
        "      "
        + "-" * method_width
        + "  " + "-" * 10
        + "  " + "-" * 10
        + "  " + "-" * 10
        + "  " + "-" * 15
        + "  " + "-" * 13
        + "  " + "-" * 10,
        flush=True,
    )
    for row in sorted(rows, key=lambda x: (not np.isfinite(x["are"]), x["are"])):
        are_str = f"{row['are']:.6f}" if np.isfinite(row["are"]) else "nan"
        others_str = f"{row['others']:.2f}s" if np.isfinite(row["others"]) else "nan"
        elapsed_str = f"{row['elapsed']:.2f}s" if np.isfinite(row["elapsed"]) else "nan"
        mem_str = f"{row['fit_mem_mb']:.1f}MB" if np.isfinite(row["fit_mem_mb"]) else "none"
        print(
            "      "
            + f"{row['method']:<{method_width}}  "
            + f"{are_str:>10}  "
            + f"{row['unique']:>10}  "
            + f"{row['total']:>10}  "
            + f"{others_str:>15}  "
            + f"{elapsed_str:>13}  "
            + f"{mem_str:>10}",
            flush=True,
        )


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


def run_case_reps(
    case: CaseData,
    *,
    args,
    methods: Sequence[str],
    init_mode: str,
    checkpoint_grid: Sequence[int],
    interaction_percents: Sequence[float],
    train_multipliers: Sequence[float],
    ) -> List[Dict[str, object]]:
    n = int(case.n)
    burnin_steps = resolve_burnin_steps(n, float(args.burnin_multiplier), float(args.burnin_exponent))
    num_pair_samples = resolve_num_pair_samples(args, n)
    max_checkpoint = int(max(checkpoint_grid))
    needs_value_samples = any(method in methods for method in ("perm", "surrogate_perm"))
    max_train_perms = (
        int(math.ceil(max(train_multipliers) * n))
        if ("surrogate_perm" in methods and train_multipliers)
        else 0
    )
    if not needs_value_samples:
        total_value_perms = 0
    elif "surrogate_perm" in methods:
        total_value_perms = int(max(max_checkpoint, max_train_perms))
    else:
        total_value_perms = int(max_checkpoint)

    results: List[Dict[str, object]] = []
    pair_lists: Dict[float, List[Tuple[int, int]]] = {}
    for pct in interaction_percents:
        pair_lists[float(pct)] = make_random_pair_list(n, float(pct), int(args.random_state) + 91_000 + n)

    print(
        f"  {case.case_label}, init={init_mode}, burnin={burnin_steps}, "
        f"value_perms={total_value_perms}, num_pair_samples={num_pair_samples}",
        flush=True,
    )

    for rep_idx in range(int(args.nreps)):
        rep_seed = int(args.random_state) + 1_000_000 * n + 10_000 * rep_idx + len(results)
        print(f"    rep {rep_idx + 1}/{args.nreps}, seed={rep_seed}", flush=True)

        if needs_value_samples:
            value_samples, acceptance_rate, sampling_seconds = draw_permutations(
                case,
                seed=rep_seed,
                num_samples=int(total_value_perms),
                burnin_steps=int(burnin_steps),
                steps_between=int(args.steps_between),
                init_mode=str(init_mode),
                report_every=int(args.print_every),
            )
        else:
            value_samples = np.empty((0, n), dtype=np.int64)
            acceptance_rate = float("nan")
            sampling_seconds = 0.0

        needs_pair_samples = any(
            method in methods for method in ("surrogate_perm", "subset", "surrogate_subset")
        )
        if needs_pair_samples:
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
        else:
            pair_samples = np.empty((0, n), dtype=np.int64)
            pair_acceptance_rate = float("nan")
            pair_sampling_seconds = 0.0

        rho_support = None
        if "subset" in methods or "surrogate_subset" in methods:
            print("      building rho support", flush=True)
            rho_support = build_rho_support(pair_samples, n)
            print(
                f"      rho support={rho_support['num_support']} "
                f"events={rho_support['num_events']} "
                f"time={rho_support['rho_estimation_seconds']:.2f}s",
                flush=True,
            )

        rep_methods: List[Dict[str, object]] = []
        if "perm" in methods:
            method_start = time.perf_counter()
            out = run_permutation_method(
                case,
                samples=value_samples[:max_checkpoint],
                sampling_seconds=float(sampling_seconds),
                checkpoints=checkpoint_grid,
            )
            out["method_seconds"] = float(time.perf_counter() - method_start)
            rep_methods.append(out)
            print_method_summary("method perm", out)

        if "surrogate_perm" in methods:
            for pct in interaction_percents:
                pair_list = pair_lists[float(pct)]
                for train_multiplier in train_multipliers:
                    train_perms = int(math.ceil(float(train_multiplier) * float(n)))
                    residual_needed = max(0, max_checkpoint - train_perms)
                    method_start = time.perf_counter()
                    out = run_surrogate_permutation_method(
                        case,
                        train_samples=value_samples[:train_perms],
                        residual_samples=value_samples[train_perms : train_perms + residual_needed],
                        pair_samples=pair_samples,
                        pair_list=pair_list,
                        sampling_seconds=float(sampling_seconds),
                        pair_sampling_seconds=float(pair_sampling_seconds),
                        checkpoints=checkpoint_grid,
                        train_permutation_budget=train_perms,
                        interaction_percent=float(pct),
                        train_multiplier=float(train_multiplier),
                        fit_label=f"surrogate_perm k={train_multiplier:g}, p={pct:g}",
                    )
                    out["method_seconds"] = float(time.perf_counter() - method_start)
                    rep_methods.append(out)
                    print_method_summary(
                        f"method surrogate_perm k={train_multiplier:g} ip={pct:g}",
                        out,
                    )

        if "subset" in methods and rho_support is not None:
            method_start = time.perf_counter()
            out = run_subset_method(
                case,
                rho_support=rho_support,
                pair_sampling_seconds=float(pair_sampling_seconds),
                checkpoints=checkpoint_grid,
                seed=rep_seed + 7_000_000,
                chunk_size=int(args.subset_chunk_size),
            )
            out["method_seconds"] = float(time.perf_counter() - method_start)
            rep_methods.append(out)
            print_method_summary("method subset", out)

        if "surrogate_subset" in methods and rho_support is not None:
            for pct in interaction_percents:
                pair_list = pair_lists[float(pct)]
                for train_multiplier in train_multipliers:
                    method_start = time.perf_counter()
                    out = run_surrogate_subset_method(
                        case,
                        rho_support=rho_support,
                        pair_samples=pair_samples,
                        pair_sampling_seconds=float(pair_sampling_seconds),
                        pair_list=pair_list,
                        checkpoints=checkpoint_grid,
                        seed=rep_seed + 8_000_000,
                        chunk_size=int(args.subset_chunk_size),
                        train_multiplier=float(train_multiplier),
                        interaction_percent=float(pct),
                        fit_label=f"surrogate_subset k={train_multiplier:g}, p={pct:g}",
                    )
                    out["method_seconds"] = float(time.perf_counter() - method_start)
                    rep_methods.append(out)
                    print_method_summary(
                        f"method surrogate_subset k={train_multiplier:g} ip={pct:g}",
                        out,
                    )

        print_rep_comparison_table(rep_methods)

        results.append(
            {
                "rep_idx": int(rep_idx),
                "seed": int(rep_seed),
                "acceptance_rate": float(acceptance_rate),
                "pair_acceptance_rate": float(pair_acceptance_rate),
                "sampling_seconds": float(sampling_seconds),
                "pair_sampling_seconds": float(pair_sampling_seconds),
                "methods": rep_methods,
            }
        )

    return results


def method_key(method_result: Dict[str, object]) -> str:
    method = str(method_result["method"])
    if method in {"surrogate_perm", "surrogate_subset"}:
        return (
            f"{method}"
            f"_k{float(method_result['train_multiplier']):g}"
            f"_ip{float(method_result['interaction_percent']):g}"
        )
    return method


def summarize_case_results(rep_results: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for rep in rep_results:
        for method_result in rep["methods"]:
            grouped.setdefault(method_key(method_result), []).append(method_result)
    return {key: summarize_method_reps(value) for key, value in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPASV surrogate-assisted accuracy experiments.")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "scenario1", "scenario2"])
    parser.add_argument("--n_grid", type=str, default="32,128,512,2048")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument("--nreps", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--burnin_multiplier", type=float, default=1.0)
    parser.add_argument("--burnin_exponent", type=float, default=2.5)
    parser.add_argument("--steps_between", type=int, default=1000)
    parser.add_argument("--init_mode", type=str, default="greedy", choices=["all", "random", "greedy", "topological"])
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--interaction_percents", type=str, default="0,0.1,0.25")
    parser.add_argument("--proxy_train_multipliers", type=str, default="1,2")
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
    methods = parse_methods(args.methods)
    init_modes = parse_init_modes(args.init_mode)
    interaction_percents = parse_float_list_arg(args.interaction_percents)
    train_multipliers = parse_float_list_arg(args.proxy_train_multipliers)
    checkpoint_grid = make_checkpoint_grid(int(args.num_samples), int(args.checkpoint_every))
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
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_grid": list(checkpoint_grid),
        "print_every": int(args.print_every),
        "nreps": int(args.nreps),
        "random_state": int(args.random_state),
        "burnin_multiplier": float(args.burnin_multiplier),
        "burnin_exponent": float(args.burnin_exponent),
        "steps_between": int(args.steps_between),
        "init_mode": str(args.init_mode),
        "methods": list(methods),
        "interaction_percents": list(interaction_percents),
        "proxy_train_multipliers": list(train_multipliers),
        "num_pair_samples": int(args.num_pair_samples) if args.num_pair_samples is not None else None,
        "subset_chunk_size": int(args.subset_chunk_size),
        "scenario1_lambda_cases": parse_list_arg(args.scenario1_lambda_cases),
        "scenario1_omega_values": parse_float_list_arg(args.scenario1_omega_values),
        "scenario2_block_size": int(args.scenario2_block_size),
        "scenario2_edge_prob": float(args.scenario2_edge_prob),
        "dag_seed": int(args.dag_seed),
        "scenario2_lambda_cases": parse_list_arg(args.scenario2_lambda_cases),
        "scenario2_omega_cases": parse_list_arg(args.scenario2_omega_cases),
        "scenario2_shared_graph_across_n": bool(args.scenario2_shared_graph_across_n),
    }

    print("Surrogate experiment configuration")
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
                    checkpoint_grid=checkpoint_grid,
                    interaction_percents=interaction_percents,
                    train_multipliers=train_multipliers,
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
