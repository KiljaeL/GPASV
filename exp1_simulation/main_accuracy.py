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
else:
    from .sampler import mh_sampler


DEFAULT_N_GRID = (64, 256, 1024, 4096)
DEFAULT_BLOCK_SIZE = 16
EPS_OMEGA = 1e-8
EPS_LAMBDA = 1e-12
GREEDY_CHAIN_SEED_OFFSET = 1_000_000_007
UINT32_MOD = 2**32


def _normalize_seed(seed: int) -> int:
    # NumPy RandomState requires seeds in [0, 2**32 - 1].
    return int(seed) % UINT32_MOD


def _chain_seed_for_init_mode(base_seed: int, init_mode: str) -> int:
    mode = normalize_init_mode(init_mode)
    offset = GREEDY_CHAIN_SEED_OFFSET if mode == "greedy" else 0
    return _normalize_seed(int(base_seed) + int(offset))


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
                            mem_kib = float(parts[1])
                            return mem_kib / (1024.0 ** 2)
        except OSError:
            pass
    return float("nan")


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


def _save_pickle(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


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


def make_accuracy_output_path(
    save_dir: Path,
    scenario: str,
    *,
    n: Optional[int] = None,
    init_mode_tag: Optional[str] = None,
) -> Path:
    stem = f"accuracy_{str(scenario)}"
    if init_mode_tag is not None:
        stem += f"_{normalize_init_mode(init_mode_tag)}"
    if n is not None:
        stem += f"_n{int(n)}"
    return Path(save_dir) / f"{stem}.pkl"


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

    rng = np.random.RandomState(_normalize_seed(int(seed)))
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
            probs = choice_weights / weight_sum
            chosen = int(rng.choice(active, p=probs))

        ordering[pos] = chosen
        remaining[chosen] = False
        for pred, weight in incoming[chosen]:
            if remaining[int(pred)]:
                scores[int(pred)] -= float(weight)

    return ordering


def subset_to_mask_from_perm_prefix(mask: int, player: int) -> int:
    return int(mask) | (1 << int(player))


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
    rng = np.random.RandomState(_normalize_seed(int(seed)))
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


def segment_value(table: np.ndarray, left: int, right: int) -> float:
    if int(left) > int(right):
        return 0.0
    return float(table[int(left), int(right)])


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
            denom = max(denom, EPS_LAMBDA)
            scale = coeff / denom
            for i in range(a, b + 1):
                out[i] += scale * float(lambdas[i]) * (float(omega_bar) ** (b - i))
    return out


def make_lambda_block_cycle(case: str, block_members: Sequence[np.ndarray], seed: int) -> np.ndarray:
    n = sum(len(m) for m in block_members)
    if case == "ones":
        return np.ones(int(n), dtype=np.float64)
    if case == "block_uniform_1_10":
        rng = np.random.RandomState(_normalize_seed(int(seed)))
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
        rng = np.random.RandomState(_normalize_seed(int(seed)))
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
    rng = np.random.RandomState(_normalize_seed(int(omega_seed)))
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


def build_total_order_graph(n: int, omega_bar: float) -> List[Tuple[int, int, float]]:
    edges = []
    omega_bar = float(max(omega_bar, EPS_OMEGA))
    for i in range(int(n)):
        for j in range(i + 1, int(n)):
            edges.append((int(i), int(j), omega_bar))
    return edges


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


def draw_gpasv_samples_base(
    *,
    n: int,
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    seed: int,
    num_samples: int,
    burnin_steps: int,
    steps_between: int,
    report_every: int,
    chain_seed: Optional[int] = None,
    init_mode: str = "random",
) -> Tuple[np.ndarray, float, float]:
    raw_lambda, raw_edges = base_params_to_raw(lambdas, edges_with_omega)
    init_mode = normalize_init_mode(init_mode)
    init_seed = _normalize_seed(int(seed))
    sampler_seed = _normalize_seed(int(init_seed) if chain_seed is None else int(chain_seed))
    if str(init_mode) == "greedy":
        init_pi = sample_greedy_order(
            int(n),
            np.asarray(lambdas, dtype=np.float64),
            edges_with_omega,
            int(init_seed),
        )
    else:
        init_rng = np.random.RandomState(int(init_seed))
        init_pi = init_rng.permutation(int(n)).astype(np.int64)
    t0 = time.perf_counter()
    samples, acceptance_rate = mh_sampler(
        n=int(n),
        edges=raw_edges,
        lam=raw_lambda.tolist(),
        alpha=1.0,
        beta=1.0,
        seed=int(sampler_seed),
        n_samples=int(num_samples),
        burn_in=int(burnin_steps),
        thin=int(steps_between),
        init_pi=init_pi,
        report_every=int(report_every),
    )
    elapsed = float(time.perf_counter() - t0)
    return samples, float(acceptance_rate), elapsed


def run_block_cycle_trial(
    *,
    n: int,
    block_members: Sequence[np.ndarray],
    block_of: np.ndarray,
    block_coeffs: np.ndarray,
    exact_target: np.ndarray,
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    num_samples: int,
    checkpoint_every: int,
    print_every: int,
    seed: int,
    burnin_steps: int,
    steps_between: int,
    chain_seed: Optional[int] = None,
    init_mode: str = "random",
) -> Dict[str, object]:
    exact_norm = max(float(np.linalg.norm(exact_target)), EPS_LAMBDA)
    init_seed = _normalize_seed(int(seed))
    sampler_seed = _normalize_seed(int(init_seed) if chain_seed is None else int(chain_seed))
    samples, acceptance_rate, sampling_elapsed = draw_gpasv_samples_base(
        n=int(n),
        lambdas=lambdas,
        edges_with_omega=edges_with_omega,
        seed=int(init_seed),
        num_samples=int(num_samples),
        burnin_steps=int(burnin_steps),
        steps_between=int(steps_between),
        report_every=int(print_every),
        chain_seed=int(sampler_seed),
        init_mode=str(init_mode),
    )
    phi_sum = np.zeros(int(n), dtype=np.float64)
    checkpoints: List[int] = []
    are_curve: List[float] = []
    time_curve: List[float] = []
    utility_time_curve: List[float] = []
    non_utility_time_curve: List[float] = []
    total_utility_eval_curve: List[int] = []
    unique_utility_eval_curve: List[int] = []
    reused_utility_eval_curve: List[int] = []

    utility_elapsed = 0.0
    total_utility_evals = 0
    unique_utility_evals = 0
    utility_cache: Dict[int, float] = {0: 0.0}
    block_size = int(len(block_members[0]))

    t_process = time.perf_counter()
    for sample_idx, perm in enumerate(samples, start=1):
        counts = np.zeros(len(block_members), dtype=np.int64)
        completed_mask = 0
        subset_mask = 0
        prev_u = 0.0
        for player in perm:
            player = int(player)
            subset_mask = subset_to_mask_from_perm_prefix(subset_mask, player)
            block_id = int(block_of[player])
            counts[block_id] += 1
            if counts[block_id] == block_size:
                completed_mask |= 1 << block_id
            total_utility_evals += 1
            current_u = utility_cache.get(subset_mask)
            if current_u is None:
                t_util = time.perf_counter()
                current_u = prev_u if counts[block_id] < block_size else prev_u + float(block_coeffs[block_id])
                utility_elapsed += float(time.perf_counter() - t_util)
                utility_cache[subset_mask] = float(current_u)
                unique_utility_evals += 1
            phi_sum[player] += float(current_u - prev_u)
            prev_u = float(current_u)

        if sample_idx % int(checkpoint_every) == 0:
            phi_hat = phi_sum / float(sample_idx)
            are = float(np.linalg.norm(phi_hat - exact_target) / exact_norm)
            elapsed = float(sampling_elapsed + (time.perf_counter() - t_process))
            checkpoints.append(int(sample_idx))
            are_curve.append(are)
            time_curve.append(elapsed)
            utility_time_curve.append(float(utility_elapsed))
            non_utility_time_curve.append(float(elapsed - utility_elapsed))
            total_utility_eval_curve.append(int(total_utility_evals))
            unique_utility_eval_curve.append(int(unique_utility_evals))
            reused_utility_eval_curve.append(int(total_utility_evals - unique_utility_evals))
            if sample_idx % int(print_every) == 0 or sample_idx == int(num_samples):
                print(
                    f"    [ckpt {sample_idx}/{num_samples}] "
                    f"ARE={are:.6f} elapsed={elapsed:.2f}s "
                    f"utility={utility_elapsed:.2f}s "
                    f"unique_eval={unique_utility_evals}",
                    flush=True,
                )

    elapsed_total = float(sampling_elapsed + (time.perf_counter() - t_process))
    phi_hat = phi_sum / float(len(samples))
    final_are = float(are_curve[-1] if are_curve else np.linalg.norm(phi_hat - exact_target) / exact_norm)
    aucc = float(np.mean(are_curve) if are_curve else final_are)
    return {
        "phi_hat": phi_hat,
        "checkpoints": checkpoints,
        "are_curve": are_curve,
        "time_curve": time_curve,
        "utility_time_curve": utility_time_curve,
        "non_utility_time_curve": non_utility_time_curve,
        "total_utility_eval_curve": total_utility_eval_curve,
        "unique_utility_eval_curve": unique_utility_eval_curve,
        "reused_utility_eval_curve": reused_utility_eval_curve,
        "final_are": final_are,
        "aucc": aucc,
        "elapsed_seconds": elapsed_total,
        "utility_eval_seconds": float(utility_elapsed),
        "non_utility_seconds": float(elapsed_total - utility_elapsed),
        "total_utility_evals": int(total_utility_evals),
        "unique_utility_evals": int(unique_utility_evals),
        "reused_utility_evals": int(total_utility_evals - unique_utility_evals),
        "acceptance_rate": float(acceptance_rate),
        "num_cached_subsets": int(len(utility_cache)),
        "num_samples": int(len(samples)),
        "init_seed": int(init_seed),
        "chain_seed": int(sampler_seed),
    }


def run_total_order_trial(
    *,
    n: int,
    interval_weights: np.ndarray,
    segment_table: np.ndarray,
    exact_target: np.ndarray,
    lambdas: np.ndarray,
    omega_bar: float,
    num_samples: int,
    checkpoint_every: int,
    print_every: int,
    seed: int,
    burnin_steps: int,
    steps_between: int,
    chain_seed: Optional[int] = None,
    init_mode: str = "random",
) -> Dict[str, object]:
    exact_norm = max(float(np.linalg.norm(exact_target)), EPS_LAMBDA)
    edges_with_omega = build_total_order_graph(int(n), float(omega_bar))
    init_seed = _normalize_seed(int(seed))
    sampler_seed = _normalize_seed(int(init_seed) if chain_seed is None else int(chain_seed))
    samples, acceptance_rate, sampling_elapsed = draw_gpasv_samples_base(
        n=int(n),
        lambdas=lambdas,
        edges_with_omega=edges_with_omega,
        seed=int(init_seed),
        num_samples=int(num_samples),
        burnin_steps=int(burnin_steps),
        steps_between=int(steps_between),
        report_every=int(print_every),
        chain_seed=int(sampler_seed),
        init_mode=str(init_mode),
    )
    phi_sum = np.zeros(int(n), dtype=np.float64)
    checkpoints: List[int] = []
    are_curve: List[float] = []
    time_curve: List[float] = []
    utility_time_curve: List[float] = []
    non_utility_time_curve: List[float] = []
    total_utility_eval_curve: List[int] = []
    unique_utility_eval_curve: List[int] = []
    reused_utility_eval_curve: List[int] = []

    utility_elapsed = 0.0
    total_utility_evals = 0
    unique_utility_evals = 0
    utility_cache: Dict[int, float] = {0: 0.0}

    t_process = time.perf_counter()
    for sample_idx, perm in enumerate(samples, start=1):
        seen = np.zeros(int(n), dtype=np.bool_)
        run_left = np.arange(int(n), dtype=np.int64)
        run_right = np.arange(int(n), dtype=np.int64)
        subset_mask = 0
        prev_u = 0.0

        for player in perm:
            player = int(player)
            left = int(player)
            right = int(player)
            if player > 0 and seen[player - 1]:
                left = int(run_left[player - 1])
            if player + 1 < int(n) and seen[player + 1]:
                right = int(run_right[player + 1])

            subset_mask = subset_to_mask_from_perm_prefix(subset_mask, player)
            total_utility_evals += 1
            current_u = utility_cache.get(subset_mask)
            if current_u is None:
                t_util = time.perf_counter()
                marginal = segment_value(segment_table, left, right)
                marginal -= segment_value(segment_table, left, player - 1)
                marginal -= segment_value(segment_table, player + 1, right)
                current_u = prev_u + marginal
                utility_elapsed += float(time.perf_counter() - t_util)
                utility_cache[subset_mask] = float(current_u)
                unique_utility_evals += 1

            seen[player] = True
            run_left[player] = left
            run_right[player] = right
            run_right[left] = right
            run_left[right] = left

            phi_sum[player] += float(current_u - prev_u)
            prev_u = float(current_u)

        if sample_idx % int(checkpoint_every) == 0:
            phi_hat = phi_sum / float(sample_idx)
            are = float(np.linalg.norm(phi_hat - exact_target) / exact_norm)
            elapsed = float(sampling_elapsed + (time.perf_counter() - t_process))
            checkpoints.append(int(sample_idx))
            are_curve.append(are)
            time_curve.append(elapsed)
            utility_time_curve.append(float(utility_elapsed))
            non_utility_time_curve.append(float(elapsed - utility_elapsed))
            total_utility_eval_curve.append(int(total_utility_evals))
            unique_utility_eval_curve.append(int(unique_utility_evals))
            reused_utility_eval_curve.append(int(total_utility_evals - unique_utility_evals))
            if sample_idx % int(print_every) == 0 or sample_idx == int(num_samples):
                print(
                    f"    [ckpt {sample_idx}/{num_samples}] "
                    f"ARE={are:.6f} elapsed={elapsed:.2f}s "
                    f"utility={utility_elapsed:.2f}s "
                    f"unique_eval={unique_utility_evals}",
                    flush=True,
                )

    elapsed_total = float(sampling_elapsed + (time.perf_counter() - t_process))
    phi_hat = phi_sum / float(len(samples))
    final_are = float(are_curve[-1] if are_curve else np.linalg.norm(phi_hat - exact_target) / exact_norm)
    aucc = float(np.mean(are_curve) if are_curve else final_are)
    return {
        "phi_hat": phi_hat,
        "checkpoints": checkpoints,
        "are_curve": are_curve,
        "time_curve": time_curve,
        "utility_time_curve": utility_time_curve,
        "non_utility_time_curve": non_utility_time_curve,
        "total_utility_eval_curve": total_utility_eval_curve,
        "unique_utility_eval_curve": unique_utility_eval_curve,
        "reused_utility_eval_curve": reused_utility_eval_curve,
        "final_are": final_are,
        "aucc": aucc,
        "elapsed_seconds": elapsed_total,
        "utility_eval_seconds": float(utility_elapsed),
        "non_utility_seconds": float(elapsed_total - utility_elapsed),
        "total_utility_evals": int(total_utility_evals),
        "unique_utility_evals": int(unique_utility_evals),
        "reused_utility_evals": int(total_utility_evals - unique_utility_evals),
        "acceptance_rate": float(acceptance_rate),
        "num_cached_subsets": int(len(utility_cache)),
        "num_samples": int(len(samples)),
        "init_seed": int(init_seed),
        "chain_seed": int(sampler_seed),
    }


def summarize_rep_results(rep_results: List[Dict[str, object]]) -> Dict[str, object]:
    are_curves = np.asarray([rep["are_curve"] for rep in rep_results], dtype=np.float64)
    time_curves = np.asarray([rep["time_curve"] for rep in rep_results], dtype=np.float64)
    utility_time_curves = np.asarray([rep["utility_time_curve"] for rep in rep_results], dtype=np.float64)
    non_utility_time_curves = np.asarray([rep["non_utility_time_curve"] for rep in rep_results], dtype=np.float64)
    total_utility_eval_curves = np.asarray([rep["total_utility_eval_curve"] for rep in rep_results], dtype=np.float64)
    unique_utility_eval_curves = np.asarray([rep["unique_utility_eval_curve"] for rep in rep_results], dtype=np.float64)
    reused_utility_eval_curves = np.asarray([rep["reused_utility_eval_curve"] for rep in rep_results], dtype=np.float64)
    return {
        "checkpoints": rep_results[0]["checkpoints"] if rep_results else [],
        "are_curve": summarize_curve(are_curves),
        "time_curve": summarize_curve(time_curves),
        "utility_time_curve": summarize_curve(utility_time_curves),
        "non_utility_time_curve": summarize_curve(non_utility_time_curves),
        "total_utility_eval_curve": summarize_curve(total_utility_eval_curves),
        "unique_utility_eval_curve": summarize_curve(unique_utility_eval_curves),
        "reused_utility_eval_curve": summarize_curve(reused_utility_eval_curves),
        "final_are": summarize_scalar([float(rep["final_are"]) for rep in rep_results]),
        "aucc": summarize_scalar([float(rep["aucc"]) for rep in rep_results]),
        "elapsed_seconds": summarize_scalar([float(rep["elapsed_seconds"]) for rep in rep_results]),
        "utility_eval_seconds": summarize_scalar([float(rep["utility_eval_seconds"]) for rep in rep_results]),
        "non_utility_seconds": summarize_scalar([float(rep["non_utility_seconds"]) for rep in rep_results]),
        "total_utility_evals": summarize_scalar([float(rep["total_utility_evals"]) for rep in rep_results]),
        "unique_utility_evals": summarize_scalar([float(rep["unique_utility_evals"]) for rep in rep_results]),
        "reused_utility_evals": summarize_scalar([float(rep["reused_utility_evals"]) for rep in rep_results]),
        "acceptance_rate": summarize_scalar([float(rep["acceptance_rate"]) for rep in rep_results]),
        "num_cached_subsets": summarize_scalar([float(rep["num_cached_subsets"]) for rep in rep_results]),
    }


def run_block_cycle_scenario(
    *,
    n_grid: Sequence[int],
    block_size: int,
    edge_prob: float,
    dag_seed: int,
    lambda_cases: Sequence[str],
    omega_cases: Sequence[str],
    num_samples: int,
    checkpoint_every: int,
    print_every: int,
    nreps: int,
    random_state: int,
    burnin_multiplier: float,
    burnin_exponent: float,
    steps_between: int,
    save_dir: Path,
    payload_config: Dict[str, object],
    init_mode: str,
    save_init_mode_tag: Optional[str] = None,
) -> Dict[str, object]:
    results = []
    for n in n_grid:
        n = int(n)
        burnin_steps = resolve_burnin_steps(n, float(burnin_multiplier), float(burnin_exponent))
        print(f"[Scenario 2] n={n}, burnin_steps={burnin_steps}", flush=True)
        for lambda_case in lambda_cases:
            for omega_case in omega_cases:
                case_seed = int(random_state) + 100_000 * n + 1_000 * len(results)
                graph = build_block_cycle_graph(
                    n=n,
                    block_size=int(block_size),
                    edge_prob=float(edge_prob),
                    dag_seed=_normalize_seed(int(dag_seed) + n),
                    omega_case=str(omega_case),
                    omega_seed=_normalize_seed(case_seed + 11),
                )
                coeff_rng = np.random.RandomState(_normalize_seed(case_seed + 17))
                block_coeffs = coeff_rng.uniform(0.5, 1.5, size=graph["num_blocks"]).astype(np.float64)
                lambdas = make_lambda_block_cycle(str(lambda_case), graph["block_members"], _normalize_seed(case_seed + 23))
                exact_target = exact_target_block_cycle(block_coeffs, graph["block_members"])

                rep_results = []
                for rep_idx in range(int(nreps)):
                    rep_seed = _normalize_seed(case_seed + 100 + rep_idx)
                    chain_seed = _chain_seed_for_init_mode(rep_seed, str(init_mode))
                    print(
                        f"  case lambda={lambda_case}, omega={omega_case}, rep={rep_idx + 1}/{nreps}, "
                        f"init_seed={rep_seed}, chain_seed={chain_seed}",
                        flush=True,
                    )
                    rep_result = run_block_cycle_trial(
                        n=n,
                        block_members=graph["block_members"],
                        block_of=graph["block_of"],
                        block_coeffs=block_coeffs,
                        exact_target=exact_target,
                        lambdas=lambdas,
                        edges_with_omega=graph["edges"],
                        num_samples=int(num_samples),
                        checkpoint_every=int(checkpoint_every),
                        print_every=int(print_every),
                        seed=int(rep_seed),
                        burnin_steps=int(burnin_steps),
                        steps_between=int(steps_between),
                        chain_seed=int(chain_seed),
                        init_mode=str(init_mode),
                    )
                    rep_results.append(rep_result)

                result = {
                    "n": n,
                    "lambda_case": str(lambda_case),
                    "omega_case": str(omega_case),
                    "num_blocks": int(graph["num_blocks"]),
                    "block_size": int(block_size),
                    "block_edges": list(graph["block_edges"]),
                    "block_edge_weights": {k: float(v) for k, v in graph["block_edge_weights"].items()},
                    "internal_cycle_edges": list(graph["internal_cycle_edges"]),
                    "internal_cycle_weights": {int(k): float(v) for k, v in graph["internal_cycle_weights"].items()},
                    "block_coeffs": block_coeffs.copy(),
                    "exact_target": exact_target.copy(),
                    "result": {
                        "rep_results": rep_results,
                        "summary": summarize_rep_results(rep_results),
                    },
                }
                results.append(result)

        partial_payload = {"config": dict(payload_config), "scenario2": {"results": results}}
        partial_path = make_accuracy_output_path(
            Path(save_dir),
            "scenario2",
            n=n,
            init_mode_tag=save_init_mode_tag,
        )
        _save_pickle(partial_path, partial_payload)
        print(f"Saved accuracy Scenario 2 partial results to {partial_path}", flush=True)

    return {
        "results": results,
        "lambda_cases": list(lambda_cases),
        "omega_cases": list(omega_cases),
        "block_size": int(block_size),
        "edge_prob": float(edge_prob),
        "burnin_multiplier": float(burnin_multiplier),
        "burnin_exponent": float(burnin_exponent),
    }


def run_total_order_scenario(
    *,
    n_grid: Sequence[int],
    lambda_cases: Sequence[str],
    omega_values: Sequence[float],
    num_samples: int,
    checkpoint_every: int,
    print_every: int,
    nreps: int,
    random_state: int,
    burnin_multiplier: float,
    burnin_exponent: float,
    steps_between: int,
    save_dir: Path,
    payload_config: Dict[str, object],
    init_mode: str,
    save_init_mode_tag: Optional[str] = None,
) -> Dict[str, object]:
    results = []
    for n in n_grid:
        n = int(n)
        burnin_steps = resolve_burnin_steps(n, float(burnin_multiplier), float(burnin_exponent))
        print(f"[Scenario 1] n={n}, burnin_steps={burnin_steps}", flush=True)
        for lambda_case in lambda_cases:
            for omega_bar in omega_values:
                case_seed = int(random_state) + 200_000 * n + 1_000 * len(results)
                lambdas = make_lambda_total_order(str(lambda_case), n, _normalize_seed(case_seed + 13))
                interval_rng = np.random.RandomState(_normalize_seed(case_seed + 29))
                starts, ends, coeffs = sample_uniform_intervals(n, n * n, interval_rng)
                interval_weights = build_interval_weight_matrix(n, starts, ends, coeffs)
                segment_table = build_segment_weight_table(interval_weights)
                exact_target = exact_target_total_order(interval_weights, lambdas, float(omega_bar))

                rep_results = []
                for rep_idx in range(int(nreps)):
                    rep_seed = _normalize_seed(case_seed + 100 + rep_idx)
                    chain_seed = _chain_seed_for_init_mode(rep_seed, str(init_mode))
                    print(
                        f"  case lambda={lambda_case}, omega_bar={omega_bar}, rep={rep_idx + 1}/{nreps}, "
                        f"init_seed={rep_seed}, chain_seed={chain_seed}",
                        flush=True,
                    )
                    rep_result = run_total_order_trial(
                        n=n,
                        interval_weights=interval_weights,
                        segment_table=segment_table,
                        exact_target=exact_target,
                        lambdas=lambdas,
                        omega_bar=float(omega_bar),
                        num_samples=int(num_samples),
                        checkpoint_every=int(checkpoint_every),
                        print_every=int(print_every),
                        seed=int(rep_seed),
                        burnin_steps=int(burnin_steps),
                        steps_between=int(steps_between),
                        chain_seed=int(chain_seed),
                        init_mode=str(init_mode),
                    )
                    rep_results.append(rep_result)

                result = {
                    "n": n,
                    "lambda_case": str(lambda_case),
                    "omega_bar": float(omega_bar),
                    "num_terms": int(n * n),
                    "interval_weights": interval_weights.copy(),
                    "exact_target": exact_target.copy(),
                    "result": {
                        "rep_results": rep_results,
                        "summary": summarize_rep_results(rep_results),
                    },
                }
                results.append(result)

        partial_payload = {"config": dict(payload_config), "scenario1": {"results": results}}
        partial_path = make_accuracy_output_path(
            Path(save_dir),
            "scenario1",
            n=n,
            init_mode_tag=save_init_mode_tag,
        )
        _save_pickle(partial_path, partial_payload)
        print(f"Saved accuracy Scenario 1 partial results to {partial_path}", flush=True)

    return {
        "results": results,
        "lambda_cases": list(lambda_cases),
        "omega_values": [float(x) for x in omega_values],
        "burnin_multiplier": float(burnin_multiplier),
        "burnin_exponent": float(burnin_exponent),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "scenario1", "scenario2"])
    parser.add_argument("--n_grid", type=str, default="32,64,128,256")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--nreps", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--burnin_multiplier", type=float, default=1.0)
    parser.add_argument("--burnin_exponent", type=float, default=2.0)
    parser.add_argument("--steps_between", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--init_mode", type=str, default="random", choices=["all", "random", "greedy", "topological"])
    parser.add_argument("--scenario1_lambda_cases", type=str, default="ones,uniform_1_10")
    parser.add_argument("--scenario1_omega_values", type=str, default="0.7,0.3")
    parser.add_argument("--scenario2_block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--scenario2_edge_prob", type=float, default=0.8)
    parser.add_argument("--dag_seed", type=int, default=42)
    parser.add_argument("--scenario2_lambda_cases", type=str, default="ones,block_uniform_1_10")
    parser.add_argument("--scenario2_omega_cases", type=str, default="uniform_0.5_1,uniform_0_0.5")
    args = parser.parse_args()

    n_grid = parse_n_grid(args.n_grid)
    init_modes = parse_init_modes(args.init_mode)
    scenario1_lambda_cases = parse_list_arg(args.scenario1_lambda_cases)
    scenario1_omega_values = parse_float_list_arg(args.scenario1_omega_values)
    scenario2_lambda_cases = parse_list_arg(args.scenario2_lambda_cases)
    scenario2_omega_cases = parse_list_arg(args.scenario2_omega_cases)
    if args.save_dir is None:
        save_dir = Path(__file__).resolve().parent / "save" / "accuracy"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print_system_info()

    config = {
        "scenario": str(args.scenario),
        "n_grid": list(n_grid),
        "num_samples": int(args.num_samples),
        "checkpoint_every": int(args.checkpoint_every),
        "print_every": int(args.print_every),
        "nreps": int(args.nreps),
        "random_state": int(args.random_state),
        "burnin_multiplier": float(args.burnin_multiplier),
        "burnin_exponent": float(args.burnin_exponent),
        "steps_between": int(args.steps_between),
        "init_mode": "all" if len(init_modes) > 1 else str(init_modes[0]),
        "scenario1_lambda_cases": list(scenario1_lambda_cases),
        "scenario1_omega_values": list(scenario1_omega_values),
        "scenario2_block_size": int(args.scenario2_block_size),
        "scenario2_edge_prob": float(args.scenario2_edge_prob),
        "dag_seed": int(args.dag_seed),
        "scenario2_lambda_cases": list(scenario2_lambda_cases),
        "scenario2_omega_cases": list(scenario2_omega_cases),
        "chain_seed_policy": (
            "init_seed is shared across init modes; random chain_seed=init_seed; "
            f"greedy chain_seed=init_seed+{GREEDY_CHAIN_SEED_OFFSET} mod 2^32"
        ),
    }
    if len(init_modes) > 1:
        config["init_modes"] = list(init_modes)

    payload = {"config": config}

    print("Accuracy experiment configuration")
    for key, value in payload["config"].items():
        print(f"  {key}: {value}")
    print()

    if len(init_modes) == 1:
        active_init_mode = str(init_modes[0])
        if args.scenario in {"all", "scenario1"}:
            payload["scenario1"] = run_total_order_scenario(
                n_grid=n_grid,
                lambda_cases=scenario1_lambda_cases,
                omega_values=scenario1_omega_values,
                num_samples=int(args.num_samples),
                checkpoint_every=int(args.checkpoint_every),
                print_every=int(args.print_every),
                nreps=int(args.nreps),
                random_state=int(args.random_state),
                burnin_multiplier=float(args.burnin_multiplier),
                burnin_exponent=float(args.burnin_exponent),
                steps_between=int(args.steps_between),
                save_dir=save_dir,
                payload_config=payload["config"],
                init_mode=active_init_mode,
                save_init_mode_tag=None,
            )

        if args.scenario in {"all", "scenario2"}:
            payload["scenario2"] = run_block_cycle_scenario(
                n_grid=n_grid,
                block_size=int(args.scenario2_block_size),
                edge_prob=float(args.scenario2_edge_prob),
                dag_seed=int(args.dag_seed),
                lambda_cases=scenario2_lambda_cases,
                omega_cases=scenario2_omega_cases,
                num_samples=int(args.num_samples),
                checkpoint_every=int(args.checkpoint_every),
                print_every=int(args.print_every),
                nreps=int(args.nreps),
                random_state=int(args.random_state),
                burnin_multiplier=float(args.burnin_multiplier),
                burnin_exponent=float(args.burnin_exponent),
                steps_between=int(args.steps_between),
                save_dir=save_dir,
                payload_config=payload["config"],
                init_mode=active_init_mode,
                save_init_mode_tag=None,
            )

        out_path = make_accuracy_output_path(save_dir, str(args.scenario))
        _save_pickle(out_path, payload)
        print()
        print(f"Saved results to {out_path}")
        return

    payload["runs"] = {}
    for active_init_mode in init_modes:
        print()
        print(f"Running accuracy experiments with init_mode={active_init_mode}")

        mode_config = dict(payload["config"])
        mode_config["init_mode"] = str(active_init_mode)
        mode_config.pop("init_modes", None)
        mode_payload = {"config": mode_config}

        if args.scenario in {"all", "scenario1"}:
            mode_payload["scenario1"] = run_total_order_scenario(
                n_grid=n_grid,
                lambda_cases=scenario1_lambda_cases,
                omega_values=scenario1_omega_values,
                num_samples=int(args.num_samples),
                checkpoint_every=int(args.checkpoint_every),
                print_every=int(args.print_every),
                nreps=int(args.nreps),
                random_state=int(args.random_state),
                burnin_multiplier=float(args.burnin_multiplier),
                burnin_exponent=float(args.burnin_exponent),
                steps_between=int(args.steps_between),
                save_dir=save_dir,
                payload_config=mode_config,
                init_mode=str(active_init_mode),
                save_init_mode_tag=str(active_init_mode),
            )

        if args.scenario in {"all", "scenario2"}:
            mode_payload["scenario2"] = run_block_cycle_scenario(
                n_grid=n_grid,
                block_size=int(args.scenario2_block_size),
                edge_prob=float(args.scenario2_edge_prob),
                dag_seed=int(args.dag_seed),
                lambda_cases=scenario2_lambda_cases,
                omega_cases=scenario2_omega_cases,
                num_samples=int(args.num_samples),
                checkpoint_every=int(args.checkpoint_every),
                print_every=int(args.print_every),
                nreps=int(args.nreps),
                random_state=int(args.random_state),
                burnin_multiplier=float(args.burnin_multiplier),
                burnin_exponent=float(args.burnin_exponent),
                steps_between=int(args.steps_between),
                save_dir=save_dir,
                payload_config=mode_config,
                init_mode=str(active_init_mode),
                save_init_mode_tag=str(active_init_mode),
            )

        mode_out_path = make_accuracy_output_path(
            save_dir,
            str(args.scenario),
            init_mode_tag=str(active_init_mode),
        )
        _save_pickle(mode_out_path, mode_payload)
        payload["runs"][str(active_init_mode)] = {"path": str(mode_out_path)}
        print(f"Saved {active_init_mode} results to {mode_out_path}")

    out_path = make_accuracy_output_path(save_dir, str(args.scenario))
    _save_pickle(out_path, payload)
    print()
    print(f"Saved run manifest to {out_path}")


if __name__ == "__main__":
    main()
