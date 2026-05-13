import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS_OMEGA = 1e-8
EPS_LAMBDA = 1e-12


def parse_int_grid(grid_str: str) -> List[int]:
    return [int(x.strip()) for x in str(grid_str).split(",") if x.strip()]


def parse_float_grid(grid_str: str) -> List[float]:
    return [float(x.strip()) for x in str(grid_str).split(",") if x.strip()]


def parse_interval_grid(interval_str: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for token in str(interval_str).split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid interval token: {token!r}")
        lo = float(parts[0])
        hi = float(parts[1])
        if lo > hi:
            raise ValueError(f"Invalid interval with lo > hi: ({lo}, {hi})")
        out.append((lo, hi))
    return out


def graph_case_grid() -> List[Dict[str, Optional[float]]]:
    return [
        {"family": "total_order", "p": None},
        {"family": "random_dag", "p": 0.4},
        {"family": "random_dag", "p": 0.8},
        {"family": "random_digraph", "p": 0.2},
        {"family": "random_digraph", "p": 0.4},
    ]


def build_total_order_edges(n: int) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for i in range(int(n)):
        for j in range(i + 1, int(n)):
            edges.append((int(i), int(j)))
    return edges


def build_random_dag_edges(n: int, p: float, rng: np.random.RandomState) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    p = float(p)
    for i in range(int(n)):
        for j in range(i + 1, int(n)):
            if rng.rand() < p:
                edges.append((int(i), int(j)))
    return edges


def build_random_digraph_edges(n: int, p: float, rng: np.random.RandomState) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    p = float(p)
    for i in range(int(n)):
        for j in range(int(n)):
            if i == j:
                continue
            if rng.rand() < p:
                edges.append((int(i), int(j)))
    return edges


def build_graph_edges(
    *,
    n: int,
    family: str,
    p: Optional[float],
    seed: int,
) -> List[Tuple[int, int]]:
    family = str(family).strip().lower()
    rng = np.random.RandomState(int(seed))
    if family == "total_order":
        return build_total_order_edges(int(n))
    if family == "random_dag":
        if p is None:
            raise ValueError("random_dag requires p.")
        return build_random_dag_edges(int(n), float(p), rng)
    if family == "random_digraph":
        if p is None:
            raise ValueError("random_digraph requires p.")
        return build_random_digraph_edges(int(n), float(p), rng)
    raise ValueError(f"Unknown graph family: {family}")


def sample_lambda_iid(n: int, u_lambda: float, rng: np.random.RandomState) -> np.ndarray:
    return rng.uniform(1.0, float(u_lambda), size=int(n)).astype(np.float64)


def attach_omega_iid(
    base_edges: Sequence[Tuple[int, int]],
    omega_interval: Tuple[float, float],
    rng: np.random.RandomState,
) -> List[Tuple[int, int, float]]:
    lo, hi = float(omega_interval[0]), float(omega_interval[1])
    if lo > hi:
        raise ValueError(f"Invalid omega interval: ({lo}, {hi})")
    out: List[Tuple[int, int, float]] = []
    for u, v in base_edges:
        w = float(rng.uniform(lo, hi))
        w = min(max(w, EPS_OMEGA), 1.0)
        out.append((int(u), int(v), w))
    return out


def normalize_init_mode(init_mode: str) -> str:
    mode = str(init_mode).strip().lower()
    if mode == "topological":
        return "greedy"
    if mode in {"random", "greedy"}:
        return mode
    raise ValueError(f"Unknown init_mode: {init_mode}")


def to_raw_params(
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


def sample_random_init_perms(n: int, num_chains: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    out = np.empty((int(num_chains), int(n)), dtype=np.int64)
    for idx in range(int(num_chains)):
        out[idx, :] = rng.permutation(int(n))
    return out


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
            probs = choice_weights / weight_sum
            chosen = int(rng.choice(active, p=probs))

        ordering[pos] = chosen
        remaining[chosen] = False
        for pred, weight in incoming[chosen]:
            if remaining[int(pred)]:
                scores[int(pred)] -= float(weight)

    return ordering


def sample_greedy_init_perms(
    n: int,
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    num_chains: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    out = np.empty((int(num_chains), int(n)), dtype=np.int64)
    for idx in range(int(num_chains)):
        chain_seed = int(rng.randint(0, np.iinfo(np.int32).max))
        out[idx, :] = sample_greedy_order(
            int(n),
            np.asarray(lambdas, dtype=np.float64),
            edges_with_omega,
            chain_seed,
        )
    return out
