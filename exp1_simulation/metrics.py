from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    from sampler import NUMBA_AVAILABLE, njit
else:
    from .sampler import NUMBA_AVAILABLE, njit


def build_doubling_checkpoints(
    *,
    max_t: int,
    t0: int = 1,
) -> List[int]:
    max_t = int(max_t)
    if max_t <= 0:
        return []
    t0 = max(1, int(t0))
    if t0 > max_t:
        return []

    out: List[int] = []
    t = int(t0)
    while t <= max_t:
        out.append(int(t))
        t *= 2
    return out


@njit(cache=True)
def _pairwise_counts_nb(samples: np.ndarray) -> np.ndarray:
    num_samples, n = samples.shape
    counts = np.zeros((n, n), dtype=np.int64)
    pos = np.empty(n, dtype=np.int64)

    for row in range(num_samples):
        for idx in range(n):
            pos[samples[row, idx]] = idx
        for i in range(n - 1):
            pi = pos[i]
            for j in range(i + 1, n):
                if pi < pos[j]:
                    counts[i, j] += 1
                else:
                    counts[j, i] += 1
    return counts


def empirical_pairwise_matrix(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim != 2:
        raise ValueError(f"Expected 2D samples, got shape {samples.shape}")
    if samples.shape[0] <= 0:
        raise ValueError("Need at least one sample permutation.")

    if NUMBA_AVAILABLE:
        counts = _pairwise_counts_nb(samples)
    else:
        n = int(samples.shape[1])
        counts = np.zeros((n, n), dtype=np.int64)
        pos = np.empty(n, dtype=np.int64)
        for row in samples:
            for idx, node in enumerate(row.tolist()):
                pos[int(node)] = int(idx)
            for i in range(n - 1):
                pi = int(pos[i])
                for j in range(i + 1, n):
                    if pi < int(pos[j]):
                        counts[i, j] += 1
                    else:
                        counts[j, i] += 1

    probs = counts.astype(np.float64) / float(samples.shape[0])
    np.fill_diagonal(probs, 0.0)
    return probs


def exact_discrepancy(m_emp: np.ndarray, m_exact: np.ndarray) -> float:
    m_emp = np.asarray(m_emp, dtype=np.float64)
    m_exact = np.asarray(m_exact, dtype=np.float64)
    return float(np.max(np.abs(m_emp - m_exact)))


def cauchy_like_gap(m_t: np.ndarray, m_2t: np.ndarray) -> float:
    m_t = np.asarray(m_t, dtype=np.float64)
    m_2t = np.asarray(m_2t, dtype=np.float64)
    return float(np.max(np.abs(m_t - m_2t)))


def between_start_gap(m_rand: np.ndarray, m_greedy: np.ndarray) -> float:
    m_rand = np.asarray(m_rand, dtype=np.float64)
    m_greedy = np.asarray(m_greedy, dtype=np.float64)
    return float(np.max(np.abs(m_rand - m_greedy)))


def first_time_below_band(
    *,
    values: Sequence[float],
    checkpoints: Sequence[int],
    epsilon: float,
    epsilon0: float,
) -> Optional[int]:
    low = float(epsilon) - float(epsilon0)
    for t, v in zip(checkpoints, values):
        if float(v) <= low:
            return int(t)
    return None


def classify_curve_by_band(
    *,
    values: Sequence[float],
    epsilon: float,
    epsilon0: float,
) -> Dict[str, List[int]]:
    low = float(epsilon) - float(epsilon0)
    high = float(epsilon) + float(epsilon0)
    below: List[int] = []
    middle: List[int] = []
    above: List[int] = []
    for idx, value in enumerate(values):
        v = float(value)
        if v <= low:
            below.append(int(idx))
        elif v > high:
            above.append(int(idx))
        else:
            middle.append(int(idx))
    return {"below": below, "middle": middle, "above": above}
