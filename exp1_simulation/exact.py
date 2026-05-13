from typing import List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    from sampler import njit
else:
    from .sampler import njit

EPS_OMEGA = 1e-12
MAX_EXACT_N = 30


def _edge_factor_matrix(n: int, edges_with_omega: Sequence[Tuple[int, int, float]]) -> np.ndarray:
    factors = np.ones((int(n), int(n)), dtype=np.float64)
    for u, v, omega in edges_with_omega:
        w = float(min(max(float(omega), EPS_OMEGA), 1.0))
        factors[int(u), int(v)] = w
    return factors


@njit(cache=True)
def _exact_pairwise_subset_dp_nb(lam: np.ndarray, factors: np.ndarray) -> Tuple[np.ndarray, float]:
    n = int(lam.shape[0])
    m = 1 << n
    full = m - 1

    bit_index = np.empty(m, dtype=np.int64)
    bit_index[0] = -1
    for i in range(n):
        bit_index[1 << i] = i

    g = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        g[i, 0] = 1.0

    for mask in range(1, m):
        lsb = mask & -mask
        b = bit_index[lsb]
        prev = mask ^ lsb
        for i in range(n):
            g[i, mask] = g[i, prev] * factors[i, b]

    gsum = np.zeros(m, dtype=np.float64)
    hsum = np.zeros(m, dtype=np.float64)
    for mask in range(1, m):
        tmp = mask
        s_g = 0.0
        s_h = 0.0
        while tmp != 0:
            lsb = tmp & -tmp
            i = bit_index[lsb]
            gi = g[i, mask]
            s_g += gi
            s_h += lam[i] * gi
            tmp ^= lsb
        gsum[mask] = s_g
        hsum[mask] = s_h

    forward = np.zeros(m, dtype=np.float64)
    forward[0] = 1.0
    for mask in range(1, m):
        denom = hsum[mask]
        if denom <= 0.0:
            forward[mask] = 0.0
            continue
        state_factor = gsum[mask] / denom
        total = 0.0
        tmp = mask
        while tmp != 0:
            lsb = tmp & -tmp
            i = bit_index[lsb]
            prev = mask ^ lsb
            weight_i = lam[i] * g[i, mask] * state_factor
            total += forward[prev] * weight_i
            tmp ^= lsb
        forward[mask] = total

    z = forward[full]
    if z <= 0.0:
        return np.zeros((n, n), dtype=np.float64), 0.0

    popcount = np.zeros(m, dtype=np.int64)
    for mask in range(1, m):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    backward = np.zeros(m, dtype=np.float64)
    backward[full] = 1.0
    for size in range(n - 1, -1, -1):
        for mask in range(m):
            if popcount[mask] != size:
                continue
            rem = full ^ mask
            if rem == 0:
                continue
            total = 0.0
            tmp = rem
            while tmp != 0:
                lsb = tmp & -tmp
                k = bit_index[lsb]
                nxt = mask | lsb
                denom = hsum[nxt]
                if denom > 0.0:
                    state_factor = gsum[nxt] / denom
                    weight_k = lam[k] * g[k, nxt] * state_factor
                    total += weight_k * backward[nxt]
                tmp ^= lsb
            backward[mask] = total

    pair = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        bit_j = 1 << j
        for tmask in range(m):
            if (tmask & bit_j) != 0:
                continue
            smask = tmask | bit_j
            denom = hsum[smask]
            if denom <= 0.0:
                continue
            state_factor = gsum[smask] / denom
            weight_j = lam[j] * g[j, smask] * state_factor
            contrib = forward[tmask] * weight_j * backward[smask]
            if contrib == 0.0:
                continue
            tmp = tmask
            while tmp != 0:
                lsb = tmp & -tmp
                i = bit_index[lsb]
                pair[i, j] += contrib
                tmp ^= lsb

    for i in range(n):
        pair[i, i] = 0.0
    pair /= z
    return pair, z


def exact_pairwise_matrix_subset_dp(
    lambdas: np.ndarray,
    edges_with_omega: Sequence[Tuple[int, int, float]],
    check_symmetry: bool = True,
) -> Tuple[np.ndarray, float]:
    lambdas = np.asarray(lambdas, dtype=np.float64)
    n = int(lambdas.shape[0])
    if n <= 0:
        raise ValueError("n must be positive.")
    if n > MAX_EXACT_N:
        raise ValueError(f"Exact subset-DP is configured for n <= {MAX_EXACT_N}, got n={n}.")

    factors = _edge_factor_matrix(n, edges_with_omega)
    pair, z = _exact_pairwise_subset_dp_nb(lambdas, factors)

    if check_symmetry:
        for i in range(n):
            for j in range(i + 1, n):
                s = float(pair[i, j] + pair[j, i])
                if s > 0.0:
                    pair[i, j] /= s
                    pair[j, i] /= s
                else:
                    pair[i, j] = 0.5
                    pair[j, i] = 0.5
        np.fill_diagonal(pair, 0.0)

    return pair.astype(np.float64), float(z)
