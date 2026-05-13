import math
import random
from typing import List, Optional, Tuple

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _deco(f):
            return f
        return _deco

# -----------------------------
# Graph builders
# -----------------------------
def build_adjacency(n, edges):
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]
    for u, v, w in edges:
        w = float(w)
        out_adj[u].append((v, w))
        in_adj[v].append((u, w))

    # sort in adjacency for two pointer merge
    for v in range(n):
        in_adj[v].sort(key=lambda x: x[0])

    return out_adj, in_adj


def _adj_to_arrays(adj):
    """Convert adjacency list-of-lists into CSR-like arrays."""
    n = len(adj)
    indptr = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for i in range(n):
        total += len(adj[i])
        indptr[i + 1] = total
    indices = np.zeros(total, dtype=np.int64)
    weights = np.zeros(total, dtype=np.float64)
    p = 0
    for i in range(n):
        for j, w in adj[i]:
            indices[p] = int(j)
            weights[p] = float(w)
            p += 1
    return indptr, indices, weights

def random_digraph_edges(n, out_degree=10, weight_scale=1.0, seed=0, allow_self_loops=False):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        if allow_self_loops:
            candidates = list(range(n))
        else:
            candidates = [v for v in range(n) if v != u]

        if out_degree >= len(candidates):
            chosen = candidates
        else:
            chosen = rng.sample(candidates, out_degree)

        for v in chosen:
            w = weight_scale * rng.random()
            edges.append((u, v, w))
    return edges

# -----------------------------
# Core utilities
# -----------------------------
def perm_to_pos(pi, n):
    pos = [0] * n
    for idx, node in enumerate(pi):
        pos[node] = idx
    return pos

def V_prefix(k, t_max, pos, out_adj):
    s = 0.0
    for j, w in out_adj[k]:
        if pos[j] <= t_max:
            s += w
    return s


@njit(cache=True)
def _V_prefix_nb(k, t_max, pos, out_indptr, out_indices, out_weights):
    s = 0.0
    st = out_indptr[k]
    en = out_indptr[k + 1]
    for ptr in range(st, en):
        j = out_indices[ptr]
        w = out_weights[ptr]
        if pos[j] <= t_max:
            s += w
    return s

def V_prefix_with_b(b, cut_i0, b_node, pos, out_adj):
    s = 0.0
    for j, w in out_adj[b]:
        if (pos[j] < cut_i0) or (j == b_node):
            s += w
    return s


@njit(cache=True)
def _V_prefix_with_b_nb(b, cut_i0, b_node, pos, out_indptr, out_indices, out_weights):
    s = 0.0
    st = out_indptr[b]
    en = out_indptr[b + 1]
    for ptr in range(st, en):
        j = out_indices[ptr]
        w = out_weights[ptr]
        if (pos[j] < cut_i0) or (j == b_node):
            s += w
    return s

def delta_adjacent_swap(pi, pos, i0, out_adj):
    a = pi[i0]
    b = pi[i0 + 1]

    term1 = V_prefix_with_b(b, i0, b, pos, out_adj)
    term2 = V_prefix(a, i0, pos, out_adj)
    term3 = V_prefix(a, i0 + 1, pos, out_adj)
    term4 = V_prefix(b, i0 + 1, pos, out_adj)

    return term1 - term2 + term3 - term4


def _logsumexp(log_terms):
    """Stable log(sum(exp(log_terms)))."""
    if not log_terms:
        return -math.inf
    m = max(log_terms)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in log_terms))


def log_unnormalized_pmf(pi, lam, alpha, beta, out_adj):
    n = len(pi)
    pos = perm_to_pos(pi, n)
    logp = 0.0

    for t in range(n):
        num_log = None
        log_c_terms = []
        log_b_terms = []

        for k in pi[: t + 1]:
            v_k = V_prefix(k, t, pos, out_adj)
            log_g_k = -beta * v_k
            log_c_terms.append(log_g_k)

            log_bg_k = (-alpha * lam[k]) + log_g_k
            log_b_terms.append(log_bg_k)

            if k == pi[t]:
                num_log = log_bg_k

        if num_log is None:
            raise RuntimeError("Current item was not found in prefix.")

        log_c_t = _logsumexp(log_c_terms)
        log_b_t = _logsumexp(log_b_terms)
        if log_c_t == -math.inf or log_b_t == -math.inf:
            return -math.inf

        logp += num_log + log_c_t - log_b_t

    return logp


def unnormalized_pmf(pi, lam, alpha, beta, out_adj):
    """
    Unnormalized pmf value. Prefer log_unnormalized_pmf() for numerical work.
    """
    lp = log_unnormalized_pmf(pi, lam, alpha, beta, out_adj)
    if lp == -math.inf:
        return 0.0
    try:
        return math.exp(lp)
    except OverflowError:
        return math.inf

# -----------------------------
# Cache for U_i, W_i and g_i
# -----------------------------
def cache_build_prefix_stats(pi, pos, i0, beta, out_adj, exp_node):
    U = 0.0
    W = 0.0
    g_map = {}

    prefix_nodes = pi[:i0 + 1]
    for k in prefix_nodes:
        v_k = V_prefix(k, i0, pos, out_adj)
        g = math.exp(-beta * v_k)
        g_map[k] = g
        U += g
        W += exp_node[k] * g

    return U, W, g_map

def cache_get_prefix_stats(pi, pos, i0, beta, out_adj, exp_node, U_cache, W_cache, g_cache, has_cache):
    if has_cache[i0]:
        return U_cache[i0], W_cache[i0], g_cache[i0]

    U, W, g_map = cache_build_prefix_stats(pi, pos, i0, beta, out_adj, exp_node)
    U_cache[i0] = U
    W_cache[i0] = W
    g_cache[i0] = g_map
    has_cache[i0] = True
    return U, W, g_map

# -----------------------------
# Proposal update 
# -----------------------------
def prefix_stats_after_swap(
    pi, pos, i0, beta, out_adj, in_adj, exp_node, U_cur, W_cur, g_map_cur
):
    a = pi[i0]
    b = pi[i0 + 1]

    # remove a using cached g
    g_a = g_map_cur[a]
    U_prop = U_cur - g_a
    W_prop = W_cur - exp_node[a] * g_a

    # add b needs g_b computed once
    v_b = V_prefix_with_b(b, i0, b, pos, out_adj)
    g_b = math.exp(-beta * v_b)
    U_prop += g_b
    W_prop += exp_node[b] * g_b

    # merge in neighbors of a and b
    neigh_a = in_adj[a]
    neigh_b = in_adj[b]
    ia = 0
    ib = 0

    # record only changed g values, apply on accept
    # entries are (k, g_new)
    updates = []

    while ia < len(neigh_a) or ib < len(neigh_b):
        if ib >= len(neigh_b):
            k = neigh_a[ia][0]
            w_ka = neigh_a[ia][1]
            w_kb = 0.0
            ia += 1
        elif ia >= len(neigh_a):
            k = neigh_b[ib][0]
            w_ka = 0.0
            w_kb = neigh_b[ib][1]
            ib += 1
        else:
            ka = neigh_a[ia][0]
            kb = neigh_b[ib][0]
            if ka == kb:
                k = ka
                w_ka = neigh_a[ia][1]
                w_kb = neigh_b[ib][1]
                ia += 1
                ib += 1
            elif ka < kb:
                k = ka
                w_ka = neigh_a[ia][1]
                w_kb = 0.0
                ia += 1
            else:
                k = kb
                w_ka = 0.0
                w_kb = neigh_b[ib][1]
                ib += 1

        # only k in prefix and k not equal to a can change V
        if k == a:
            continue
        if pos[k] > i0:
            continue

        dv = w_kb - w_ka
        if dv == 0.0:
            continue

        # g_old from cache, no recompute of V
        g_old = g_map_cur.get(k)

        g_new = g_old * math.exp(-beta * dv)
        dg = g_new - g_old

        U_prop += dg
        W_prop += exp_node[k] * dg
        updates.append((k, g_new))

    return U_prop, W_prop, g_b, updates

def log_accept_ratio(
    pi, pos, i0, beta, out_adj, in_adj, exp_node,
    U_cache, W_cache, g_cache, has_cache
):
    Delta = delta_adjacent_swap(pi, pos, i0, out_adj)

    U_cur, W_cur, g_map_cur = cache_get_prefix_stats(
        pi, pos, i0, beta, out_adj, exp_node,
        U_cache, W_cache, g_cache, has_cache
    )

    if U_cur <= 0.0:
        mbar_cur = 0.0
    else:
        mbar_cur = W_cur / U_cur

    U_prop, W_prop, g_b, updates = prefix_stats_after_swap(
        pi, pos, i0, beta, out_adj, in_adj, exp_node, U_cur, W_cur, g_map_cur
    )

    if U_prop <= 0.0:
        mbar_prop = 0.0
    else:
        mbar_prop = W_prop / U_prop

    if mbar_cur <= 0.0 or mbar_prop <= 0.0:
        if mbar_cur == mbar_prop:
            log_mbar_ratio = 0.0
        elif mbar_cur <= 0.0 and mbar_prop > 0.0:
            log_mbar_ratio = -math.inf
        else:
            log_mbar_ratio = math.inf
    else:
        log_mbar_ratio = math.log(mbar_cur) - math.log(mbar_prop)

    log_r = (-beta * Delta) + log_mbar_ratio
    return log_r, U_prop, W_prop, g_b, updates

# -----------------------------
# Sampler
# -----------------------------
@njit(cache=True)
def _mh_collect_samples_nb(
    pi,
    pos,
    beta,
    out_indptr,
    out_indices,
    out_weights,
    in_indptr,
    in_indices,
    in_weights,
    exp_node,
    lazy_prob,
    n_samples,
    burn_in,
    thin,
    seed,
):
    np.random.seed(seed)

    n = pi.shape[0]
    samples = np.empty((n_samples, n), dtype=np.int64)

    U_cache = np.zeros(n - 1, dtype=np.float64)
    W_cache = np.zeros(n - 1, dtype=np.float64)
    has_cache = np.zeros(n - 1, dtype=np.uint8)
    g_cache = np.zeros((n - 1, n), dtype=np.float64)
    upd_keys = np.zeros(n, dtype=np.int64)
    upd_vals = np.zeros(n, dtype=np.float64)

    accept = 0
    total_steps = 0
    num_collected = 0

    while num_collected < n_samples:
        total_steps += 1

        if np.random.rand() >= lazy_prob:
            i0 = np.random.randint(0, n - 1)

            log_r, U_prop, W_prop, g_b, n_updates = _log_accept_ratio_nb(
                pi,
                pos,
                i0,
                beta,
                out_indptr,
                out_indices,
                out_weights,
                in_indptr,
                in_indices,
                in_weights,
                exp_node,
                U_cache,
                W_cache,
                g_cache,
                has_cache,
                upd_keys,
                upd_vals,
            )

            if math.log(np.random.rand()) < (0.0 if log_r > 0.0 else log_r):
                a = pi[i0]
                b = pi[i0 + 1]

                pi[i0], pi[i0 + 1] = b, a
                pos[a], pos[b] = i0 + 1, i0

                U_cache[i0] = U_prop
                W_cache[i0] = W_prop
                has_cache[i0] = 1
                g_cache[i0, a] = 0.0
                g_cache[i0, b] = g_b
                for u in range(n_updates):
                    g_cache[i0, upd_keys[u]] = upd_vals[u]

                accept += 1

        if total_steps >= burn_in and ((total_steps - burn_in) % thin == 0):
            samples[num_collected, :] = pi
            num_collected += 1

    return samples, accept, total_steps


@njit(cache=True)
def _mh_collect_uniform_samples_nb(
    pi,
    lazy_prob,
    n_samples,
    burn_in,
    thin,
    seed,
):
    np.random.seed(seed)

    n = pi.shape[0]
    samples = np.empty((n_samples, n), dtype=np.int64)

    accept = 0
    total_steps = 0
    num_collected = 0

    while num_collected < n_samples:
        total_steps += 1

        if np.random.rand() >= lazy_prob:
            i0 = np.random.randint(0, n - 1)
            a = pi[i0]
            b = pi[i0 + 1]
            pi[i0], pi[i0 + 1] = b, a
            accept += 1

        if total_steps >= burn_in and ((total_steps - burn_in) % thin == 0):
            samples[num_collected, :] = pi
            num_collected += 1

    return samples, accept, total_steps


def _reference_mh_sampler(
    n,
    edges,
    lam,
    alpha=1.0,
    beta=1.0,
    seed=0,
    n_samples=1000,
    burn_in=50000,
    thin=200,
    init_pi=None,
    report_every=100,
):
    rng = random.Random(seed)
    out_adj, in_adj = build_adjacency(n, edges)
    use_numba_fastpath = NUMBA_AVAILABLE
    if use_numba_fastpath:
        out_indptr, out_indices, out_weights = _adj_to_arrays(out_adj)
        in_indptr, in_indices, in_weights = _adj_to_arrays(in_adj)

    # probability of a lazy step (stay in place without proposing a swap)
    lazy_prob = 0.5

    exp_node = [math.exp(-alpha * lam[k]) for k in range(n)]
    exp_node_arr = np.asarray(exp_node, dtype=np.float64)

    if init_pi is None:
        pi = list(range(n))
        rng.shuffle(pi)
    else:
        pi = list(init_pi)

    pos = perm_to_pos(pi, n)
    if use_numba_fastpath:
        pi_arr = np.asarray(pi, dtype=np.int64)
        pos_arr = np.asarray(pos, dtype=np.int64)

    U_cache = [0.0] * (n - 1)
    W_cache = [0.0] * (n - 1)
    g_cache = [None] * (n - 1)
    has_cache = [False] * (n - 1)
    if use_numba_fastpath:
        U_cache_nb = np.zeros(n - 1, dtype=np.float64)
        W_cache_nb = np.zeros(n - 1, dtype=np.float64)
        has_cache_nb = np.zeros(n - 1, dtype=np.uint8)
        g_cache_nb = np.zeros((n - 1, n), dtype=np.float64)
        upd_keys = np.zeros(n, dtype=np.int64)
        upd_vals = np.zeros(n, dtype=np.float64)

    samples = []
    accept = 0
    total_steps = 0

    # for reporting based on samples
    accepted_since_report = 0
    steps_since_report = 0
    last_reported_samples = 0

    logu = math.log
    rand = rng.random
    randrange = rng.randrange

    while len(samples) < n_samples:
        # lazy step: with probability lazy_prob, do nothing and stay in place
        total_steps += 1
        steps_since_report += 1

        if rand() >= lazy_prob:
            i0 = randrange(0, n - 1)

            if use_numba_fastpath:
                (
                    log_r,
                    U_prop,
                    W_prop,
                    g_b,
                    n_updates,
                ) = _log_accept_ratio_nb(
                    pi_arr,
                    pos_arr,
                    i0,
                    beta,
                    out_indptr,
                    out_indices,
                    out_weights,
                    in_indptr,
                    in_indices,
                    in_weights,
                    exp_node_arr,
                    U_cache_nb,
                    W_cache_nb,
                    g_cache_nb,
                    has_cache_nb,
                    upd_keys,
                    upd_vals,
                )
            else:
                log_r, U_prop, W_prop, g_b, updates = log_accept_ratio(
                    pi, pos, i0, beta, out_adj, in_adj, exp_node,
                    U_cache, W_cache, g_cache, has_cache
                )

            accepted = False
            if logu(rand()) < (0.0 if log_r > 0.0 else log_r):
                accepted = True
                a = pi[i0]
                b = pi[i0 + 1]

                pi[i0], pi[i0 + 1] = b, a
                pos[a], pos[b] = i0 + 1, i0
                if use_numba_fastpath:
                    pi_arr[i0], pi_arr[i0 + 1] = b, a
                    pos_arr[a], pos_arr[b] = i0 + 1, i0

                if use_numba_fastpath:
                    U_cache_nb[i0] = U_prop
                    W_cache_nb[i0] = W_prop
                    has_cache_nb[i0] = 1
                    g_cache_nb[i0, a] = 0.0
                    g_cache_nb[i0, b] = g_b
                    for u in range(n_updates):
                        g_cache_nb[i0, upd_keys[u]] = upd_vals[u]
                else:
                    U_cache[i0] = U_prop
                    W_cache[i0] = W_prop
                    has_cache[i0] = True

                    g_map = g_cache[i0]
                    if g_map is None:
                        g_map = {}
                        g_cache[i0] = g_map

                    if a in g_map:
                        del g_map[a]
                    g_map[b] = g_b
                    for k, g_new in updates:
                        g_map[k] = g_new

                accept += 1
                accepted_since_report += 1

        if total_steps >= burn_in and ((total_steps - burn_in) % thin == 0):
            samples.append(tuple(pi))

            # report based on number of newly collected samples
            if report_every is not None and report_every > 0:
                if len(samples) - last_reported_samples >= report_every:
                    new_samples = len(samples) - last_reported_samples
                    print(
                        "num_samples", len(samples),
                        "new_samples", new_samples,
                        "accept_rate_since_report", round(accepted_since_report / max(1, steps_since_report), 4),
                        "overall_accept_rate", round(accept / total_steps, 4),
                        "total_steps", total_steps,
                    )
                    last_reported_samples = len(samples)
                    accepted_since_report = 0
                    steps_since_report = 0

    return samples, accept / (total_steps if total_steps else 1)


@njit(cache=True)
def _log_accept_ratio_nb(
    pi,
    pos,
    i0,
    beta,
    out_indptr,
    out_indices,
    out_weights,
    in_indptr,
    in_indices,
    in_weights,
    exp_node,
    U_cache,
    W_cache,
    g_cache,
    has_cache,
    upd_keys,
    upd_vals,
):
    a = pi[i0]
    b = pi[i0 + 1]

    term1 = _V_prefix_with_b_nb(b, i0, b, pos, out_indptr, out_indices, out_weights)
    term2 = _V_prefix_nb(a, i0, pos, out_indptr, out_indices, out_weights)
    term3 = _V_prefix_nb(a, i0 + 1, pos, out_indptr, out_indices, out_weights)
    term4 = _V_prefix_nb(b, i0 + 1, pos, out_indptr, out_indices, out_weights)
    Delta = term1 - term2 + term3 - term4

    if has_cache[i0] == 0:
        U_cur = 0.0
        W_cur = 0.0
        for t in range(i0 + 1):
            k = pi[t]
            v_k = _V_prefix_nb(k, i0, pos, out_indptr, out_indices, out_weights)
            g = math.exp(-beta * v_k)
            g_cache[i0, k] = g
            U_cur += g
            W_cur += exp_node[k] * g
        U_cache[i0] = U_cur
        W_cache[i0] = W_cur
        has_cache[i0] = 1
    else:
        U_cur = U_cache[i0]
        W_cur = W_cache[i0]

    if U_cur <= 0.0:
        mbar_cur = 0.0
    else:
        mbar_cur = W_cur / U_cur

    g_a = g_cache[i0, a]
    U_prop = U_cur - g_a
    W_prop = W_cur - exp_node[a] * g_a

    v_b = _V_prefix_with_b_nb(b, i0, b, pos, out_indptr, out_indices, out_weights)
    g_b = math.exp(-beta * v_b)
    U_prop += g_b
    W_prop += exp_node[b] * g_b

    ia = in_indptr[a]
    ia_end = in_indptr[a + 1]
    ib = in_indptr[b]
    ib_end = in_indptr[b + 1]
    n_updates = 0

    while ia < ia_end or ib < ib_end:
        if ib >= ib_end:
            k = in_indices[ia]
            w_ka = in_weights[ia]
            w_kb = 0.0
            ia += 1
        elif ia >= ia_end:
            k = in_indices[ib]
            w_ka = 0.0
            w_kb = in_weights[ib]
            ib += 1
        else:
            ka = in_indices[ia]
            kb = in_indices[ib]
            if ka == kb:
                k = ka
                w_ka = in_weights[ia]
                w_kb = in_weights[ib]
                ia += 1
                ib += 1
            elif ka < kb:
                k = ka
                w_ka = in_weights[ia]
                w_kb = 0.0
                ia += 1
            else:
                k = kb
                w_ka = 0.0
                w_kb = in_weights[ib]
                ib += 1

        if k == a:
            continue
        if pos[k] > i0:
            continue
        dv = w_kb - w_ka
        if dv == 0.0:
            continue

        g_old = g_cache[i0, k]
        g_new = g_old * math.exp(-beta * dv)
        dg = g_new - g_old

        U_prop += dg
        W_prop += exp_node[k] * dg
        upd_keys[n_updates] = k
        upd_vals[n_updates] = g_new
        n_updates += 1

    if U_prop <= 0.0:
        mbar_prop = 0.0
    else:
        mbar_prop = W_prop / U_prop

    if mbar_cur <= 0.0 or mbar_prop <= 0.0:
        if mbar_cur == mbar_prop:
            log_mbar_ratio = 0.0
        elif mbar_cur <= 0.0 and mbar_prop > 0.0:
            log_mbar_ratio = -math.inf
        else:
            log_mbar_ratio = math.inf
    else:
        log_mbar_ratio = math.log(mbar_cur) - math.log(mbar_prop)

    log_r = (-beta * Delta) + log_mbar_ratio
    return log_r, U_prop, W_prop, g_b, n_updates


def mh_sampler(
    n,
    edges,
    lam,
    alpha=1.0,
    beta=1.0,
    seed=0,
    n_samples=1000,
    burn_in=50000,
    thin=200,
    init_pi=None,
    report_every=100,
    lazy_prob=0.5,
):
    """
    Simulation-oriented accelerated sampler.

    This file intentionally contains both the generic reference sampler and the
    accelerated simulation path so exp1_simulation stays self-contained.
    """
    if not NUMBA_AVAILABLE:
        return _reference_mh_sampler(
            n=n,
            edges=edges,
            lam=lam,
            alpha=alpha,
            beta=beta,
            seed=seed,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            init_pi=init_pi,
            report_every=report_every,
        )

    out_adj, in_adj = build_adjacency(int(n), edges)
    out_indptr, out_indices, out_weights = _adj_to_arrays(out_adj)
    in_indptr, in_indices, in_weights = _adj_to_arrays(in_adj)

    exp_node = np.exp(-float(alpha) * np.asarray(lam, dtype=np.float64))
    is_uniform_target = (len(edges) == 0) and np.allclose(exp_node, exp_node[0])

    if init_pi is None:
        rng = np.random.RandomState(int(seed))
        pi = rng.permutation(int(n)).astype(np.int64)
    else:
        pi = np.asarray(init_pi, dtype=np.int64).copy()

    pos = np.empty(int(n), dtype=np.int64)
    pos[pi] = np.arange(int(n), dtype=np.int64)

    n_samples = int(n_samples)
    burn_in = int(burn_in)
    thin = int(thin)
    report_every = int(report_every) if report_every is not None else 0
    seed = int(seed)

    if report_every <= 0:
        if is_uniform_target:
            samples, accept, total_steps = _mh_collect_uniform_samples_nb(
                pi=pi,
                lazy_prob=float(lazy_prob),
                n_samples=n_samples,
                burn_in=burn_in,
                thin=thin,
                seed=seed,
            )
        else:
            samples, accept, total_steps = _mh_collect_samples_nb(
                pi=pi,
                pos=pos,
                beta=float(beta),
                out_indptr=out_indptr,
                out_indices=out_indices,
                out_weights=out_weights,
                in_indptr=in_indptr,
                in_indices=in_indices,
                in_weights=in_weights,
                exp_node=exp_node,
                lazy_prob=float(lazy_prob),
                n_samples=n_samples,
                burn_in=burn_in,
                thin=thin,
                seed=seed,
            )
        return samples, float(accept) / float(max(total_steps, 1))

    chunks = []
    total_accept = 0
    total_steps = 0
    collected = 0
    chunk_id = 0
    current_burn_in = burn_in

    while collected < n_samples:
        chunk_id += 1
        chunk_size = min(report_every, n_samples - collected)
        chunk_seed = seed + chunk_id - 1
        if is_uniform_target:
            chunk_samples, chunk_accept, chunk_steps = _mh_collect_uniform_samples_nb(
                pi=pi,
                lazy_prob=float(lazy_prob),
                n_samples=chunk_size,
                burn_in=current_burn_in,
                thin=thin,
                seed=chunk_seed,
            )
        else:
            chunk_samples, chunk_accept, chunk_steps = _mh_collect_samples_nb(
                pi=pi,
                pos=pos,
                beta=float(beta),
                out_indptr=out_indptr,
                out_indices=out_indices,
                out_weights=out_weights,
                in_indptr=in_indptr,
                in_indices=in_indices,
                in_weights=in_weights,
                exp_node=exp_node,
                lazy_prob=float(lazy_prob),
                n_samples=chunk_size,
                burn_in=current_burn_in,
                thin=thin,
                seed=chunk_seed,
            )
        chunks.append(chunk_samples)
        collected += chunk_size
        total_accept += int(chunk_accept)
        total_steps += int(chunk_steps)
        print(
            "num_samples", collected,
            "new_samples", chunk_size,
            "accept_rate_since_report", round(float(chunk_accept) / max(1, int(chunk_steps)), 4),
            "overall_accept_rate", round(float(total_accept) / max(1, int(total_steps)), 4),
            "total_steps", total_steps,
        )
        current_burn_in = 0

    samples = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
    return samples, float(total_accept) / float(max(total_steps, 1))


def draw_independent_final_permutations(
    *,
    n,
    edges,
    lam,
    steps,
    num_chains,
    chain_seeds,
    init_perms: Optional[np.ndarray] = None,
    alpha=1.0,
    beta=1.0,
    lazy_prob=0.5,
):
    """
    Draw one final permutation from each independent chain after `steps` MH moves.

    This helper is used by mixing experiments where we need the chain law P^t
    from many independent starts.
    """
    n = int(n)
    steps = int(steps)
    num_chains = int(num_chains)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    seeds = np.asarray(chain_seeds, dtype=np.int64)
    if seeds.shape[0] != num_chains:
        raise ValueError(f"chain_seeds length mismatch: expected {num_chains}, got {seeds.shape[0]}")

    if init_perms is not None:
        init_perms = np.asarray(init_perms, dtype=np.int64)
        if init_perms.shape != (num_chains, n):
            raise ValueError(
                f"init_perms shape mismatch: expected {(num_chains, n)}, got {tuple(init_perms.shape)}"
            )

    out = np.empty((num_chains, n), dtype=np.int64)
    acc_rates = np.empty(num_chains, dtype=np.float64)

    for idx in range(num_chains):
        init_pi = None if init_perms is None else init_perms[idx, :]
        samples, acceptance_rate = mh_sampler(
            n=n,
            edges=edges,
            lam=lam,
            alpha=float(alpha),
            beta=float(beta),
            seed=int(seeds[idx]),
            n_samples=1,
            burn_in=steps,
            thin=1,
            init_pi=init_pi,
            report_every=0,
            lazy_prob=float(lazy_prob),
        )
        out[idx, :] = np.asarray(samples[0], dtype=np.int64)
        acc_rates[idx] = float(acceptance_rate)

    return out, float(np.mean(acc_rates))

# ===========================================================================
# PASV adjacent-swap Metropolis-Hastings sampler.
#
# Mirrors the GPASV sampler above but targets the PASV distribution on the
# induced DAG: support is restricted to linear extensions Pi^{<=}, with the
# stage-wise factor max_<=(S_t) replacing the GPASV V_prefix term.
# ===========================================================================

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _deco(f):
            return f
        return _deco


MAX_N_BITMASK = 63  # uint64 has 64 bits, reserve one to avoid sign issues


# -----------------------------
# DAG utilities (reachability)
# -----------------------------
def _normalize_edge_pairs(edges) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if u == v:
            continue
        pairs.append((u, v))
    return pairs


def compute_reach_bitset(n: int, edges) -> np.ndarray:
    """
    Compute the strict-descendants bitmask of a DAG.

    Returns reach (np.uint64, length n) where bit j of reach[k] is set iff
    there is a directed path from k to j (k itself is NOT included in reach[k]).

    Assumes the DAG is acyclic but does not assume topological indexing.
    """
    if int(n) > MAX_N_BITMASK:
        raise ValueError(
            f"compute_reach_bitset uses a uint64 bitmask; requires n<={MAX_N_BITMASK}, got {n}"
        )
    n = int(n)
    pairs = _normalize_edge_pairs(edges)
    in_adj: List[List[int]] = [[] for _ in range(n)]
    out_deg = np.zeros(n, dtype=np.int64)
    in_deg = np.zeros(n, dtype=np.int64)
    for (u, v) in pairs:
        in_adj[v].append(u)
        out_deg[u] += 1
        in_deg[v] += 1

    # Kahn's to get a topological order (largest-index-first of DAG sinks)
    topo: List[int] = []
    frontier = [i for i in range(n) if out_deg[i] == 0]
    work_out = out_deg.copy()
    out_adj_list: List[List[int]] = [[] for _ in range(n)]
    for (u, v) in pairs:
        out_adj_list[u].append(v)
    # Process in reverse-topological order (sinks first).
    visited = np.zeros(n, dtype=bool)
    while frontier:
        k = frontier.pop()
        if visited[k]:
            continue
        visited[k] = True
        topo.append(k)
        for u in in_adj[k]:
            work_out[u] -= 1
            if work_out[u] == 0:
                frontier.append(u)
    if len(topo) != n:
        raise ValueError("compute_reach_bitset: DAG has a cycle or unreachable nodes")

    reach = np.zeros(n, dtype=np.uint64)
    one = np.uint64(1)
    # Iterate sinks first; for each v processed, its descendants are already final.
    for v in topo:
        v_bit = one << np.uint64(v)
        for w in out_adj_list[v]:
            reach[v] |= (one << np.uint64(w)) | reach[w]
        # sanity: self bit not set
        # (no-op; we never set reach[v] |= v_bit)
        _ = v_bit
    return reach


# -----------------------------
# Core bitmask routines (Numba)
# -----------------------------
@njit(cache=True)
def _compute_Wadm_Mcount(prefix_mask, exp_node, reach, n):
    """
    Given a prefix set as a uint64 bitmask, compute (W, M):
      W is the sum of exp_node[k] over k in the maximal-element set of the prefix,
      M is the size of that maximal-element set.
    """
    W = 0.0
    cnt = 0
    one = np.uint64(1)
    for k in range(n):
        k_bit = one << np.uint64(k)
        if (prefix_mask & k_bit) != np.uint64(0):
            others = prefix_mask & (~k_bit)
            if (reach[k] & others) == np.uint64(0):
                W += exp_node[k]
                cnt += 1
    return W, cnt


@njit(cache=True)
def _prefix_mask_upto(pi, i0):
    m = np.uint64(0)
    one = np.uint64(1)
    for t in range(i0 + 1):
        m |= one << np.uint64(pi[t])
    return m


# -----------------------------
# Numba fastpath main loop
# -----------------------------
@njit(cache=True)
def _mh_collect_samples_pasv_nb(
    pi,
    pos,
    exp_node,
    reach,
    lazy_prob,
    n_samples,
    burn_in,
    thin,
    seed,
):
    np.random.seed(seed)
    n = pi.shape[0]
    samples = np.empty((n_samples, n), dtype=np.int64)

    Wadm_cache = np.zeros(n - 1, dtype=np.float64)
    Mcount_cache = np.zeros(n - 1, dtype=np.int64)
    has_cache = np.zeros(n - 1, dtype=np.uint8)

    accept = 0
    total_steps = 0
    num_collected = 0
    one = np.uint64(1)

    while num_collected < n_samples:
        total_steps += 1

        if np.random.rand() >= lazy_prob:
            i0 = np.random.randint(0, n - 1)
            a = pi[i0]
            b = pi[i0 + 1]

            a_bit = one << np.uint64(a)
            b_bit = one << np.uint64(b)

            # Skip the swap unless a and b are incomparable in the DAG.
            if ((reach[a] & b_bit) == np.uint64(0)) and ((reach[b] & a_bit) == np.uint64(0)):
                if has_cache[i0] != 0:
                    W_cur = Wadm_cache[i0]
                    M_cur = Mcount_cache[i0]
                else:
                    cur_prefix = _prefix_mask_upto(pi, i0)
                    W_cur, M_cur = _compute_Wadm_Mcount(cur_prefix, exp_node, reach, n)
                    Wadm_cache[i0] = W_cur
                    Mcount_cache[i0] = M_cur
                    has_cache[i0] = 1

                cur_prefix_for_prop = _prefix_mask_upto(pi, i0)
                prop_prefix = (cur_prefix_for_prop & (~a_bit)) | b_bit
                W_prop, M_prop = _compute_Wadm_Mcount(prop_prefix, exp_node, reach, n)

                if W_cur > 0.0 and W_prop > 0.0 and M_cur > 0 and M_prop > 0:
                    log_r = (math.log(W_cur) - math.log(float(M_cur))) - (
                        math.log(W_prop) - math.log(float(M_prop))
                    )
                    u = np.random.rand()
                    lu = math.log(u)
                    threshold = 0.0 if log_r > 0.0 else log_r
                    if lu < threshold:
                        pi[i0], pi[i0 + 1] = b, a
                        pos[a], pos[b] = i0 + 1, i0
                        Wadm_cache[i0] = W_prop
                        Mcount_cache[i0] = M_prop
                        has_cache[i0] = 1
                        accept += 1

        if total_steps >= burn_in and ((total_steps - burn_in) % thin == 0):
            samples[num_collected, :] = pi
            num_collected += 1

    return samples, accept, total_steps


# -----------------------------
# Python reference loop (fallback)
# -----------------------------
def _reference_mh_sampler_pasv(
    n,
    edges,
    lam,
    alpha=1.0,
    seed=0,
    n_samples=1000,
    burn_in=50000,
    thin=200,
    init_pi=None,
    report_every=100,
    lazy_prob=0.5,
):
    n = int(n)
    rng = random.Random(int(seed))
    pairs = _normalize_edge_pairs(edges)
    reach = compute_reach_bitset(n, pairs)
    exp_node = np.exp(-float(alpha) * np.asarray(lam, dtype=np.float64))

    if init_pi is None:
        pi = topological_order(n, pairs, seed=int(seed))
    else:
        pi = np.asarray(init_pi, dtype=np.int64).copy()
    pos = np.empty(n, dtype=np.int64)
    pos[pi] = np.arange(n, dtype=np.int64)

    one = np.uint64(1)

    def compute_Wadm(prefix_mask):
        W = 0.0
        cnt = 0
        for k in range(n):
            k_bit = one << np.uint64(k)
            if (prefix_mask & k_bit) != np.uint64(0):
                others = prefix_mask & (~k_bit)
                if (reach[k] & others) == np.uint64(0):
                    W += float(exp_node[k])
                    cnt += 1
        return W, cnt

    def prefix_mask_upto(i0):
        m = np.uint64(0)
        for t in range(i0 + 1):
            m |= one << np.uint64(pi[t])
        return m

    Wadm_cache = [0.0] * (n - 1)
    Mcount_cache = [0] * (n - 1)
    has_cache = [False] * (n - 1)

    samples: List[Tuple[int, ...]] = []
    accept = 0
    total_steps = 0
    last_reported = 0
    accepted_since_report = 0
    steps_since_report = 0

    while len(samples) < n_samples:
        total_steps += 1
        steps_since_report += 1

        if rng.random() >= lazy_prob:
            i0 = rng.randrange(0, n - 1)
            a = int(pi[i0])
            b = int(pi[i0 + 1])
            a_bit = one << np.uint64(a)
            b_bit = one << np.uint64(b)

            if ((reach[a] & b_bit) == np.uint64(0)) and ((reach[b] & a_bit) == np.uint64(0)):
                if has_cache[i0]:
                    W_cur = Wadm_cache[i0]
                    M_cur = Mcount_cache[i0]
                else:
                    W_cur, M_cur = compute_Wadm(prefix_mask_upto(i0))
                    Wadm_cache[i0] = W_cur
                    Mcount_cache[i0] = M_cur
                    has_cache[i0] = True

                cur_prefix_for_prop = prefix_mask_upto(i0)
                prop_prefix = (cur_prefix_for_prop & (~a_bit)) | b_bit
                W_prop, M_prop = compute_Wadm(prop_prefix)

                if W_cur > 0.0 and W_prop > 0.0 and M_cur > 0 and M_prop > 0:
                    log_r = (math.log(W_cur) - math.log(float(M_cur))) - (
                        math.log(W_prop) - math.log(float(M_prop))
                    )
                    if math.log(rng.random()) < (0.0 if log_r > 0.0 else log_r):
                        pi[i0], pi[i0 + 1] = b, a
                        pos[a], pos[b] = i0 + 1, i0
                        Wadm_cache[i0] = W_prop
                        Mcount_cache[i0] = M_prop
                        has_cache[i0] = True
                        accept += 1
                        accepted_since_report += 1

        if total_steps >= burn_in and ((total_steps - burn_in) % thin == 0):
            samples.append(tuple(int(x) for x in pi))
            if report_every and report_every > 0 and len(samples) - last_reported >= report_every:
                new_samples = len(samples) - last_reported
                print(
                    "num_samples", len(samples),
                    "new_samples", new_samples,
                    "accept_rate_since_report", round(accepted_since_report / max(1, steps_since_report), 4),
                    "overall_accept_rate", round(accept / max(1, total_steps), 4),
                    "total_steps", total_steps,
                )
                last_reported = len(samples)
                accepted_since_report = 0
                steps_since_report = 0

    samples_arr = np.asarray(samples, dtype=np.int64)
    return samples_arr, float(accept) / float(max(total_steps, 1))


# -----------------------------
# Topological-order initializer (Kahn)
# -----------------------------
def topological_order(n: int, edges, seed: int = 0) -> np.ndarray:
    """
    Return any topological ordering of the DAG as a length-n int64 array.
    Tie-breaking uses a seeded RNG for reproducibility.
    """
    n = int(n)
    pairs = _normalize_edge_pairs(edges)
    in_deg = np.zeros(n, dtype=np.int64)
    out_adj: List[List[int]] = [[] for _ in range(n)]
    for (u, v) in pairs:
        out_adj[u].append(v)
        in_deg[v] += 1

    rng = random.Random(int(seed))
    frontier = [i for i in range(n) if in_deg[i] == 0]
    rng.shuffle(frontier)

    order: List[int] = []
    while frontier:
        idx = rng.randrange(0, len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        node = frontier.pop()
        order.append(int(node))
        for v in out_adj[node]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                frontier.append(int(v))

    if len(order) != n:
        raise ValueError("topological_order: DAG has a cycle")
    return np.asarray(order, dtype=np.int64)


# -----------------------------
# Public sampler entry point
# -----------------------------
def mh_sampler_pasv(
    n,
    edges,
    lam,
    alpha=1.0,
    seed=0,
    n_samples=1000,
    burn_in=50000,
    thin=200,
    init_pi=None,
    report_every=100,
    lazy_prob=0.5,
):
    """Adjacent-swap MH for PASV on the DAG induced by `edges`.

    Mirrors mh_sampler() but the DAG's hard precedence replaces the GPASV beta
    temperature. Edge weights are ignored. Returns (samples, acceptance_rate).
    """
    n = int(n)
    pairs = _normalize_edge_pairs(edges)

    if not NUMBA_AVAILABLE:
        return _reference_mh_sampler_pasv(
            n=n, edges=pairs, lam=lam, alpha=alpha, seed=seed,
            n_samples=n_samples, burn_in=burn_in, thin=thin, init_pi=init_pi,
            report_every=report_every, lazy_prob=lazy_prob,
        )

    reach = compute_reach_bitset(n, pairs)
    exp_node = np.exp(-float(alpha) * np.asarray(lam, dtype=np.float64))

    if init_pi is None:
        pi = topological_order(n, pairs, seed=int(seed))
    else:
        pi = np.asarray(init_pi, dtype=np.int64).copy()
    pos = np.empty(n, dtype=np.int64)
    pos[pi] = np.arange(n, dtype=np.int64)

    n_samples = int(n_samples)
    burn_in = int(burn_in)
    thin = int(thin)
    seed = int(seed)
    report_every = int(report_every) if report_every is not None else 0

    if report_every <= 0:
        samples, accept, total_steps = _mh_collect_samples_pasv_nb(
            pi=pi, pos=pos, exp_node=exp_node, reach=reach,
            lazy_prob=float(lazy_prob), n_samples=n_samples,
            burn_in=burn_in, thin=thin, seed=seed,
        )
        return samples, float(accept) / float(max(total_steps, 1))

    chunks = []
    total_accept = 0
    total_steps = 0
    collected = 0
    chunk_id = 0
    current_burn_in = burn_in
    while collected < n_samples:
        chunk_id += 1
        chunk_size = min(report_every, n_samples - collected)
        chunk_seed = seed + chunk_id - 1
        chunk_samples, chunk_accept, chunk_steps = _mh_collect_samples_pasv_nb(
            pi=pi, pos=pos, exp_node=exp_node, reach=reach,
            lazy_prob=float(lazy_prob), n_samples=chunk_size,
            burn_in=current_burn_in, thin=thin, seed=chunk_seed,
        )
        chunks.append(chunk_samples)
        collected += chunk_size
        total_accept += int(chunk_accept)
        total_steps += int(chunk_steps)
        print(
            "num_samples", collected,
            "new_samples", chunk_size,
            "accept_rate_since_report", round(float(chunk_accept) / max(1, int(chunk_steps)), 4),
            "overall_accept_rate", round(float(total_accept) / max(1, int(total_steps)), 4),
            "total_steps", total_steps,
        )
        current_burn_in = 0

    samples = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
    return samples, float(total_accept) / float(max(total_steps, 1))


def draw_independent_final_permutations_pasv(
    *,
    n,
    edges,
    lam,
    steps,
    num_chains,
    chain_seeds,
    init_perms: Optional[np.ndarray] = None,
    alpha=1.0,
    lazy_prob=0.5,
):
    """
    Analogue of sampler.draw_independent_final_permutations for PASV. Runs
    `num_chains` independent MH chains for `steps` MH moves each and returns
    the final permutation from each.
    """
    n = int(n)
    steps = int(steps)
    num_chains = int(num_chains)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    seeds = np.asarray(chain_seeds, dtype=np.int64)
    if seeds.shape[0] != num_chains:
        raise ValueError(
            f"chain_seeds length mismatch: expected {num_chains}, got {seeds.shape[0]}"
        )

    if init_perms is not None:
        init_perms = np.asarray(init_perms, dtype=np.int64)
        if init_perms.shape != (num_chains, n):
            raise ValueError(
                f"init_perms shape mismatch: expected {(num_chains, n)}, got {tuple(init_perms.shape)}"
            )

    out = np.empty((num_chains, n), dtype=np.int64)
    acc_rates = np.empty(num_chains, dtype=np.float64)

    for idx in range(num_chains):
        init_pi = None if init_perms is None else init_perms[idx, :]
        samples, acceptance_rate = mh_sampler_pasv(
            n=n, edges=edges, lam=lam, alpha=float(alpha),
            seed=int(seeds[idx]), n_samples=1, burn_in=steps, thin=1,
            init_pi=init_pi, report_every=0, lazy_prob=float(lazy_prob),
        )
        out[idx, :] = np.asarray(samples[0], dtype=np.int64)
        acc_rates[idx] = float(acceptance_rate)

    return out, float(np.mean(acc_rates))
