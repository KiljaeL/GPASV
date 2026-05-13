import math
import random
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