from __future__ import annotations

import argparse
import json
import pickle
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import HF_TOKEN
from graph import MODELS, build_edges, build_pairwise_matrices, run_ranking
from load_data import load_chatbot_arena_df, load_mtbench_single_df, prepare_benchmark_data
import sampler as sampler_module
from sampler import _logsumexp, build_adjacency, log_unnormalized_pmf


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="qwen3.5-35b-fp8",
    choices=["qwen3-30b", "qwen3-30b-fp8", "qwen3.5-35b", "qwen3.5-35b-fp8", "qwen3.5-9b", "qwen3.5-4b"],
    help="Metadata only (no model load).",
)
parser.add_argument(
    "--sweep-type",
    type=str,
    default="exp",
    choices=["linear", "exp"],
    help="Same as main_subset.py.",
)
parser.add_argument("--sweep-max", type=int, default=32)
parser.add_argument("--sweep-step", type=int, default=1, help="Linear sweeps only.")
parser.add_argument(
    "--n-samples",
    type=int,
    default=1000,
    help="Permutations per param pair in Phase 1; target MC count when no ESS reuse.",
)
parser.add_argument(
    "--n-min",
    type=int,
    default=500,
    help="Same as main_subset: n_new = max(N_min, ceil(n_samples - ESS)), capped at n_samples.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--question",
    type=str,
    default="all",
    metavar="SPEC",
    help='1-80 numbering: "1,5,15" or "all".',
)
parser.add_argument(
    "--use-numba",
    type=str2bool,
    default=True,
    help="Sampler numba path (same as main_subset).",
)
parser.add_argument(
    "--no-ess-reuse",
    action="store_true",
    help="Disable ESS/SNIS reuse (full n_samples MC per pair, matching main_subset --no-ess-reuse).",
)
parser.add_argument(
    "--sweep-output",
    type=str,
    default="all",
    choices=["all", "alpha", "beta", "both"],
    help="Which sweeps to compute values for (default: all three, like subset enumeration order).",
)
parser.add_argument(
    "--cache-style",
    type=str,
    default="single",
    choices=["single", "shard"],
    help="single: cache/subset_cache_{suffix}.db; shard: subset_cache_shard_* (main_value.py).",
)
parser.add_argument("--n-shards", type=int, default=1, help="Only for --cache-style shard.")
args = parser.parse_args()


def parse_question_spec(spec):
    s = (spec or "").strip()
    if not s or s.lower() == "all":
        return list(range(80))

    def to_index(n):
        if not 1 <= n <= 80:
            raise ValueError(f"Question number must be in 1–80; got {n}")
        return n - 1

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"Empty question list in {spec!r}")
        return sorted(set(to_index(int(p)) for p in parts))
    return [to_index(int(s))]


try:
    EVAL_QS = parse_question_spec(args.question)
except ValueError as e:
    parser.error(str(e))


def cache_question_suffix(eval_qs):
    if len(eval_qs) == 80 and eval_qs == list(range(80)):
        return "all"
    if len(eval_qs) == 1:
        return str(eval_qs[0] + 1)
    return "-".join(str(i + 1) for i in eval_qs)


def mask_to_subset(mask, n):
    return np.array([i for i in range(n) if (mask >> i) & 1], dtype=np.int64)


def build_lam_and_out_adj(df_pairwise):
    W, T, models, _ = build_pairwise_matrices(df_pairwise)
    edges = build_edges(W, T, models)
    out_adj, _ = build_adjacency(len(models), edges)
    lam = [
        0 if m in ["gpt-4", "gpt-3.5-turbo", "claude-v1", "claude-instant-v1", "palm-2"] else 1
        for m in models
    ]
    return lam, out_adj


def log_ess_from_log_weights(log_weights):
    if not log_weights:
        return 0.0
    lse_1 = _logsumexp(log_weights)
    lse_2 = _logsumexp([2.0 * lw for lw in log_weights])
    if lse_1 == -np.inf or lse_2 == -np.inf:
        return 0.0
    return float(np.exp(2.0 * lse_1 - lse_2))


def _build_shard_paths(n_shards, q_suffix):
    return [Path("cache") / f"subset_cache_shard_{shard_id}_{n_shards}_{q_suffix}.db" for shard_id in range(1, n_shards + 1)]


def build_cache_paths_single(eval_qs):
    """
    Prefer one DB: cache/subset_cache_{combined_suffix}.db (same --question list as subset run).

    If that file is missing and multiple questions were requested, fall back to the union of
    per-question DBs: cache/subset_cache_{q}.db for each 1-based q (main_subset one job per question).
    """
    combined_suffix = cache_question_suffix(eval_qs)
    p = Path("cache") / f"subset_cache_{combined_suffix}.db"
    if p.exists():
        return [p], f"single:{combined_suffix}"

    if len(eval_qs) == 1:
        raise FileNotFoundError(
            f"Missing cache DB (main_subset style): {p}\n"
            f"Run main_subset.py with matching --question, or use per-question caches with multiple questions."
        )

    paths = []
    missing = []
    for q in eval_qs:
        one_suffix = cache_question_suffix([q])
        pq = Path("cache") / f"subset_cache_{one_suffix}.db"
        if pq.exists():
            paths.append(pq)
        else:
            missing.append(str(pq))

    if missing:
        raise FileNotFoundError(
            "Missing per-question cache DB(s) (expected one DB per question):\n  "
            + "\n  ".join(missing)
            + f"\n\nAlternatively create a combined DB at: {p}"
        )

    unique_paths = list(dict.fromkeys(paths))
    return unique_paths, f"single_per_question:{combined_suffix}"


def build_cache_paths_shard(n_shards, eval_qs):
    """Same combined / per-question fallback as main_value.py."""
    combined_suffix = cache_question_suffix(eval_qs)
    combined_paths = _build_shard_paths(n_shards, combined_suffix)
    if all(path.exists() for path in combined_paths):
        return combined_paths, f"shard_combined:{combined_suffix}"

    if len(eval_qs) == 1:
        missing = [str(path) for path in combined_paths if not path.exists()]
        raise FileNotFoundError(f"Missing cache shard(s): {missing}")

    paths = []
    missing = []
    for q in eval_qs:
        single_suffix = cache_question_suffix([q])
        single_paths = _build_shard_paths(n_shards, single_suffix)
        for path in single_paths:
            if path.exists():
                paths.append(path)
            else:
                missing.append(str(path))

    if missing:
        raise FileNotFoundError(f"Missing cache shard(s) for per-question fallback: {missing}")

    unique_paths = list(dict.fromkeys(paths))
    return unique_paths, "shard_per_question"


def open_cache_connections(paths):
    return [sqlite3.connect(str(path)) for path in paths]


def close_cache_connections(conns):
    for conn in conns:
        conn.close()


def batched(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def load_scores_for_masks(conns, masks, eval_qs):
    scores_by_mask = {}
    mask_list = sorted(int(m) for m in masks)
    if not mask_list:
        return scores_by_mask

    q_list = sorted(int(q) for q in eval_qs)
    q_placeholders = ",".join("?" for _ in q_list)

    for conn in conns:
        for mask_chunk in batched(mask_list, 400):
            mask_placeholders = ",".join("?" for _ in mask_chunk)
            query = (
                "SELECT subset_mask, question_index, judge_turn1_rating, judge_turn2_rating "
                f"FROM cache_entries WHERE question_index IN ({q_placeholders}) "
                f"AND subset_mask IN ({mask_placeholders})"
            )
            params = tuple(q_list) + tuple(mask_chunk)
            for subset_mask, question_index, j1, j2 in conn.execute(query, params):
                scores_by_mask.setdefault(int(subset_mask), {})[int(question_index)] = (int(j1), int(j2))
    return scores_by_mask


def utility_from_cache(subset_mask, scores_by_mask, eval_qs, n):
    rows = scores_by_mask.get(int(subset_mask))
    if rows is None:
        subset = list(map(int, mask_to_subset(subset_mask, n)))
        raise KeyError(f"Missing cache entries for subset_mask={int(subset_mask)}, subset={subset}")
    total = 0.0
    for q in eval_qs:
        cached = rows.get(int(q))
        if cached is None:
            subset = list(map(int, mask_to_subset(subset_mask, n)))
            raise KeyError(
                f"Missing cache entry for subset_mask={int(subset_mask)}, subset={subset}, question={int(q) + 1}"
            )
        total += cached[0] + cached[1]
    return total / (2 * len(eval_qs))


def collect_needed_masks_from_perms(perms, n_players):
    needed = set()
    for perm in perms:
        mask = 0
        for player in perm:
            mask |= 1 << int(player)
            needed.add(mask)
    return needed


def permutation_marginal_vector(perm, scores_by_mask, eval_qs, n_players, utility_cache):
    """Unnormalized per-permutation Shapley path contributions (one vector in R^n)."""
    contrib = np.zeros(n_players, dtype=np.float64)
    u_old = 0.0
    mask = 0
    for player in perm:
        mask |= 1 << int(player)
        if mask not in utility_cache:
            utility_cache[mask] = utility_from_cache(mask, scores_by_mask, eval_qs, n_players)
        u_cur = utility_cache[mask]
        contrib[int(player)] += u_cur - u_old
        u_old = u_cur
    return contrib


def estimate_value_mc(perms, scores_by_mask, eval_qs, n_players):
    utility_cache = {}
    acc = np.zeros(n_players, dtype=np.float64)
    m = len(perms)
    if m == 0:
        return acc, utility_cache
    for perm in perms:
        acc += permutation_marginal_vector(perm, scores_by_mask, eval_qs, n_players, utility_cache)
    acc /= float(m)
    return acc, utility_cache


def running_mc_phi_rows(perms, scores_by_mask, eval_qs, n_players):
    """
    After each permutation k, phi_k = (1/m) * sum_{i=1}^k contrib_i with m = len(perms).
    Matches main_value.py: each marginal increment is scaled by 1/m as it is accumulated.
    """
    utility_cache = {}
    phi = np.zeros(n_players, dtype=np.float64)
    rows = []
    m = len(perms)
    if m == 0:
        return rows, phi
    for perm in perms:
        c = permutation_marginal_vector(perm, scores_by_mask, eval_qs, n_players, utility_cache)
        phi = phi + c / float(m)
        rows.append(phi.copy())
    return rows, phi


def running_snis_phi_rows(perms, log_weights_unnorm, scores_by_mask, eval_qs, n_players):
    """After k samples, SNIS estimate using first k permutations and renormalized weights."""
    utility_cache = {}
    m = len(perms)
    if m == 0:
        return [], np.zeros(n_players, dtype=np.float64)
    contribs = []
    for perm in perms:
        contribs.append(
            permutation_marginal_vector(perm, scores_by_mask, eval_qs, n_players, utility_cache)
        )
    contribs_arr = np.stack(contribs, axis=0)
    rows = []
    for k in range(1, m + 1):
        lw = np.array(log_weights_unnorm[:k], dtype=np.float64)
        lse = _logsumexp(lw.tolist())
        if lse == -np.inf or not np.isfinite(lse):
            w = np.ones(k) / k
        else:
            w = np.exp(lw - lse)
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                w = np.ones(k) / k
            else:
                w = w / s
        phi_k = w @ contribs_arr[:k]
        rows.append(phi_k)
    return rows, rows[-1]


def estimate_value_snis(perms, log_weights_unnorm, scores_by_mask, eval_qs, n_players):
    """SNIS: sum_i w_i * contrib_i with w_i proportional to exp(log_weights_unnorm[i]), normalized."""
    utility_cache = {}
    if not perms:
        return np.zeros(n_players, dtype=np.float64), np.zeros(n_players, dtype=np.float64), 0.0

    lw = np.array(log_weights_unnorm, dtype=np.float64)
    lse = _logsumexp(lw.tolist())
    if lse == -np.inf or not np.isfinite(lse):
        w = np.ones(len(perms)) / len(perms)
    else:
        w = np.exp(lw - lse)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            w = np.ones(len(perms)) / len(perms)
        else:
            w = w / s

    acc = np.zeros(n_players, dtype=np.float64)
    for perm, wi in zip(perms, w):
        acc += wi * permutation_marginal_vector(perm, scores_by_mask, eval_qs, n_players, utility_cache)

    ess = float(1.0 / np.sum(w * w)) if len(w) else 0.0
    return acc, w, ess


def format_param_tag(alpha, beta):
    return f"a{alpha:g}_b{beta:g}".replace("/", "_")


def main():
    if args.use_numba:
        if not sampler_module.NUMBA_AVAILABLE:
            print("ERROR: --use-numba is True, but numba is not available.", flush=True)
            sys.exit(1)
        sampler_module.NUMBA_AVAILABLE = True
        print("Sampler mode: numba enabled", flush=True)
    else:
        sampler_module.NUMBA_AVAILABLE = False
        print("Sampler mode: pure Python fallback", flush=True)

    seed = args.seed
    np.random.seed(seed)
    rng = random.Random(seed)
    print(f"Seed fixed: {seed}", flush=True)

    q_1based = sorted(i + 1 for i in EVAL_QS)
    print(f"Questions: {len(EVAL_QS)} (1-based: {q_1based})", flush=True)

    use_ess_reuse = not args.no_ess_reuse
    print(f"ESS/SNIS reuse: {use_ess_reuse}", flush=True)

    sweep_max = args.sweep_max
    sweep_type = args.sweep_type
    sweep_step = getattr(args, "sweep_step", 1) if sweep_type == "linear" else None
    n_samples = args.n_samples
    N_min = args.n_min
    n_players = len(MODELS)

    if sweep_type == "exp":

        def build_grid(max_value):
            vals = []
            x = 1
            while x <= max_value:
                vals.append(x)
                x *= 2
            return vals

        sweep_vals = build_grid(sweep_max)
        print(f"Sweep type: exp, max={sweep_max} -> vals {sweep_vals}", flush=True)
    else:
        step = max(1, int(sweep_step))
        sweep_vals = list(range(step, sweep_max + 1, step))
        if sweep_vals and sweep_vals[-1] != sweep_max:
            sweep_vals.append(sweep_max)
        print(f"Sweep type: linear, max={sweep_max}, step={step} -> vals {sweep_vals}", flush=True)

    alpha_sweep = [(0, 0)] + [(k, 0) for k in sweep_vals]
    beta_sweep = [(0, 0)] + [(0, k) for k in sweep_vals]
    both_sweep = [(0, 0)] + [(k, k) for k in sweep_vals]

    unique_pairs = [(0, 0)]
    for k in sweep_vals:
        unique_pairs.append((k, 0))
    for k in sweep_vals:
        unique_pairs.append((0, k))
    for k in sweep_vals:
        unique_pairs.append((k, k))
    unique_pairs = list(dict.fromkeys(unique_pairs))

    df_benchmark = load_mtbench_single_df()
    df_pairwise = load_chatbot_arena_df(HF_TOKEN)
    df_benchmark, df_pairwise = prepare_benchmark_data(df_benchmark, df_pairwise)
    lam, out_adj = build_lam_and_out_adj(df_pairwise)
    print(f"Prepared graph with {n_players} players", flush=True)

    # ---------- Phase 1: same as main_subset ----------
    print("Phase 1: sampling permutations for each unique (alpha, beta)...", flush=True)
    samples_by_pair = {}
    for i, (alpha, beta) in enumerate(unique_pairs):
        print(f"  Sampling {i+1}/{len(unique_pairs)}: alpha={alpha}, beta={beta}", flush=True)
        samples, df_rank = run_ranking(
            df_pairwise,
            n_samples=n_samples,
            burn_in=100000,
            thin=1000,
            seed=seed + i,
            report_every=1000,
            alpha=alpha,
            beta=beta,
        )
        samples_by_pair[(alpha, beta)] = samples

    kept_0_0 = list(samples_by_pair[(0, 0)])
    rng.shuffle(kept_0_0)
    kept_0_0 = kept_0_0[:n_samples]
    print(f"(0,0) shared pool: {len(kept_0_0)} permutations", flush=True)

    pickle_suffix = "_snis" if use_ess_reuse else ""

    sweeps_to_run = []
    if args.sweep_output in ("all", "alpha"):
        sweeps_to_run.append(("alpha", alpha_sweep))
    if args.sweep_output in ("all", "beta"):
        sweeps_to_run.append(("beta", beta_sweep))
    if args.sweep_output in ("all", "both"):
        sweeps_to_run.append(("both", both_sweep))

    save_root = Path("save") / "value"
    save_root.mkdir(parents=True, exist_ok=True)

    samples_path = save_root / "samples.pkl"
    samples_payload = {
        "version": 1,
        "samples_by_pair": samples_by_pair,
        "kept_0_0_shared": kept_0_0,
        "unique_pairs": unique_pairs,
        "n_samples": n_samples,
        "n_min": N_min,
        "seed": seed,
        "model_key": args.model,
        "questions_1based": q_1based,
        "sweep_type": sweep_type,
        "sweep_max": sweep_max,
        "sweep_step": int(sweep_step) if sweep_type == "linear" else None,
        "sweep_vals": list(sweep_vals),
        "alpha_sweep": alpha_sweep,
        "beta_sweep": beta_sweep,
        "both_sweep": both_sweep,
        "sweep_output": args.sweep_output,
    }
    with open(samples_path, "wb") as f:
        pickle.dump(samples_payload, f)
    print(f"Saved Phase-1 samples: {samples_path}", flush=True)

    print(
        f"Output root: {save_root}/<question>/  "
        f"({'*_snis.pkl' if use_ess_reuse else '*.pkl, no _snis infix'})",
        flush=True,
    )

    for qi, q_idx in enumerate(EVAL_QS):
        q_one_based = q_idx + 1
        eval_qs_cur = [q_idx]
        print(
            f"\n=== Question {qi + 1}/{len(EVAL_QS)} (1-based index {q_one_based}) ===",
            flush=True,
        )

        if args.cache_style == "single":
            cache_paths, cache_mode = build_cache_paths_single(eval_qs_cur)
        else:
            if args.n_shards < 1:
                parser.error("--n-shards must be >= 1 for --cache-style shard")
            cache_paths, cache_mode = build_cache_paths_shard(args.n_shards, eval_qs_cur)

        print(f"  Cache mode: {cache_mode}", flush=True)
        print(f"  Cache DB(s): {[str(p) for p in cache_paths]}", flush=True)
        cache_conns = open_cache_connections(cache_paths)

        save_dir = save_root / str(q_one_based)
        save_dir.mkdir(parents=True, exist_ok=True)

        def run_one_sweep(sweep_name, sweep_list):
            kept_by_pair = {}
            meta_by_pair = {}

            def save_value_pick(df_value: pd.DataFrame, p_pair, meta):
                out = df_value.copy()
                out.attrs["sweep"] = sweep_name
                out.attrs["alpha"] = float(p_pair[0])
                out.attrs["beta"] = float(p_pair[1])
                out.attrs["seed"] = seed
                out.attrs["n_samples_config"] = n_samples
                out.attrs["n_min"] = N_min
                out.attrs["ess_reuse"] = use_ess_reuse
                out.attrs["questions_1based"] = [q_one_based]
                out.attrs["model_key"] = args.model
                out.attrs["cache_paths"] = [str(p) for p in cache_paths]
                out.attrs["cache_mode"] = cache_mode
                out.attrs["pickle_suffix"] = pickle_suffix
                out.attrs["meta"] = json.dumps(meta, default=str)
                base = f"{sweep_name}_{format_param_tag(p_pair[0], p_pair[1])}{pickle_suffix}.pkl"
                out_path = save_dir / base
                out.to_pickle(out_path)
                print(f"    Saved {out_path} ({len(out)} trajectory rows)", flush=True)

            p0 = sweep_list[0]
            kept_by_pair[p0] = kept_0_0

            needed0 = collect_needed_masks_from_perms(kept_0_0, n_players)
            scores0 = load_scores_for_masks(cache_conns, needed0, eval_qs_cur)
            rows0, phi0 = running_mc_phi_rows(kept_0_0, scores0, eval_qs_cur, n_players)
            df0 = pd.DataFrame(rows0, columns=MODELS)
            df0.index = pd.RangeIndex(start=1, stop=len(rows0) + 1, step=1, name="permutations_processed")
            meta_by_pair[p0] = {
                "mode": "mc_shared_00",
                "n_perms": len(kept_0_0),
                "ess_from_weights": None,
                "n_new": None,
                "w_snis": None,
                "w_fresh": None,
                "trajectory_rows": len(rows0),
                "trajectory_kind": "mc",
            }
            print(f"  [{sweep_name}] {p0}: MC on shared (0,0), n={len(kept_0_0)}", flush=True)
            save_value_pick(df0, p0, meta_by_pair[p0])

            for i in range(1, len(sweep_list)):
                p_pair = sweep_list[i]
                q_pair = sweep_list[i - 1]
                q_samples = kept_by_pair[q_pair]

                if use_ess_reuse:
                    log_q = [log_unnormalized_pmf(pi, lam, q_pair[0], q_pair[1], out_adj) for pi in q_samples]
                    log_weights = []
                    for pi, lq in zip(q_samples, log_q):
                        lp = log_unnormalized_pmf(pi, lam, p_pair[0], p_pair[1], out_adj)
                        log_weights.append(lp - lq)
                    ess = log_ess_from_log_weights(log_weights)
                    n_new = int(np.ceil(n_samples - ess))
                    n_new = max(N_min, n_new)
                    n_new = min(n_samples, n_new)

                    raw = list(samples_by_pair[p_pair])
                    rng.shuffle(raw)
                    fresh = raw[:n_new]
                    kept_by_pair[p_pair] = fresh

                    needed_masks = collect_needed_masks_from_perms(q_samples, n_players)
                    needed_masks |= collect_needed_masks_from_perms(fresh, n_players)
                    scores_by_mask = load_scores_for_masks(cache_conns, needed_masks, eval_qs_cur)

                    phi_snis, _, ess_w = estimate_value_snis(
                        q_samples, log_weights, scores_by_mask, eval_qs_cur, n_players
                    )
                    phi_fresh, _ = estimate_value_mc(fresh, scores_by_mask, eval_qs_cur, n_players)

                    denom = ess + float(n_new)
                    if denom > 0:
                        w_snis = ess / denom
                        w_fresh = float(n_new) / denom
                    else:
                        w_snis, w_fresh = 0.0, 1.0

                    rows_snis, phi_snis_tr = running_snis_phi_rows(
                        q_samples, log_weights, scores_by_mask, eval_qs_cur, n_players
                    )
                    rows_fresh, phi_fresh_tr = running_mc_phi_rows(
                        fresh, scores_by_mask, eval_qs_cur, n_players
                    )
                    if not np.allclose(phi_snis_tr, phi_snis, rtol=1e-9, atol=1e-9):
                        raise RuntimeError("SNIS trajectory final mismatch")
                    if not np.allclose(phi_fresh_tr, phi_fresh, rtol=1e-9, atol=1e-9):
                        raise RuntimeError("Fresh MC trajectory final mismatch")

                    phi_snis_final = rows_snis[-1]
                    combined_rows = list(rows_snis)
                    for row_f in rows_fresh:
                        combined_rows.append(w_snis * phi_snis_final + w_fresh * row_f)
                    phi = combined_rows[-1]
                    if not np.allclose(phi, w_snis * phi_snis + w_fresh * phi_fresh, rtol=1e-9, atol=1e-9):
                        raise RuntimeError("Blended value mismatch")

                    df_p = pd.DataFrame(combined_rows, columns=MODELS)
                    df_p.index = pd.RangeIndex(
                        start=1, stop=len(combined_rows) + 1, step=1, name="trajectory_step"
                    )

                    meta_by_pair[p_pair] = {
                        "mode": "snis_mc_blend",
                        "n_q": len(q_samples),
                        "n_new": n_new,
                        "ess_logsumexp": ess,
                        "ess_normalized_weights": ess_w,
                        "w_snis": w_snis,
                        "w_fresh": w_fresh,
                        "q_pair": q_pair,
                        "trajectory_rows": len(combined_rows),
                        "trajectory_kind": "snis_prefix_then_blend",
                        "trajectory_n_snis": len(rows_snis),
                        "trajectory_n_blend": len(rows_fresh),
                    }
                    print(
                        f"  [{sweep_name}] {q_pair} -> {p_pair}: ESS(logw)={ess:.2f}, n_new={n_new}, "
                        f"w_snis={w_snis:.4f}, w_fresh={w_fresh:.4f}",
                        flush=True,
                    )
                    save_value_pick(df_p, p_pair, meta_by_pair[p_pair])
                else:
                    raw = list(samples_by_pair[p_pair])
                    rng.shuffle(raw)
                    kept = raw[:n_samples]
                    kept_by_pair[p_pair] = kept
                    needed_masks = collect_needed_masks_from_perms(kept, n_players)
                    scores_by_mask = load_scores_for_masks(cache_conns, needed_masks, eval_qs_cur)
                    rows_k, phi = running_mc_phi_rows(kept, scores_by_mask, eval_qs_cur, n_players)
                    df_p = pd.DataFrame(rows_k, columns=MODELS)
                    df_p.index = pd.RangeIndex(
                        start=1, stop=len(rows_k) + 1, step=1, name="permutations_processed"
                    )
                    meta_by_pair[p_pair] = {
                        "mode": "mc_full",
                        "n_perms": len(kept),
                        "ess_from_weights": None,
                        "n_new": None,
                        "w_snis": None,
                        "w_fresh": None,
                        "trajectory_rows": len(rows_k),
                        "trajectory_kind": "mc",
                    }
                    print(f"  [{sweep_name}] {p_pair}: MC n={len(kept)} (no ESS reuse)", flush=True)
                    save_value_pick(df_p, p_pair, meta_by_pair[p_pair])

        for sweep_name, sweep_list in sweeps_to_run:
            print(f"=== Sweep: {sweep_name} ({len(sweep_list)} points) ===", flush=True)
            run_one_sweep(sweep_name, sweep_list)

        close_cache_connections(cache_conns)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
