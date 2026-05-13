import argparse
import random
import sqlite3
import sys
import time

import numpy as np
import torch

from config import HF_TOKEN, MAX_NEW_TOKENS, MODEL_CONFIGS
from graph import MODELS, build_edges, build_pairwise_matrices, run_ranking
from load_data import load_chatbot_arena_df, load_mtbench_single_df, prepare_benchmark_data, prepare_qa_data
from inference import load_model_and_tokenizer, generate_batch
from prompt import (
    aggregate,
    judge_wo_ref,
    judge_with_ref,
    response_to_rating,
    build_aggregate_prompt,
    build_judge_wo_ref_prompt,
    build_judge_with_ref_prompt,
)
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
)
parser.add_argument(
    "--sweep-type",
    type=str,
    default="exp",
    choices=["linear", "exp"],
    help="Sweep style: linear (use step and max) or exp (power-of-two up to max, e.g. 1,2,4,8,16,32).",
)
parser.add_argument(
    "--sweep-max",
    type=int,
    default=32,
    help="Max value for sweep. Linear: points at step,2*step,...,sweep_max. Exp: 1,2,4,...,sweep_max.",
)
parser.add_argument(
    "--sweep-step",
    type=int,
    default=1,
    help="Linear only: step size. E.g. max=10 step=1 -> 0,1,...,10; max=10 step=2 -> 0,2,4,6,8,10.",
)
parser.add_argument("--n-samples", type=int, default=1000, help="Number of permutations to draw per param pair (before ESS thinning)")
parser.add_argument(
    "--n-min",
    type=int,
    default=500,
    help="Minimum number of fresh samples per step: n_new = max(N_min, ceil(n_samples - ESS)), then capped at n_samples.",
)
parser.add_argument("--shard", type=int, default=1, help="Shard index (1-indexed)")
parser.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
parser.add_argument(
    "--perm-chunk-size",
    type=int,
    default=100,
    help="Group permutations into chunks for subset ordering.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="vLLM: prompts per generate_batch call; also max uncached tasks per wave (4 turns then DB write).",
)
parser.add_argument(
    "--commit-batch-size",
    type=int,
    default=1000,
    help="After each wave (all 4 turns), upsert+commit SQLite in chunks of this many rows (smaller = more frequent durable commits).",
)
parser.add_argument(
    "--question",
    type=str,
    default="all",
    metavar="SPEC",
    help='Which questions: 1-80 numbering, e.g. "1,5,15" or "all".',
)
parser.add_argument(
    "--use-numba",
    type=str2bool,
    default=True,
    help="Use numba fast path in sampler (true/false). If true but numba is unavailable, exits with error.",
)
parser.add_argument(
    "--no-ess-reuse",
    action="store_true",
    help="Disable ESS-based reuse: each param pair keeps full n_samples (no thinning). Use to compare unique subset count without reuse.",
)
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


model_key = args.model


def subset_to_mask(S):
    mask = 0
    for x in S:
        mask |= 1 << int(x)
    return mask


def mask_to_subset(mask, n):
    return np.array([i for i in range(n) if (mask >> i) & 1], dtype=np.int64)


def init_cache_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache_entries (
            subset_mask INTEGER NOT NULL,
            question_index INTEGER NOT NULL,
            aggregate_turn1_response TEXT,
            aggregate_turn2_response TEXT,
            judge_turn1_response TEXT,
            judge_turn2_response TEXT,
            judge_turn1_rating INTEGER,
            judge_turn2_rating INTEGER,
            PRIMARY KEY (subset_mask, question_index)
        )
    """)
    conn.commit()
    return conn


def get_cached_entry(conn, subset_mask, question_index):
    row = conn.execute("""
        SELECT aggregate_turn1_response, aggregate_turn2_response,
               judge_turn1_response, judge_turn2_response,
               judge_turn1_rating, judge_turn2_rating
        FROM cache_entries
        WHERE subset_mask = ? AND question_index = ?
    """, (int(subset_mask), int(question_index))).fetchone()
    if row is None:
        return None
    return {
        "aggregate_turn1_response": row[0],
        "aggregate_turn2_response": row[1],
        "judge_turn1_response": row[2],
        "judge_turn2_response": row[3],
        "judge_turn1_rating": row[4],
        "judge_turn2_rating": row[5],
    }


def upsert_cached_entry(conn, subset_mask, question_index, a1, a2, j1r, j2r, u1, u2):
    conn.execute("""
        INSERT INTO cache_entries (
            subset_mask, question_index,
            aggregate_turn1_response, aggregate_turn2_response,
            judge_turn1_response, judge_turn2_response,
            judge_turn1_rating, judge_turn2_rating
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subset_mask, question_index) DO UPDATE SET
            aggregate_turn1_response = excluded.aggregate_turn1_response,
            aggregate_turn2_response = excluded.aggregate_turn2_response,
            judge_turn1_response = excluded.judge_turn1_response,
            judge_turn2_response = excluded.judge_turn2_response,
            judge_turn1_rating = excluded.judge_turn1_rating,
            judge_turn2_rating = excluded.judge_turn2_rating
    """, (int(subset_mask), int(question_index), a1, a2, j1r, j2r, int(u1), int(u2)))


def compute_question_result(S, q, MODELS, df_q, df_a, df_r, rng, aggregate_config, judge_config):
    question_id = int(df_q[q][0])
    xq1, xq2 = df_q[q][1], df_q[q][2]
    selected_models = MODELS[S]
    yq1 = df_a[(df_a["model"].isin(selected_models)) & (df_a["question_id"] == question_id)]["y1"].dropna().tolist()
    yq2 = df_a[(df_a["model"].isin(selected_models)) & (df_a["question_id"] == question_id)]["y2"].dropna().tolist()

    if len(yq1) == 0 or len(yq2) == 0:
        raise ValueError(f"Missing candidate answers for question_index={q}, question_id={question_id}, subset={list(map(int, S))}")

    if len(S) == 1:
        yq1S, yq2S = yq1[0], yq2[0]
    else:
        yq1S = aggregate(xq1, yq1, rng, turn=1, aggregate_config=aggregate_config)
        yq2S = aggregate([xq1, yq1S, xq2], yq2, rng, turn=2, aggregate_config=aggregate_config)

    if 20 <= q < 50:
        rq1, rq2 = df_r[q][1], df_r[q][2]
        Uq1_response = judge_with_ref([xq1, rq1], yq1S, turn=1, judge_config=judge_config)
        Uq1 = response_to_rating(Uq1_response)
        Uq2_response = judge_with_ref([xq1, rq1, yq1S, xq2, rq2], yq2S, turn=2, judge_config=judge_config)
        Uq2 = response_to_rating(Uq2_response)
    else:
        Uq1_response = judge_wo_ref(xq1, yq1S, turn=1, judge_config=judge_config)
        Uq1 = response_to_rating(Uq1_response)
        Uq2_response = judge_wo_ref([xq1, yq1S, xq2], yq2S, turn=2, judge_config=judge_config)
        Uq2 = response_to_rating(Uq2_response)

    if Uq1 is None:
        Uq1 = 5
    if Uq2 is None:
        Uq2 = 5

    return {
        "aggregate_turn1_response": yq1S,
        "aggregate_turn2_response": yq2S,
        "judge_turn1_response": Uq1_response,
        "judge_turn2_response": Uq2_response,
        "judge_turn1_rating": Uq1,
        "judge_turn2_rating": Uq2,
    }


# ---------- ESS / sweep helpers (no PSIS) ----------
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


def main():
    if args.use_numba:
        if not sampler_module.NUMBA_AVAILABLE:
            print("ERROR: --use-numba is True, but numba is not available in this environment.", flush=True)
            sys.exit(1)
        sampler_module.NUMBA_AVAILABLE = True
        print("Sampler mode: numba fast path enabled (--use-numba=true)", flush=True)
    else:
        sampler_module.NUMBA_AVAILABLE = False
        print("Sampler mode: pure Python fallback (--use-numba=false)", flush=True)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory
            print(f"GPU {i}: {name}, VRAM: {total_mem / (1024**3):.2f} GB", flush=True)
    else:
        print("No GPU available.", flush=True)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    print(f"Seed fixed: {seed}", flush=True)
    print(f"Questions: {len(EVAL_QS)} (1-based: {sorted(i+1 for i in EVAL_QS)})", flush=True)

    sweep_max = args.sweep_max
    sweep_type = args.sweep_type
    sweep_step = getattr(args, "sweep_step", 1) if sweep_type == "linear" else None
    n_samples = args.n_samples
    N_min = args.n_min
    n = len(MODELS)

    if sweep_type == "exp":
        # Power-of-two: 1, 2, 4, 8, ..., sweep_max
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
        # Linear: step, 2*step, ..., up to sweep_max (include sweep_max)
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
    unique_pairs = list(dict.fromkeys(unique_pairs))  # preserve order, no duplicates; (0,0) only once

    df_benchmark = load_mtbench_single_df()
    df_pairwise = load_chatbot_arena_df(HF_TOKEN)
    df_benchmark, df_pairwise = prepare_benchmark_data(df_benchmark, df_pairwise)
    df_a, df_q, df_r = prepare_qa_data(df_benchmark)
    lam, out_adj = build_lam_and_out_adj(df_pairwise)
    print(f"Prepared graph with {n} players", flush=True)

    # ---------- Phase 1: sample 1000 per unique param pair ----------
    sampling_t0 = time.time()
    print(f"Sampling start timestamp: {sampling_t0:.3f}", flush=True)
    samples_by_pair = {}
    for i, (alpha, beta) in enumerate(unique_pairs):
        print(f"Sampling {i+1}/{len(unique_pairs)}: alpha={alpha}, beta={beta}", flush=True)
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
        # print(df_rank, flush=True)
    sampling_t1 = time.time()
    sampling_elapsed = sampling_t1 - sampling_t0
    print(f"Sampling end timestamp: {sampling_t1:.3f}", flush=True)
    print(
        f"Sampling elapsed time for all sweep pairs: {sampling_elapsed:.2f} sec ({sampling_elapsed / 60.0:.2f} min)",
        flush=True,
    )

    # ---------- Phase 2: (0,0) kept once, shared by all sweeps; then ESS-based thinning per sweep ----------
    kept_0_0 = list(samples_by_pair[(0, 0)])
    rng.shuffle(kept_0_0)
    kept_0_0 = kept_0_0[:n_samples]
    print(f"(0,0) kept once: {len(kept_0_0)} samples (shared by alpha, beta, both)", flush=True)

    use_ess_reuse = not args.no_ess_reuse
    if not use_ess_reuse:
        print("ESS reuse disabled: each param pair keeps full n_samples (no thinning)", flush=True)

    def run_sweep_thinning(sweep_name, sweep_list, shared_kept_00):
        kept_by_pair = {}
        p0 = sweep_list[0]
        kept_by_pair[p0] = shared_kept_00
        print(f"  {sweep_name} {p0}: use shared {len(kept_by_pair[p0])} (direct)", flush=True)

        for i in range(1, len(sweep_list)):
            p_pair = sweep_list[i]
            if use_ess_reuse:
                q_pair = sweep_list[i - 1]
                q_samples = kept_by_pair[q_pair]
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
                kept_by_pair[p_pair] = raw[:n_new]
                print(f"  {sweep_name} {q_pair} -> {p_pair}: ESS={ess:.1f}, keep {n_new} fresh (direct)", flush=True)
            else:
                raw = list(samples_by_pair[p_pair])
                rng.shuffle(raw)
                kept_by_pair[p_pair] = raw[:n_samples]
                print(f"  {sweep_name} {p_pair}: keep all {n_samples} (no ESS reuse)", flush=True)
        return kept_by_pair

    if use_ess_reuse:
        print("ESS-based thinning (alpha sweep)", flush=True)
    else:
        print("No thinning (alpha sweep)", flush=True)
    kept_alpha = run_sweep_thinning("alpha", alpha_sweep, kept_0_0)
    if use_ess_reuse:
        print("ESS-based thinning (beta sweep)", flush=True)
    else:
        print("No thinning (beta sweep)", flush=True)
    kept_beta = run_sweep_thinning("beta", beta_sweep, kept_0_0)
    if use_ess_reuse:
        print("ESS-based thinning (both sweep)", flush=True)
    else:
        print("No thinning (both sweep)", flush=True)
    kept_both = run_sweep_thinning("both", both_sweep, kept_0_0)

    # ---------- Phase 3: collect unique subsets from each pair's direct kept samples only (each pair once) ----------
    perm_chunk_size = args.perm_chunk_size
    seen_masks = set()
    unique_masks = []
    checkpoint_after_indices = []

    items = []
    for (sweep_name, sweep_list, kept_dict) in [
        ("alpha", alpha_sweep, kept_alpha),
        ("beta", beta_sweep, kept_beta),
        ("both", both_sweep, kept_both),
    ]:
        for param in sweep_list:
            items.append((sweep_name, param, kept_dict[param]))

    for sweep_name, (alpha, beta), kept in items:
        n_perms = len(kept)
        n_chunks = (n_perms + perm_chunk_size - 1) // perm_chunk_size
        for chunk_idx in range(n_chunks):
            start_perm = chunk_idx * perm_chunk_size
            end_perm = min((chunk_idx + 1) * perm_chunk_size, n_perms)
            chunk_perms = kept[start_perm:end_perm]
            chunk_masks = set()
            for perm in chunk_perms:
                for i in range(n):
                    S = np.array(perm[: i + 1])
                    mask = subset_to_mask(S)
                    if mask not in seen_masks:
                        chunk_masks.add(mask)
                        seen_masks.add(mask)
            chunk_masks = sorted(chunk_masks)
            rng.shuffle(chunk_masks)
            checkpoint_after_indices.append(len(unique_masks) + len(chunk_masks))
            unique_masks.extend(chunk_masks)
            print(
                f"  ({sweep_name}) (alpha={alpha}, beta={beta}) perm chunk {chunk_idx+1}/{n_chunks} (perms {start_perm+1}–{end_perm}): "
                f"{len(chunk_masks)} new subsets, total {len(unique_masks)}",
                flush=True,
            )

    n_total_subsets = len(unique_masks)
    reuse_note = "no ESS reuse" if not use_ess_reuse else "ESS reuse"
    print(f"Unique subsets: {n_total_subsets} (SNIS noexp, {reuse_note}, ordered by sweep then param then chunk)", flush=True)

    shard_id = args.shard
    n_shards = args.n_shards
    if shard_id < 1 or shard_id > n_shards:
        raise ValueError(f"--shard must be in [1, --n-shards]; got shard={shard_id}, n_shards={n_shards}")
    chunk_size = (n_total_subsets + n_shards - 1) // n_shards
    start_idx = (shard_id - 1) * chunk_size
    end_idx = min(shard_id * chunk_size, n_total_subsets)
    batch_masks = unique_masks[start_idx:end_idx]
    print(f"Shard {shard_id}/{n_shards}: subsets {start_idx+1}–{end_idx} ({len(batch_masks)} subsets)", flush=True)

    q_suffix = cache_question_suffix(EVAL_QS)
    db_path = f"cache/subset_cache_{q_suffix}.db"
    conn = init_cache_db(db_path)
    print(f"SQLite cache: {db_path}", flush=True)

    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    use_vllm = config.get("use_vllm_inprocess", False)
    print(f"Loading {model_name}{' (vLLM in-process)' if use_vllm else ''}...", file=sys.stderr, flush=True)
    model, tokenizer = load_model_and_tokenizer(model_name, config)
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Ready.", file=sys.stderr, flush=True)

    aggregate_config = {"model": model, "tokenizer": tokenizer, "config": config, "max_new_tokens": MAX_NEW_TOKENS}
    judge_config = {"model": model, "tokenizer": tokenizer, "config": config, "max_new_tokens": MAX_NEW_TOKENS}

    n_subsets_batch = len(batch_masks)
    n_tasks = n_subsets_batch * len(EVAL_QS)
    batch_size = max(1, args.batch_size)
    commit_batch_size = max(1, int(args.commit_batch_size))
    use_batch = use_vllm and batch_size > 1

    def get_yq1(S, q):
        question_id = int(df_q[q][0])
        sel = [MODELS[i] for i in S]
        return df_a[(df_a["model"].isin(sel)) & (df_a["question_id"] == question_id)]["y1"].dropna().tolist()

    def get_yq2(S, q):
        question_id = int(df_q[q][0])
        sel = [MODELS[i] for i in S]
        return df_a[(df_a["model"].isin(sel)) & (df_a["question_id"] == question_id)]["y2"].dropna().tolist()

    if use_batch:
        uncached = []
        for idx, subset_mask in enumerate(batch_masks):
            S = mask_to_subset(subset_mask, n)
            for q in EVAL_QS:
                if get_cached_entry(conn, subset_mask, q) is None:
                    uncached.append((idx, subset_mask, S, q))
        if uncached:
            n_waves = (len(uncached) + batch_size - 1) // batch_size
            print(f"  Batched run: {len(uncached)} uncached, batch_size={batch_size}, {n_waves} wave(s)", flush=True)
            for wave_idx in range(0, len(uncached), batch_size):
                chunk = uncached[wave_idx : wave_idx + batch_size]
                nc = len(chunk)
                wave_no = wave_idx // batch_size + 1
                print(f"  Wave {wave_no}/{n_waves}: tasks {wave_idx + 1}–{wave_idx + nc} ({nc} tasks)", flush=True)
                # Turn 1
                print("    Turn 1/4 (aggregate) start.", flush=True)
                out_agg1 = [None] * nc
                prompts_agg1 = []
                indices_agg1 = []
                for i, (_idx, _mask, S, q) in enumerate(chunk):
                    yq1 = get_yq1(S, q)
                    if len(yq1) == 0:
                        raise ValueError(f"Missing y1 for subset mask {_mask} question q{q}")
                    if len(S) == 1:
                        out_agg1[i] = yq1[0]
                    else:
                        xq1 = df_q[q][1]
                        prompts_agg1.append(build_aggregate_prompt(xq1, yq1, rng, 1))
                        indices_agg1.append(i)
                if prompts_agg1:
                    outs = []
                    for j in range(0, len(prompts_agg1), batch_size):
                        outs.extend(
                            generate_batch(
                                model, tokenizer, prompts_agg1[j : j + batch_size], config, MAX_NEW_TOKENS
                            )
                        )
                    for j, idx in enumerate(indices_agg1):
                        out_agg1[idx] = outs[j]
                print("    Turn 1/4 (aggregate) done.", flush=True)
                # Turn 2
                print("    Turn 2/4 (aggregate) start.", flush=True)
                out_agg2 = [None] * nc
                prompts_agg2 = []
                indices_agg2 = []
                for i, (_idx, _mask, S, q) in enumerate(chunk):
                    yq2 = get_yq2(S, q)
                    if len(yq2) == 0:
                        raise ValueError(f"Missing y2 for subset mask {_mask} question q{q}")
                    if len(S) == 1:
                        out_agg2[i] = yq2[0]
                    else:
                        xq1, xq2 = df_q[q][1], df_q[q][2]
                        y1S = out_agg1[i]
                        prompts_agg2.append(build_aggregate_prompt([xq1, y1S, xq2], yq2, rng, 2))
                        indices_agg2.append(i)
                if prompts_agg2:
                    outs2 = []
                    for j in range(0, len(prompts_agg2), batch_size):
                        outs2.extend(
                            generate_batch(
                                model, tokenizer, prompts_agg2[j : j + batch_size], config, MAX_NEW_TOKENS
                            )
                        )
                    for j, idx in enumerate(indices_agg2):
                        out_agg2[idx] = outs2[j]
                print("    Turn 2/4 (aggregate) done.", flush=True)
                # Turn 3
                print("    Turn 3/4 (judge) start.", flush=True)
                prompts_j1 = []
                for i, (_idx, _mask, S, q) in enumerate(chunk):
                    xq1 = df_q[q][1]
                    y1S = out_agg1[i]
                    if 20 <= q < 50:
                        rq1 = df_r[q][1]
                        prompts_j1.append(build_judge_with_ref_prompt([xq1, rq1], y1S, 1))
                    else:
                        prompts_j1.append(build_judge_wo_ref_prompt(xq1, y1S, 1))
                out_j1 = []
                for i in range(0, len(prompts_j1), batch_size):
                    out_j1.extend(generate_batch(model, tokenizer, prompts_j1[i : i + batch_size], config, MAX_NEW_TOKENS))
                print("    Turn 3/4 (judge) done.", flush=True)
                # Turn 4
                print("    Turn 4/4 (judge) start.", flush=True)
                prompts_j2 = []
                for i, (_idx, _mask, S, q) in enumerate(chunk):
                    xq1, xq2 = df_q[q][1], df_q[q][2]
                    y1S, y2S = out_agg1[i], out_agg2[i]
                    if 20 <= q < 50:
                        rq1, rq2 = df_r[q][1], df_r[q][2]
                        prompts_j2.append(build_judge_with_ref_prompt([xq1, rq1, y1S, xq2, rq2], y2S, 2))
                    else:
                        prompts_j2.append(build_judge_wo_ref_prompt([xq1, y1S, xq2], y2S, 2))
                out_j2 = []
                for i in range(0, len(prompts_j2), batch_size):
                    out_j2.extend(generate_batch(model, tokenizer, prompts_j2[i : i + batch_size], config, MAX_NEW_TOKENS))
                print("    Turn 4/4 (judge) done.", flush=True)
                n_commit_chunks = (nc + commit_batch_size - 1) // commit_batch_size
                for c0 in range(0, nc, commit_batch_size):
                    c1 = min(c0 + commit_batch_size, nc)
                    chunk_no = c0 // commit_batch_size + 1
                    for i in range(c0, c1):
                        _idx, subset_mask, S, q = chunk[i]
                        Uq1 = response_to_rating(out_j1[i])
                        Uq2 = response_to_rating(out_j2[i])
                        if Uq1 is None:
                            Uq1 = 5
                        if Uq2 is None:
                            Uq2 = 5
                        upsert_cached_entry(
                            conn, subset_mask, q,
                            out_agg1[i], out_agg2[i], out_j1[i], out_j2[i], Uq1, Uq2,
                        )
                    conn.commit()
                    print(
                        f"  Wave {wave_no}/{n_waves}: commit chunk {chunk_no}/{n_commit_chunks} "
                        f"rows {c0 + 1}–{c1} ({c1 - c0} rows, commit_batch_size={commit_batch_size}).",
                        flush=True,
                    )
                print(f"  Wave {wave_no}/{n_waves}: done, total {nc} rows committed.", flush=True)

    done = 0
    hits = 0
    for idx, subset_mask in enumerate(batch_masks):
        S = mask_to_subset(subset_mask, n)
        s_size = len(S)
        turn1_scores = []
        turn2_scores = []
        for q in EVAL_QS:
            done += 1
            cached = get_cached_entry(conn, subset_mask, q)
            if cached is not None:
                hits += 1
                turn1_scores.append(cached["judge_turn1_rating"])
                turn2_scores.append(cached["judge_turn2_rating"])
                if done % 1000 == 0 or done == n_tasks:
                    print(f"  [{done}/{n_tasks}] subset {idx+1}/{n_subsets_batch} (|S|={s_size}) q{q+1} cache hit (total hits {hits})", flush=True)
                continue
            result = compute_question_result(S, q, MODELS, df_q, df_a, df_r, rng, aggregate_config, judge_config)
            turn1_scores.append(result["judge_turn1_rating"])
            turn2_scores.append(result["judge_turn2_rating"])
            upsert_cached_entry(
                conn, subset_mask, q,
                result["aggregate_turn1_response"],
                result["aggregate_turn2_response"],
                result["judge_turn1_response"],
                result["judge_turn2_response"],
                result["judge_turn1_rating"],
                result["judge_turn2_rating"],
            )
            if done % 50 == 0 or done == n_tasks:
                print(f"  [{done}/{n_tasks}] subset {idx+1}/{n_subsets_batch} (|S|={s_size}) q{q+1} computed", flush=True)
        conn.commit()
        t1_avg = sum(turn1_scores) / len(turn1_scores) if turn1_scores else 0
        t2_avg = sum(turn2_scores) / len(turn2_scores) if turn2_scores else 0
        if idx % 1000 == 0:
            print(f"  subset {idx+1}/{n_subsets_batch} (|S|={s_size}) done: turn1 avg {t1_avg:.2f}, turn2 avg {t2_avg:.2f}", flush=True)

    conn.close()
    print(f"Done. Cache hits: {hits}, computed: {n_tasks - hits}", flush=True)


if __name__ == "__main__":
    main()
