import random

import numpy as np
import pandas as pd

from sampler import mh_sampler

MODELS = np.array([
    "gpt-4",
    "claude-v1",
    "claude-instant-v1",
    "gpt-3.5-turbo",
    "guanaco-33b",
    "vicuna-13b",
    "wizardlm-13b",
    "palm-2",
    "vicuna-7b",
    "koala-13b",
    "gpt4all-13b-snoozy",
    "mpt-7b-chat",
    "rwkv-4-raven-14b",
    "alpaca-13b",
    "oasst-pythia-12b",
    "fastchat-t5-3b",
    "chatglm-6b",
    "stablelm-tuned-alpha-7b",
    "dolly-v2-12b",
    "llama-13b",
])


def build_pairwise_matrices(df_pairwise, models=None):
    if models is None:
        models = MODELS
    models = np.asarray(models)
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.zeros((n, n), dtype=float)
    T = np.zeros((n, n), dtype=float)

    for _, r in df_pairwise.iterrows():
        a, b = r["model_a"], r["model_b"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        T[i, j] += 1
        T[j, i] += 1
        w = r["winner"]
        if w == "model_a":
            W[i, j] += 1
        elif w == "model_b":
            W[j, i] += 1
        else:
            W[i, j] += 0.5
            W[j, i] += 0.5
    return W, T, models, idx


def build_edges(W, T, models, min_count=50):
    n = len(models)
    P = np.zeros((n, n), dtype=float)
    mask = T > 0
    P[mask] = W[mask] / T[mask]

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            tot = T[i, j]
            if tot < min_count:
                continue
            p_ij = P[i, j]
            if np.isnan(p_ij):
                continue
            if p_ij >= 0.5:
                u, v, p = i, j, p_ij
            else:
                u, v, p = j, i, 1 - p_ij
            w = p - 0.5
            edges.append((u, v, float(w)))
    return edges


def run_ranking(
    df_pairwise,
    n_samples=1000,
    burn_in=100000,
    thin=1000,
    seed=42,
    report_every=100,
    alpha=1.0,
    beta=1.0,
):
    W, T, models, idx = build_pairwise_matrices(df_pairwise)
    edges = build_edges(W, T, models)
    n = len(models)

    lam = [
        0 if m in ["gpt-4", "gpt-3.5-turbo", "claude-v1", "claude-instant-v1", "palm-2"]
        else 1
        for m in models
    ]
    rng = random.Random(seed)

    samples, acc_rate = mh_sampler(
        n=n,
        edges=edges,
        lam=lam,
        alpha=alpha,
        beta=beta,
        seed=seed,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        init_pi=None,
        report_every=report_every,
    )

    R = np.empty((len(samples), n), dtype=np.int32)
    for t, pi in enumerate(samples):
        pos = np.empty(n, dtype=np.int32)
        for rank, model_idx in enumerate(pi):
            pos[int(model_idx)] = rank
        R[t, :] = pos + 1

    mean_rank = R.mean(axis=0)
    std_rank = R.std(axis=0, ddof=1) if len(samples) > 1 else np.zeros(n)
    se_rank = std_rank / np.sqrt(len(samples)) if len(samples) > 1 else np.zeros(n)

    df_rank = pd.DataFrame({
        "model": models,
        "mean_rank": mean_rank,
    }).sort_values("mean_rank", ascending=True).reset_index(drop=True)

    return samples, df_rank
