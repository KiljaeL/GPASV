import json
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

# Regex for parsing MT-bench prompts
USER_RE = re.compile(r"### User:\n(.*?)(?=\n\n### |\Z)", re.DOTALL)
ASST_RE = re.compile(r"### Assistant.*?:\n(.*?)(?=\n\n### |\Z)", re.DOTALL)
START_ANY = re.compile(r"<\|The Start of Assistant.*?Conversation with User\|>")
END_ANY = re.compile(r"<\|The End of Assistant.*?Conversation with User\|>")
START_REF = re.compile(r"<\|The Start of Reference Answer\|>")
END_REF = re.compile(r"<\|The End of Reference Answer\|>")
REF_ANS_RE = re.compile(r"### Reference answer:\n(.*?)(?=\n\n### |\Z)", re.DOTALL)


def load_mtbench_single_df():
    repo_id = "lmsys/mt-bench"
    repo_type = "space"
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    single_files = [
        f for f in files
        if f.startswith("data/mt_bench/model_judgment/")
        and "single" in f.lower()
        and f.lower().endswith(".jsonl")
    ]
    if not single_files:
        raise FileNotFoundError(
            "No MT-bench single judgment jsonl found under data/mt_bench/model_judgment/"
        )
    target = next((f for f in single_files if "gpt-4" in f.lower()), single_files[0])
    local_path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=target)
    rows = []
    with open(local_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for c in ["question_id", "turn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_chatbot_arena_df(token):
    ds = load_dataset(
        "lmsys/chatbot_arena_conversations",
        split="train",
        token=token,
    )
    return ds.to_pandas()


def _get_block(s):
    if not isinstance(s, str):
        return ""
    m1 = START_ANY.search(s)
    m2 = END_ANY.search(s)
    if m1 and m2 and m1.end() < m2.start():
        return s[m1.end() : m2.start()].strip()
    k = s.find("### User:\n")
    if k != -1:
        return s[k:].strip()
    return ""


def _extract_ref_block(s):
    if not isinstance(s, str):
        return ""
    m1 = START_REF.search(s)
    m2 = END_REF.search(s)
    if m1 and m2 and m1.end() < m2.start():
        return s[m1.end() : m2.start()].strip()
    return ""


def parse_turn2(s):
    block = _get_block(s)
    users = [u.strip() for u in USER_RE.findall(block)]
    assts = [a.strip() for a in ASST_RE.findall(block)]
    return {
        "x1": users[0] if len(users) > 0 else None,
        "y1": assts[0] if len(assts) > 0 else None,
        "x2": users[1] if len(users) > 1 else None,
        "y2": assts[1] if len(assts) > 1 else None,
    }


def parse_refs(s):
    block = _extract_ref_block(s)
    rs = [x.strip() for x in REF_ANS_RE.findall(block)]
    return {
        "r1": rs[0] if len(rs) > 0 else None,
        "r2": rs[1] if len(rs) > 1 else None,
    }


def normalize_pairwise(df_pairwise):
    df = df_pairwise.copy()
    df.loc[df["model_a"] == "RWKV-4-Raven-14B", "model_a"] = "rwkv-4-raven-14b"
    df.loc[df["model_b"] == "RWKV-4-Raven-14B", "model_b"] = "rwkv-4-raven-14b"
    return df


def normalize_benchmark(df_benchmark):
    df = df_benchmark.copy()
    df.loc[df["model"] == "palm-2-chat-bison-001", "model"] = "palm-2"
    df.loc[df["model"] == "vicuna-13b-v1.3", "model"] = "vicuna-13b"
    df.loc[df["model"] == "vicuna-7b-v1.3", "model"] = "vicuna-7b"
    df.loc[df["model"] == "oasst-sft-4-pythia-12b", "model"] = "oasst-pythia-12b"
    return df


def prepare_benchmark_data(df_benchmark, df_pairwise):
    df_pairwise = normalize_pairwise(df_pairwise)
    df_benchmark = normalize_benchmark(df_benchmark)
    df_benchmark = df_benchmark[
        df_benchmark["model"].isin(df_pairwise["model_a"].unique())
    ].copy()
    return df_benchmark, df_pairwise


def prepare_qa_data(df_benchmark):
    df_t2 = df_benchmark[df_benchmark["turn"] == 2].copy().reset_index(drop=True)
    df_t2["question_id"] -= 81

    parsed = df_t2["user_prompt"].apply(parse_turn2).apply(pd.Series).reset_index(drop=True)
    df_a = pd.concat([df_t2, parsed], axis=1)
    df_a = df_a[["question_id", "model", "x1", "y1", "x2", "y2"]]
    df_q = np.array(df_a[["question_id", "x1", "x2"]][:80])

    df_ref = df_t2[(df_t2["question_id"] >= 20) & (df_t2["question_id"] <= 49)].copy()
    parsed_ref = df_ref["user_prompt"].apply(parse_refs).apply(pd.Series)
    df_ref = pd.concat(
        [df_ref[["question_id"]].reset_index(drop=True), parsed_ref.reset_index(drop=True)],
        axis=1,
    )
    df_ref = df_ref.dropna(subset=["r1", "r2"], how="all")
    df_ref = df_ref.groupby("question_id", as_index=False).first()

    df_r_full = pd.DataFrame({"question_id": np.arange(80)})
    df_r_full = df_r_full.merge(df_ref, on="question_id", how="left")
    df_r = df_r_full[["question_id", "r1", "r2"]].to_numpy()

    return df_a, df_q, df_r
