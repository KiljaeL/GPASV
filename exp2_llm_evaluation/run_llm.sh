#!/bin/bash
# Reproduce the LLM-evaluation results in Section 6 (and Appendix G).
#
# Pipeline:
#   1. main_subset.py  — populate (subset, prompt) cache by running the
#                        aggregator-judge LLM pipeline on every prefix of every
#                        sampled permutation. ONE QUESTION PER INVOCATION.
#                        Dominant cost: ~25M LLM calls total across all 80
#                        questions (see App G.3 for a per-question breakdown).
#   2. main_value.py   — compute per-question GPASV value pickles for all 19
#                        priority regimes from the cache.
#   3. main_ess.py     — compute the post-hoc ESS matrix used by the SNIS reuse
#                        table in App G.3.
#   4. plot_llm.py     — render the five figures (two main + three appendix).
#
# Requirements:
#   - CUDA GPU with vLLM-compatible memory (Qwen3.5-35B-A3B-FP8 served
#     in-process). Tested on H100/H200.
#   - HF_TOKEN exported in the environment (for Chatbot Arena access).
#
# This script runs all 80 questions sequentially in step 1, which is slow
# (about 1 GPU-day per question). For real reproduction, parallelise across
# questions using SLURM array jobs or equivalent.

set -euo pipefail

cd "$(dirname "$0")"

: "${HF_TOKEN:?HF_TOKEN is not set. Run: export HF_TOKEN=hf_xxx}"

mkdir -p cache save figure

MODEL="${MODEL:-qwen3.5-35b-fp8}"
BATCH_SIZE="${BATCH_SIZE:-8300}"

# =============================================================================
# Step 1: populate the (subset, prompt) cache for all 80 MT-Bench questions.
#         Each invocation handles one question. Parallelise across multiple
#         GPUs / SLURM array tasks if available.
# =============================================================================
echo "[1/4] Populating cache (this will take many GPU-days serially)"
for q in $(seq 1 80); do
    echo "  -> question $q"
    python -u main_subset.py \
        --model "$MODEL" \
        --question "$q" \
        --batch-size "$BATCH_SIZE" \
        --no-ess-reuse
done

# =============================================================================
# Step 2: compute per-question GPASV values for all 19 priority regimes.
# =============================================================================
echo "[2/4] Computing GPASV values per question"
python -u main_value.py \
    --question all \
    --sweep-type exp

# =============================================================================
# Step 3: compute the post-hoc ESS matrix (used by the SNIS reuse table).
# =============================================================================
echo "[3/4] Computing ESS matrix"
python -u main_ess.py \
    --n-samples 1000

# =============================================================================
# Step 4: render all five figures (two main + three appendix).
# =============================================================================
echo "[4/4] Rendering figures into figure/"
python plot_llm.py --which all

echo "Done. Figures written to: $(pwd)/figure/"
