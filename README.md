# Generalized Priority-Aware Shapley Value

This repository provides the implementation used to reproduce the numerical
results in **Generalized Priority-Aware Shapley Value (GPASV)**.

## Requirements

- Python 3.11+
- Required libraries (pinned in `requirements.txt`):
  - `numpy`, `pandas`, `scipy`, `scikit-learn`
  - `numba` (sampler acceleration)
  - `matplotlib`, `seaborn`
  - `torch`, `torchvision`, `torchaudio` (LLM evaluation only)
  - `transformers`, `huggingface_hub`, `datasets` (LLM evaluation only)
  - `vllm` (batched local inference for the LLM evaluation)

Install with:

```bash
pip install -r requirements.txt
```

If you are using a CUDA-enabled GPU, install PyTorch with the appropriate CUDA
build (the LLM evaluation in `exp2_llm_evaluation/` requires a GPU):

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

The LLM evaluation downloads MT-Bench and Chatbot Arena from HuggingFace. Set
your HuggingFace token before running:

```bash
export HF_TOKEN=hf_xxx
```

## Folder Structure

- `exp1_simulation/` — synthetic-graph simulations on cyclic directed graphs (Section 5).
- `exp2_llm_evaluation/` — LLM ensemble valuation on MT-Bench with the Chatbot
  Arena pairwise-preference graph (Section 6).

## Reproducing the Experiments

### 1. Simulation Studies (Section 5)

Three simulations:

- **Sim 1: mixing time of MCMC** — `main_mixing.py` runs the adjacent-swap
  Metropolis–Hastings sampler under random and greedy initialization across
  graph families and grid sizes; `plot_mixing.py` produces the mixing-time
  figure.
- **Sim 2: Monte Carlo accuracy** — `main_accuracy.py` measures absolute relative
  error of the direct permutation estimator across two scenarios; `main_surrogate.py`
  compares against linear and quadratic surrogate-assisted estimators under
  matched utility-evaluation budgets. `plot_accuracy.py` and `plot_surrogate.py`
  render the corresponding figures.
- **Sim 3: priority sweeping** — `main_sweep.py` sweeps soft and hard priority
  temperatures and tracks group-sum GPASV under two utility families;
  `plot_sweep.py` renders the sweep figure.

To run everything end-to-end:

```bash
cd exp1_simulation
bash run_simulation.sh
```

`run_simulation.sh` runs the four `main_*.py` scripts (long-running) and then
invokes the four `plot_*.py` scripts to write figures into `figure/`.

### 2. LLM Evaluation (Section 6)

This experiment values 20 LLMs on MT-Bench using the Chatbot Arena
pairwise-preference graph as a hard cyclic priority and an open-source vs.
paid label as a soft priority.

Pipeline:

1. `main_subset.py` populates a (subset, prompt) cache by running an
   aggregator–judge pipeline (Qwen3.5-35B-A3B-FP8 served locally via vLLM)
   over every prefix coalition of every sampled permutation. One question per
   invocation; parallelise across the 80 MT-Bench prompts if you have multiple
   GPUs.
2. `main_value.py` computes per-question GPASV value pickles from the cache
   for all 19 priority regimes along the three sweeps.
3. `main_ess.py` computes the post-hoc ESS matrix used for the SNIS reuse
   table.
4. `plot_llm.py` renders the five figures (two for the main text, three for
   the appendix).

To run end-to-end (GPU + `HF_TOKEN` required):

```bash
export HF_TOKEN=hf_xxx
cd exp2_llm_evaluation
bash run_llm.sh
```


