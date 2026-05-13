#!/bin/bash
# Reproduce all four simulation results in Section 5 (and Appendix F).
#
# Outputs:
#   save/<sim>/   — pickled simulation outputs consumed by the plot scripts
#   figure/<sim>/ — final PDF figures (one subdir per simulation)
#
# This script runs everything end-to-end. Expected runtime: several hours on a
# multi-core CPU; no GPU required. Comment out individual blocks if you only
# want to reproduce a subset.

set -euo pipefail

cd "$(dirname "$0")"

# ---- Avoid nested BLAS/OpenMP oversubscription across worker processes ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# =============================================================================
# Sim 1: mixing time of MCMC (Section 5; App F.1)
#   main_mixing.py  -> save/mixing/mixing.pkl
#   plot_mixing.py  -> figure/mixing/*.pdf  (loads pkl directly)
# =============================================================================
echo "[1/4] Sim 1: mixing time"
python -u main_mixing.py

# =============================================================================
# Sim 2a: Monte Carlo accuracy (Section 5; App F.2)
#   main_accuracy.py   -> save/accuracy/accuracy_scenario{1,2}.pkl
#                         (and accuracy_scenario{1,2}_n*.pkl partials)
#   (combine step)     -> save/accuracy/accuracy_all.pkl
#   plot_accuracy.py   -> figure/accuracy/*.pdf
# =============================================================================
echo "[2/4] Sim 2a: Monte Carlo accuracy"
python -u main_accuracy.py \
    --scenario all \
    --n_grid "32,128,512,2048"

# Combine the two per-scenario pickles into the file the plot script expects.
# Each per-scenario pickle has {'config': ..., '<scenario>': {...}}; we merge
# them into a single dict with one 'config' and both scenario keys.
python -c "
import pickle
from pathlib import Path
out = {}
for scen in ('scenario1', 'scenario2'):
    p = Path('save/accuracy') / f'accuracy_{scen}.pkl'
    if not p.exists():
        continue
    d = pickle.load(open(p, 'rb'))
    if 'config' not in out and 'config' in d:
        out['config'] = d['config']
    if scen in d:
        out[scen] = d[scen]
pickle.dump(out, open('save/accuracy/accuracy_all.pkl', 'wb'))
print(f'Combined {[k for k in out if k != \"config\"]} into save/accuracy/accuracy_all.pkl')
"

# =============================================================================
# Sim 2b: surrogate-assisted estimator comparison (Section 5; App F.2)
#   main_surrogate.py  -> save/surrogate/surrogate.pkl + surrogate_partial_n*.pkl
#   plot_surrogate.py  -> figure/surrogate/*.pdf
# =============================================================================
echo "[3/4] Sim 2b: surrogate comparison"
python -u main_surrogate.py \
    --scenario all \
    --n_grid "64,128,256,512,1024"

# =============================================================================
# Sim 3: priority sweeping (Section 5; App F.3)
#   main_sweep.py  -> save/sweep/sweep_default.pkl
#   plot_sweep.py  -> figure/sweep/*.pdf
# =============================================================================
echo "[4/4] Sim 3: priority sweeping"
python -u main_sweep.py

# =============================================================================
# Plot all four figures
# =============================================================================
echo "Rendering figures into figure/"
python plot_mixing.py
python plot_accuracy.py
python plot_surrogate.py
python plot_sweep.py

echo "Done. Figures written under: $(pwd)/figure/"
