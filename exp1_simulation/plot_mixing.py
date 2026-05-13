"""
Plot script for Simulation: Mixing time of MCMC.

Reads pickled simulation outputs from save/ and writes the figure to figure/.
Auto-generated from GPASV/exp1_simulation/plot_mixing_final.ipynb.

Usage:
    python plot_mixing.py [--save-dir DIR] [--figure-dir DIR]
"""

import argparse
import gc
import pickle
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Load mixing pickle and extract a DataFrame of t*-rows for plotting.
# (Previously a separate extract_mixing_csv.py script + intermediate CSV.)
# ---------------------------------------------------------------------------

_TSTAR_KEY = "first_below_band_t"


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def _load_mixing_payload(path: Path):
    with path.open("rb") as f:
        return _CompatUnpickler(f).load()


def _extract_results(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "results" in payload:
            return payload["results"]
        if "small" in payload:
            small = payload["small"]
            if isinstance(small, list):
                return small
            if isinstance(small, dict) and "results" in small:
                return small["results"]
    raise ValueError("Unexpected mixing-pickle payload format")


def _results_to_frame(results) -> pd.DataFrame:
    records = []
    for r in results:
        omega = r.get("omega_interval", (None, None))
        dr = r.get("random", {}).get("D", {})
        dg = r.get("greedy", {}).get("D", {})
        records.append({
            "n": r.get("n"),
            "graph_family": r.get("graph_family"),
            "graph_p": r.get("graph_p"),
            "graph_rep": r.get("graph_rep"),
            "init_rep": r.get("init_rep"),
            "u_lambda": r.get("u_lambda"),
            "omega_lo": omega[0],
            "omega_hi": omega[1],
            "tstar_dr": dr.get(_TSTAR_KEY),
            "tstar_dg": dg.get(_TSTAR_KEY),
            "horizon": r.get("horizon"),
        })
    df = pd.DataFrame.from_records(records)
    for col in ["n", "graph_p", "graph_rep", "init_rep", "u_lambda",
                "omega_lo", "omega_hi", "tstar_dr", "tstar_dg", "horizon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_mixing_dataframe(pkl_path: Path) -> pd.DataFrame:
    """Load main_mixing.py's pickled output and return a flat t*-rows DataFrame."""
    payload = _load_mixing_payload(pkl_path)
    results = _extract_results(payload)
    del payload
    gc.collect()
    return _results_to_frame(results)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-dir", default="save",
                   help="Directory containing pickled simulation outputs.")
    p.add_argument("--figure-dir", default="figure",
                   help="Directory to write the rendered PDF into.")
    return p.parse_args()


def _run():
    args = _parse_args()
    PKL_PATH = Path('save/mixing/mixing.pkl')
    FIG_DIR = Path('figure/mixing')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    N_GRID = list(range(4, 23, 2))
    P_GRID = [0.2, 0.8, 1.0]
    ROW_CASES = [('random_dag', 'DAG'), ('random_digraph', 'General Graph')]
    COL_TITLES = {0.2: '$p_\\mathrm{edge}=0.2$', 0.8: '$p_\\mathrm{edge}=0.8$', 1.0: '$p_\\mathrm{edge}=1.0$'}
    TSTAR_KEY = 'first_below_band_t'
    DG_STYLE = '-'
    DR_STYLE = '--'
    CASES = {1: {'lam': 1.0, 'omega': (0.0, 0.5), 'color': '#a1c9f4', 'label': '$\\lambda\\equiv 1,\\ \\widetilde\\omega\\sim\\mathrm{Unif}(0,0.5)$', 'short_label': 'Weak $\\lambda$, Strong $\\omega$', 'restrict_family': None, 'lw': 2.4, 'ms': 5}, 2: {'lam': 1.0, 'omega': (0.5, 1.0), 'color': '#ffb482', 'label': '$\\lambda\\equiv 1,\\ \\widetilde\\omega\\sim\\mathrm{Unif}(0.5,1)$', 'short_label': 'Weak $\\lambda$, Weak $\\omega$', 'restrict_family': None, 'lw': 2.4, 'ms': 5}, 3: {'lam': 100.0, 'omega': (0.0, 0.5), 'color': '#8de5a1', 'label': '$\\lambda\\sim\\mathrm{Unif}(1,100),\\ \\widetilde\\omega\\sim\\mathrm{Unif}(0,0.5)$', 'short_label': 'Strong $\\lambda$, Strong $\\omega$', 'restrict_family': None, 'lw': 2.4, 'ms': 5}, 4: {'lam': 100.0, 'omega': (0.5, 1.0), 'color': '#ff9f9b', 'label': '$\\lambda\\sim\\mathrm{Unif}(1,100),\\ \\widetilde\\omega\\sim\\mathrm{Unif}(0.5,1)$', 'short_label': 'Strong $\\lambda$, Weak $\\omega$', 'restrict_family': None, 'lw': 2.4, 'ms': 5}, 5: {'lam': 1.0, 'omega': (0.0, 0.0), 'color': '#d0bbff', 'label': '$\\lambda\\equiv 1,\\ \\widetilde\\omega\\equiv 0$ (PSV)', 'short_label': 'Weak $\\lambda$, $\\omega\\equiv 0$ (PSV)', 'restrict_family': 'random_dag', 'lw': 3.8, 'ms': 6}, 6: {'lam': 1.0, 'omega': (0.0, 0.0), 'color': '#debb9b', 'label': '$\\lambda\\equiv 1,\\ \\widetilde\\omega\\equiv 0$ (SV)', 'short_label': 'Weak $\\lambda$, $\\omega\\equiv 0$ (SV)', 'restrict_family': 'random_digraph', 'lw': 3.8, 'ms': 6}}
    ALL_CASES = [1, 2, 3, 4, 5, 6]
    sns.set_style('ticks')
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 22
    PANEL_TITLE_FONTSIZE = 22
    AXIS_LABEL_FONTSIZE = 22
    TICK_LABEL_FONTSIZE = 17
    LEGEND_FONTSIZE = 22
    df = load_mixing_dataframe(PKL_PATH)
    print('Loaded from   :', PKL_PATH)
    print('Loaded rows   :', len(df))
    print('n values      :', sorted(df['n'].dropna().astype(int).unique().tolist()))
    print('graph families:', sorted(df['graph_family'].dropna().unique().tolist()))
    print('p values      :', sorted(df['graph_p'].dropna().astype(float).unique().tolist()))
    print('lambda values :', sorted(df['u_lambda'].dropna().astype(float).unique().tolist()))
    print('omega values  :', sorted({(float(a), float(b)) for a, b in zip(df['omega_lo'], df['omega_hi']) if np.isfinite(a) and np.isfinite(b)}))
    def _case_mask(frame: pd.DataFrame, family: str, p_value: float):
        return frame['graph_family'].eq(family) & np.isclose(frame['graph_p'], float(p_value), atol=1e-12, rtol=0.0)
    def _panel_data(frame: pd.DataFrame, lam: float, omega: Tuple[float, float]) -> pd.DataFrame:
        sub = frame[np.isclose(frame['u_lambda'], float(lam), atol=1e-12, rtol=0.0) & np.isclose(frame['omega_lo'], float(omega[0]), atol=1e-12, rtol=0.0) & np.isclose(frame['omega_hi'], float(omega[1]), atol=1e-12, rtol=0.0) & frame['n'].isin(N_GRID)]
        if sub.empty:
            return sub
        return sub.groupby('n', as_index=False)[['tstar_dr', 'tstar_dg']].mean(numeric_only=True).sort_values('n').reset_index(drop=True)
    def _resolve_color(case_id: int, palette: Optional[dict]):
        if palette is not None and case_id in palette:
            return palette[case_id]
        return CASES[case_id]['color']
    def _resolve_lw_ms(case_id: int, lw: float, ms: float, special_lw: float, special_ms: float):
        if case_id <= 4:
            return (lw, ms)
        return (special_lw, special_ms)
    def _draw_case_on_axis(ax, family: str, case_df: pd.DataFrame, case_id: int, palette: Optional[dict], lw: float, ms: float, special_lw: float, special_ms: float, alpha: float):
        spec = CASES[case_id]
        if spec['restrict_family'] is not None and spec['restrict_family'] != family:
            return
        sub = _panel_data(case_df, spec['lam'], spec['omega'])
        if sub.empty:
            return
        color = _resolve_color(case_id, palette)
        line_w, marker_s = _resolve_lw_ms(case_id, lw, ms, special_lw, special_ms)
        dg = sub[sub['tstar_dg'].notna()]
        dr = sub[sub['tstar_dr'].notna()]
        if not dg.empty:
            ax.plot(dg['n'], dg['tstar_dg'], color=color, linestyle=DG_STYLE, marker='o', markersize=marker_s, linewidth=line_w, alpha=alpha)
        if not dr.empty:
            ax.plot(dr['n'], dr['tstar_dr'], color=color, linestyle=DR_STYLE, marker='s', markersize=marker_s, linewidth=line_w, alpha=alpha)
    def _draw_reference_lines(ax):
        n_ref = np.asarray(N_GRID, dtype=np.float64)
        ax.plot(n_ref, n_ref ** 2, color='black', linestyle='-', linewidth=1.8)
        ax.plot(n_ref, 4.0 / np.pi ** 2 * n_ref ** 3 * np.log(n_ref), color='black', linestyle='--', linewidth=1.8)
    SHORT_XTICK_VALUES = [4, 8, 12, 16, 20]
    def _format_axis(ax, tick_label_fontsize: float, xtick_values: Optional[Sequence[int]]=None, xtick_rotation: float=30):
        ticks = list(N_GRID if xtick_values is None else xtick_values)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelrotation=xtick_rotation)
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(n) for n in ticks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(True, which='both', alpha=0.16)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    def _build_legend_handles(cases: Sequence[int], short: bool=False, palette: Optional[dict]=None, lw: float=2.8, special_lw: float=5.0):
        label_key = 'short_label' if short else 'label'
        case_handles = [Line2D([0], [0], color=_resolve_color(c, palette), linestyle='-', linewidth=lw if c <= 4 else special_lw, label=CASES[c][label_key]) for c in cases]
        style_handles = [Line2D([0], [0], color='gray', linestyle=DG_STYLE, linewidth=2.8, label='Greedy'), Line2D([0], [0], color='gray', linestyle=DR_STYLE, linewidth=2.8, label='Random')]
        ref_handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=2.0, label='$n^2$ (ideal)'), Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label='$\\frac{4}{\\pi^2}n^3\\log n$')]
        return (case_handles, style_handles, ref_handles)
    def _add_axes_legend(ax, handles, *, loc, bbox, ncol, frameon, edgecolor, borderaxespad, labelspacing, handlelength, fontsize):
        leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, bbox_transform=ax.transAxes, ncol=ncol, frameon=frameon, edgecolor=edgecolor, borderaxespad=borderaxespad, labelspacing=labelspacing, handlelength=handlelength, prop={'size': fontsize})
        ax.add_artist(leg)
        return leg
    def plot_mixing_full(frame: pd.DataFrame, cases: Optional[Sequence[int]]=None, output_name: str='mixing_time_full.pdf', figsize: Tuple[float, float]=(20, 10), panel_title_fontsize: float=22, axis_label_fontsize: float=22, tick_label_fontsize: float=17, legend_fontsize: float=22, case_legend_ncol: int=3, legend_y: Tuple[float, float, float]=(0.1, 0.05, -0.01), rect: Tuple[float, float, float, float]=(0.03, 0.18, 0.98, 0.98), w_pad: float=1.0, h_pad: float=0.9, palette: Optional[dict]=None, line_width: float=2.4, marker_size: float=5, special_line_width: float=3.8, special_marker_size: float=6, curve_alpha: float=0.95, xtick_rotation: float=30):
        cases = list(ALL_CASES if cases is None else cases)
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
        for i, (family, row_label) in enumerate(ROW_CASES):
            for j, p_value in enumerate(P_GRID):
                ax = axes[i, j]
                case_df = frame[_case_mask(frame, family, p_value)]
                for cid in cases:
                    _draw_case_on_axis(ax, family, case_df, cid, palette=palette, lw=line_width, ms=marker_size, special_lw=special_line_width, special_ms=special_marker_size, alpha=curve_alpha)
                _draw_reference_lines(ax)
                _format_axis(ax, tick_label_fontsize, xtick_rotation=xtick_rotation)
                if i == 0:
                    ax.set_title(COL_TITLES[p_value], fontsize=panel_title_fontsize)
                if j == 0:
                    ax.set_ylabel(row_label, fontsize=axis_label_fontsize)
                if i == len(ROW_CASES) - 1:
                    ax.set_xlabel('$n$', fontsize=axis_label_fontsize)
        case_h, style_h, ref_h = _build_legend_handles(cases, short=False, palette=palette)
        leg1 = fig.legend(handles=case_h, loc='lower center', ncol=case_legend_ncol, bbox_to_anchor=(0.5, legend_y[0]), frameon=False, prop={'size': legend_fontsize})
        fig.add_artist(leg1)
        leg2 = fig.legend(handles=style_h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, legend_y[1]), frameon=False, prop={'size': legend_fontsize})
        fig.add_artist(leg2)
        fig.legend(handles=ref_h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, legend_y[2]), frameon=False, prop={'size': legend_fontsize})
        fig.tight_layout(rect=rect, w_pad=w_pad, h_pad=h_pad)
        out_path = FIG_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)
        print(f'saved: {out_path}')
        return out_path
    def plot_mixing_short(frame: pd.DataFrame, p: float, cases: Optional[Sequence[int]]=None, output_name: str='mixing_time_short.pdf', figsize: Tuple[float, float]=(14, 5.5), panel_title_fontsize: float=18, axis_label_fontsize: float=22, tick_label_fontsize: float=17, legend_fontsize: float=14, ylabel: str='Mixing Time', case_legend_axis: int=0, case_legend_loc: str='upper left', case_legend_bbox: Tuple[float, float]=(0.02, 0.98), case_legend_ncol: int=1, aux_legend_axis: int=1, aux_legend_loc: str='upper left', aux_legend_bbox: Tuple[float, float]=(0.02, 0.98), aux_legend_ncol: int=1, legend_frameon: bool=True, legend_edgecolor: str='0.4', legend_borderaxespad: float=0.0, legend_labelspacing: float=0.4, legend_handlelength: float=2.0, xtick_values: Sequence[int]=SHORT_XTICK_VALUES, rect: Tuple[float, float, float, float]=(0.03, 0.05, 0.98, 0.98), w_pad: float=1.0, palette: Optional[dict]=None, line_width: float=2.4, marker_size: float=5, special_line_width: float=3.8, special_marker_size: float=6, curve_alpha: float=0.95, xtick_rotation: float=30):
        cases = list(ALL_CASES if cases is None else cases)
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        for j, (family, row_label) in enumerate(ROW_CASES):
            ax = axes[j]
            case_df = frame[_case_mask(frame, family, p)]
            for cid in cases:
                _draw_case_on_axis(ax, family, case_df, cid, palette=palette, lw=line_width, ms=marker_size, special_lw=special_line_width, special_ms=special_marker_size, alpha=curve_alpha)
            _draw_reference_lines(ax)
            _format_axis(ax, tick_label_fontsize, xtick_values=xtick_values, xtick_rotation=xtick_rotation)
            ax.set_title(row_label, fontsize=panel_title_fontsize)
            ax.set_xlabel('$n$', fontsize=axis_label_fontsize)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        case_h, style_h, ref_h = _build_legend_handles(cases, short=True, palette=palette)
        _add_axes_legend(axes[case_legend_axis], case_h, loc=case_legend_loc, bbox=case_legend_bbox, ncol=case_legend_ncol, frameon=legend_frameon, edgecolor=legend_edgecolor, borderaxespad=legend_borderaxespad, labelspacing=legend_labelspacing, handlelength=legend_handlelength, fontsize=legend_fontsize)
        _add_axes_legend(axes[aux_legend_axis], style_h + ref_h, loc=aux_legend_loc, bbox=aux_legend_bbox, ncol=aux_legend_ncol, frameon=legend_frameon, edgecolor=legend_edgecolor, borderaxespad=legend_borderaxespad, labelspacing=legend_labelspacing, handlelength=legend_handlelength, fontsize=legend_fontsize)
        fig.tight_layout(rect=rect, w_pad=w_pad)
        out_path = FIG_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)
        print(f'saved: {out_path}')
        return out_path
    plot_mixing_full(df, cases=[1, 2, 3, 4, 5, 6], output_name='mixing_time_full.pdf', figsize=(20, 10), panel_title_fontsize=22, axis_label_fontsize=22, tick_label_fontsize=17, legend_fontsize=22, case_legend_ncol=3, legend_y=(0.1, 0.05, -0.01), rect=(0.03, 0.18, 0.98, 0.98), w_pad=1.0, h_pad=0.9, palette=None, line_width=1.8, marker_size=2, special_line_width=3.8, special_marker_size=6, curve_alpha=0.95, xtick_rotation=30)
    plot_mixing_short(df, p=0.8, cases=[1, 2, 3, 4], output_name='mixing_time_short.pdf', figsize=(9, 5), panel_title_fontsize=18, axis_label_fontsize=18, tick_label_fontsize=18, legend_fontsize=12.8, ylabel='Mixing Time', case_legend_axis=0, case_legend_loc='upper left', case_legend_bbox=(0.005, 0.995), case_legend_ncol=1, aux_legend_axis=0, aux_legend_loc='lower right', aux_legend_bbox=(0.995, 0.005), aux_legend_ncol=1.2, legend_frameon=True, legend_edgecolor='0.4', legend_borderaxespad=0.0, legend_labelspacing=0.01, legend_handlelength=1.3, rect=(0.03, 0.05, 0.98, 0.98), w_pad=0.3, palette={1: '#3e86d4', 2: '#9f49d7', 3: '#50CA6D', 4: '#cd524d'}, line_width=1.8, marker_size=2, special_line_width=3.8, special_marker_size=6, curve_alpha=0.75, xtick_rotation=0)


if __name__ == "__main__":
    _run()
