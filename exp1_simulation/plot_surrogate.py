import argparse
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-dir", default="save",
                   help="Directory containing pickled simulation outputs.")
    p.add_argument("--figure-dir", default="figure",
                   help="Directory to write the rendered PDF into.")
    return p.parse_args()


def _run():
    args = _parse_args()
    sns.set_style('ticks')
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 14
    EXP1_ROOT = Path('.')
    SAVE_ROOT = EXP1_ROOT / 'save' / 'surrogate'
    OUTPUT_DIR = EXP1_ROOT / 'figure' / 'surrogate'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    N_GRID = [64, 128, 256, 512, 1024]
    METHOD_ORDER = ['permutation', 'linear_surrogate_subset', 'quadratic_surrogate_subset']
    METHOD_LABEL = {'permutation': 'Permutation', 'linear_surrogate_subset': 'Linear', 'quadratic_surrogate_subset': 'Quadratic'}
    METHOD_COLOR = {'permutation': '#a1c9f4', 'linear_surrogate_subset': '#ffb482', 'quadratic_surrogate_subset': '#8de5a1'}
    LEGEND_TITLE_METHOD = 'Estimator:'
    SCENARIO_ORDER = ['scenario1', 'scenario2']
    SCENARIO_ROW_LABEL = {'scenario1': 'Scenario 1', 'scenario2': 'Scenario 2'}
    CASE_ORDER = [1, 2, 3, 4]
    CASE_CONFIG = {'scenario1': {1: {'lambda_case': 'ones', 'omega_key': 0.7}, 2: {'lambda_case': 'ones', 'omega_key': 0.3}, 3: {'lambda_case': 'uniform_1_10', 'omega_key': 0.7}, 4: {'lambda_case': 'uniform_1_10', 'omega_key': 0.3}}, 'scenario2': {1: {'lambda_case': 'ones', 'omega_key': 'uniform_0.5_1'}, 2: {'lambda_case': 'ones', 'omega_key': 'uniform_0_0.5'}, 3: {'lambda_case': 'block_uniform_1_10', 'omega_key': 'uniform_0.5_1'}, 4: {'lambda_case': 'block_uniform_1_10', 'omega_key': 'uniform_0_0.5'}}}
    def _load_pickle(path):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        with open(path, 'rb') as f:
            return pickle.load(f)
    def _collect_rows(paths):
        rows = []
        seen = set()
        for path in paths:
            payload = _load_pickle(path)
            for item in payload.get('results', []):
                if str(item.get('init_mode')) != 'greedy':
                    continue
                scenario = str(item.get('scenario'))
                n = int(item.get('n'))
                meta = dict(item.get('meta', {}))
                lambda_case = str(meta.get('lambda_case'))
                omega_raw = meta.get('omega_bar') if scenario == 'scenario1' else meta.get('omega_case')
                omega_key = float(omega_raw) if scenario == 'scenario1' else str(omega_raw)
                for rep in item.get('rep_results', []):
                    rep_idx = rep.get('rep_idx')
                    for method_out in rep.get('methods', []):
                        method = str(method_out.get('method'))
                        if method not in METHOD_ORDER:
                            continue
                        key = (scenario, n, lambda_case, omega_key, method, rep_idx)
                        if key in seen:
                            continue
                        seen.add(key)
                        fit_diag = method_out.get('fit_diagnostics') or {}
                        rows.append({'scenario': scenario, 'n': n, 'lambda_case': lambda_case, 'omega_key': omega_key, 'method': method, 'final_are': method_out.get('final_are', np.nan), 'runtime_others': method_out.get('non_utility_seconds', np.nan), 'fit_mem_gb': fit_diag.get('proxy_tracemalloc_peak_mb', np.nan) / 1024.0})
        return pd.DataFrame(rows)
    PICKLE_PATHS = sorted(SAVE_ROOT.glob('surrogate_partial_n*.pkl'))
    if not PICKLE_PATHS:
        PICKLE_PATHS = [SAVE_ROOT / 'surrogate.pkl']
    DF = _collect_rows(PICKLE_PATHS)
    print('Save directory  :', SAVE_ROOT)
    print('Output directory:', OUTPUT_DIR)
    print('Pickle files    :', len(PICKLE_PATHS))
    print('Rows            :', len(DF))
    print('n values        :', sorted(DF['n'].unique().tolist()))
    def _panel_spec(row_idx, col_idx):
        scenario = SCENARIO_ORDER[row_idx]
        case_id = CASE_ORDER[col_idx]
        cfg = CASE_CONFIG[scenario][case_id]
        return (scenario, case_id, cfg['lambda_case'], cfg['omega_key'])
    def _format_n_xtick(n, pow2=True):

        if pow2 and n > 0:
            e = np.log2(n)
            if abs(e - round(e)) < 1e-09:
                return f'$2^{{{int(round(e))}}}$'
        return str(n)
    def _draw_box_group(ax, panel_df, metric_col, xtick_fontsize=None, ytick_fontsize=None, xtick_rotation=None, xtick_pow2=True):
        base = np.arange(len(N_GRID), dtype=np.float64)
        width = 0.22
        offsets = dict(zip(METHOD_ORDER, np.linspace(-width, width, num=len(METHOD_ORDER))))
        for method in METHOD_ORDER:
            positions, data = ([], [])
            for i, n in enumerate(N_GRID):
                vals = panel_df[(panel_df['method'] == method) & (panel_df['n'] == int(n))][metric_col].dropna().to_numpy(dtype=np.float64)
                if vals.size == 0:
                    continue
                positions.append(base[i] + offsets[method])
                data.append(vals)
            if not data:
                continue
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=True, medianprops={'color': 'black', 'linewidth': 1.2}, whiskerprops={'color': '#4a4a4a', 'linewidth': 1.0}, capprops={'color': '#4a4a4a', 'linewidth': 1.0}, boxprops={'edgecolor': '#4a4a4a', 'linewidth': 1.0})
            for patch in bp['boxes']:
                patch.set_facecolor(METHOD_COLOR[method])
                patch.set_alpha(0.9)
        ax.set_xticks(base)
        tick_kwargs = {}
        if xtick_rotation is not None:
            tick_kwargs['rotation'] = xtick_rotation
        ax.set_xticklabels([_format_n_xtick(n, pow2=xtick_pow2) for n in N_GRID], **tick_kwargs)
        if xtick_fontsize is not None:
            ax.tick_params(axis='x', labelsize=xtick_fontsize)
        if ytick_fontsize is not None:
            ax.tick_params(axis='y', labelsize=ytick_fontsize)
        for x_sep in base[:-1] + 0.5:
            ax.axvline(x=x_sep, color='#9a9a9a', linestyle='--', linewidth=0.9, alpha=0.6, zorder=0)
        ax.set_xlim(base[0] - 0.55, base[-1] + 0.55)
        ax.grid(axis='y', alpha=0.2)
    def _draw_bar_group(ax, panel_df, metric_col, methods, xtick_fontsize=None, ytick_fontsize=None, xtick_rotation=None, xtick_pow2=True):
        base = np.arange(len(N_GRID), dtype=np.float64)
        if len(methods) == 1:
            width = 0.6
            offsets = {methods[0]: 0.0}
        else:
            width = 0.8 / len(methods)
            centers = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, num=len(methods))
            offsets = {m: float(v) for m, v in zip(methods, centers)}
        bottom_candidates = []
        for method in methods:
            positions, means, stds = ([], [], [])
            for i, n in enumerate(N_GRID):
                vals = panel_df[(panel_df['method'] == method) & (panel_df['n'] == int(n))][metric_col].dropna().to_numpy(dtype=np.float64)
                if vals.size == 0:
                    continue
                positions.append(base[i] + offsets[method])
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            if not positions:
                continue
            means_arr = np.asarray(means, dtype=np.float64)
            positive = means_arr[means_arr > 0]
            if positive.size:
                bottom_candidates.append(float(positive.min()))
            ax.bar(positions, means, width=width * 0.9, yerr=stds, capsize=2.5, facecolor=METHOD_COLOR[method], edgecolor='#4a4a4a', linewidth=1.0, alpha=0.9, error_kw={'elinewidth': 1.0, 'ecolor': '#4a4a4a'})
        ax.set_xticks(base)
        tick_kwargs = {}
        if xtick_rotation is not None:
            tick_kwargs['rotation'] = xtick_rotation
        ax.set_xticklabels([_format_n_xtick(n, pow2=xtick_pow2) for n in N_GRID], **tick_kwargs)
        if xtick_fontsize is not None:
            ax.tick_params(axis='x', labelsize=xtick_fontsize)
        if ytick_fontsize is not None:
            ax.tick_params(axis='y', labelsize=ytick_fontsize)
        for x_sep in base[:-1] + 0.5:
            ax.axvline(x=x_sep, color='#9a9a9a', linestyle='--', linewidth=0.9, alpha=0.6, zorder=0)
        ax.set_xlim(base[0] - 0.55, base[-1] + 0.55)
        ax.set_yscale('log')
        if bottom_candidates:
            ax.set_ylim(bottom=min(bottom_candidates) * 0.5)
        ax.grid(axis='y', which='both', alpha=0.2)
    def plot_final_are_grid(frame, output_name, figsize=(16, 5), ylabel='Final ARE', legend_bbox=(0.58, 0.04), layout_rect=(0.08, 0.07, 0.99, 0.95), supylabel_xy=(0.1, 0.54), title_fontsize=None, xlabel_fontsize=None, ylabel_fontsize=None, supylabel_fontsize=None, xtick_fontsize=None, ytick_fontsize=None, xtick_pow2=True, legend_fontsize=None):
        fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=False, sharey=False)
        for row_idx in range(2):
            for col_idx in range(4):
                scenario, case_id, lambda_case, omega_key = _panel_spec(row_idx, col_idx)
                ax = axes[row_idx, col_idx]
                panel_df = frame[(frame['scenario'] == scenario) & (frame['lambda_case'] == lambda_case) & (frame['omega_key'] == omega_key)]
                _draw_box_group(ax, panel_df, metric_col='final_are', xtick_fontsize=xtick_fontsize, ytick_fontsize=ytick_fontsize, xtick_pow2=xtick_pow2)
                if row_idx == 0:
                    ax.set_title(f'Case {case_id}', fontsize=title_fontsize)
                if col_idx == 0:
                    ax.set_ylabel(SCENARIO_ROW_LABEL[scenario], fontsize=ylabel_fontsize)
                if row_idx == 1:
                    ax.set_xlabel('$n$', fontsize=xlabel_fontsize)
        handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor='#4a4a4a', label=METHOD_LABEL[m]) for m in METHOD_ORDER]
        leg_kwargs = dict(handles=handles, loc='lower center', ncol=len(METHOD_ORDER), frameon=False, bbox_to_anchor=legend_bbox)
        if legend_fontsize is not None:
            leg_kwargs['fontsize'] = legend_fontsize
        fig.legend(**leg_kwargs)
        sup_kwargs = dict(x=supylabel_xy[0], y=supylabel_xy[1])
        if supylabel_fontsize is not None:
            sup_kwargs['fontsize'] = supylabel_fontsize
        fig.supylabel(ylabel, **sup_kwargs)
        fig.tight_layout(rect=layout_rect, w_pad=0.8, h_pad=0.8)
        out_path = OUTPUT_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print('Saved', out_path)
    def plot_single_box(frame, case_id, output_name, scenarios=('scenario1', 'scenario2'), figsize=(8.0, 3.2), ylabel='Final ARE', legend_inside_panel=0, legend_inside_loc='upper left', legend_inside_bbox=(0.0, 1.0), legend_inside_borderaxespad=0.0, legend_inside_y_headroom=0.2, legend_inside_frameon=True, legend_inside_framealpha=0.9, legend_inside_edgecolor='0.4', legend_inside_labelspacing=0.3, legend_fontsize=12, legend_handlelength=1.6, legend_handletextpad=0.5, layout_rect=(0.0, 0.04, 1.0, 0.98), title_fontsize=None, xlabel_fontsize=None, ylabel_fontsize=None, xtick_fontsize=None, ytick_fontsize=None, xtick_rotation=None, xtick_pow2=True, w_pad=1.5):

        scenarios = tuple(scenarios)
        fig, axes = plt.subplots(1, len(scenarios), figsize=figsize, sharey=False)
        if len(scenarios) == 1:
            axes = np.array([axes])
        inside_panel_idx = legend_inside_panel % len(scenarios)
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[col_idx]
            cfg = CASE_CONFIG[scenario][case_id]
            panel_df = frame[(frame['scenario'] == scenario) & (frame['lambda_case'] == cfg['lambda_case']) & (frame['omega_key'] == cfg['omega_key'])]
            _draw_box_group(ax, panel_df, metric_col='final_are', xtick_fontsize=xtick_fontsize, ytick_fontsize=ytick_fontsize, xtick_rotation=xtick_rotation, xtick_pow2=xtick_pow2)
            ax.set_title(SCENARIO_ROW_LABEL[scenario], fontsize=title_fontsize)
            ax.set_xlabel('$n$', fontsize=xlabel_fontsize)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
            if col_idx == inside_panel_idx:
                ymin, ymax = ax.get_ylim()
                span = max(ymax - ymin, 1e-12)
                ax.set_ylim(ymin, ymax + legend_inside_y_headroom * span)
        target_ax = axes[inside_panel_idx]
        handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor='#4a4a4a', label=METHOD_LABEL[m]) for m in METHOD_ORDER]
        target_ax.legend(handles=handles, loc=legend_inside_loc, ncol=1, frameon=legend_inside_frameon, framealpha=legend_inside_framealpha, edgecolor=legend_inside_edgecolor, bbox_to_anchor=legend_inside_bbox, bbox_transform=target_ax.transAxes, borderaxespad=legend_inside_borderaxespad, labelspacing=legend_inside_labelspacing, handlelength=legend_handlelength, handletextpad=legend_handletextpad, fontsize=legend_fontsize)
        fig.tight_layout(rect=layout_rect, w_pad=w_pad)
        out_path = OUTPUT_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print('Saved', out_path)
    plot_final_are_grid(frame=DF, output_name='surrogate_full.pdf', figsize=(16, 7), ylabel='ARE at Convergence', legend_bbox=(0.58, 0.04), layout_rect=(0.08, 0.07, 0.99, 0.95), supylabel_xy=(0.1, 0.54), title_fontsize=19, xlabel_fontsize=19, ylabel_fontsize=18, supylabel_fontsize=19, xtick_fontsize=15, ytick_fontsize=15, legend_fontsize=15)
    plot_single_box(frame=DF, case_id=2, output_name='surrogate_short.pdf', scenarios=('scenario1', 'scenario2'), figsize=(7.4, 3.7), ylabel='ARE at Convergence', legend_inside_panel=0, legend_inside_loc='upper left', legend_inside_bbox=(0.0, 1.0), legend_inside_borderaxespad=0.0, legend_inside_y_headroom=0.2, legend_inside_frameon=True, legend_inside_framealpha=0.9, legend_inside_edgecolor='0.4', legend_inside_labelspacing=0.3, legend_fontsize=14.3, legend_handlelength=1.6, legend_handletextpad=0.5, layout_rect=(0.0, 0.04, 1.0, 0.98), title_fontsize=19, xlabel_fontsize=19, ylabel_fontsize=16, xtick_fontsize=19, ytick_fontsize=16, xtick_rotation=0, w_pad=0)
    def _case_id_of_row(scenario, lambda_case, omega_key):
        for case_id, cfg in CASE_CONFIG[scenario].items():
            if cfg['lambda_case'] == lambda_case and cfg['omega_key'] == omega_key:
                return case_id
        return np.nan
    summary = DF.groupby(['scenario', 'lambda_case', 'omega_key', 'n', 'method'], dropna=False).agg(runtime_mean=('runtime_others', 'mean'), runtime_std=('runtime_others', 'std'), memory_mean_gb=('fit_mem_gb', 'mean'), memory_std_gb=('fit_mem_gb', 'std')).reset_index()
    summary['case'] = summary.apply(lambda r: _case_id_of_row(r['scenario'], r['lambda_case'], r['omega_key']), axis=1)
    summary = summary.sort_values(['scenario', 'case', 'n', 'method']).reset_index(drop=True)
    runtime_table = summary[['scenario', 'case', 'n', 'method', 'runtime_mean', 'runtime_std']]
    memory_table = summary[['scenario', 'case', 'n', 'method', 'memory_mean_gb', 'memory_std_gb']]
    runtime_csv = OUTPUT_DIR / 'table_surrogate_runtime_by_case.csv'
    memory_csv = OUTPUT_DIR / 'table_surrogate_memory_by_case.csv'
    runtime_table.to_csv(runtime_csv, index=False)
    memory_table.to_csv(memory_csv, index=False)


if __name__ == "__main__":
    _run()
