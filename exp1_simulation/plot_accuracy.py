import argparse
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator


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
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 15
    EXP1_ROOT = Path('.')
    SAVE_ROOT = EXP1_ROOT / 'save' / 'accuracy'
    OUTPUT_DIR = EXP1_ROOT / 'figure' / 'accuracy'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PASTEL_PALETTE = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b']
    SCENARIO_ORDER = ['scenario1', 'scenario2']
    SCENARIO_ROW_LABEL = {'scenario1': 'Scenario 1', 'scenario2': 'Scenario 2'}
    CASE_ORDER = [1, 2, 3, 4]
    CASE_CONFIG = {'scenario1': {1: {'lambda_case': 'ones', 'omega_key': 0.7}, 2: {'lambda_case': 'ones', 'omega_key': 0.3}, 3: {'lambda_case': 'uniform_1_10', 'omega_key': 0.7}, 4: {'lambda_case': 'uniform_1_10', 'omega_key': 0.3}}, 'scenario2': {1: {'lambda_case': 'ones', 'omega_key': 'uniform_0.5_1'}, 2: {'lambda_case': 'ones', 'omega_key': 'uniform_0_0.5'}, 3: {'lambda_case': 'block_uniform_1_10', 'omega_key': 'uniform_0.5_1'}, 4: {'lambda_case': 'block_uniform_1_10', 'omega_key': 'uniform_0_0.5'}}}
    def _load_pickle(path):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        with open(path, 'rb') as f:
            return pickle.load(f)
    method_path = SAVE_ROOT / 'accuracy_all_greedy.pkl'
    shared_path = SAVE_ROOT / 'accuracy_all.pkl'
    payload = _load_pickle(method_path if method_path.exists() else shared_path)
    N_VALUES = list(payload['config']['n_grid'])
    N_COLORS = dict(zip(N_VALUES, PASTEL_PALETTE[:len(N_VALUES)]))
    ROW_LOOKUP = {}
    for scenario_name in SCENARIO_ORDER:
        for item in payload.get(scenario_name, {}).get('results', []):
            summary = item['result']['summary']
            omega_key = float(item['omega_bar']) if scenario_name == 'scenario1' else str(item['omega_case'])
            key = (scenario_name, str(item['lambda_case']), omega_key, int(item['n']))
            ROW_LOOKUP[key] = {'checkpoints': np.asarray(summary['checkpoints'], dtype=np.int64), 'are_curve_mean': np.asarray(summary['are_curve']['mean'], dtype=np.float64), 'are_curve_std': np.asarray(summary['are_curve']['std'], dtype=np.float64), 'unique_curve_mean': np.asarray(summary['unique_utility_eval_curve']['mean'], dtype=np.float64), 'unique_curve_std': np.asarray(summary['unique_utility_eval_curve']['std'], dtype=np.float64), 'aucc_mean': float(summary['aucc']['mean']), 'aucc_std': float(summary['aucc']['std']), 'runtime_mean': float(summary['non_utility_seconds']['mean']), 'runtime_std': float(summary['non_utility_seconds']['std'])}
    print('Save directory  :', SAVE_ROOT)
    print('Output directory:', OUTPUT_DIR)
    print('n values        :', N_VALUES)
    def _row(scenario, case_id, n):
        case_cfg = CASE_CONFIG[scenario][case_id]
        return ROW_LOOKUP.get((scenario, case_cfg['lambda_case'], case_cfg['omega_key'], n))
    def _n_label(n, prefix=True, pow2=False):
        """Format an n value as a legend label.

        If ``pow2`` is True and ``n`` is an integer power of 2, render as
        ``$n=2^{k}$`` (or ``$2^{k}$`` when ``prefix`` is False); otherwise fall
        back to the plain integer form.
        """
        if pow2 and n > 0:
            e = np.log2(n)
            if abs(e - round(e)) < 1e-09:
                ek = int(round(e))
                return f'$n=2^{{{ek}}}$' if prefix else f'$2^{{{ek}}}$'
        return f'$n={n}$' if prefix else str(n)
    def _n_legend_handles(prefix=True, pow2=False):
        return [Line2D([0], [0], color=N_COLORS[n], linewidth=2.2, label=_n_label(n, prefix=prefix, pow2=pow2)) for n in N_VALUES]
    def _draw_curve(ax, row, metric_key, color, band_alpha=0.12):
        x = row['checkpoints']
        y = np.asarray(row[metric_key + '_mean'], dtype=np.float64)
        y_sd = np.asarray(row[metric_key + '_std'], dtype=np.float64)
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, np.maximum(y - y_sd, 0.0), y + y_sd, color=color, alpha=band_alpha, linewidth=0.0)
        return (y, y_sd)
    def _apply_yscale(ax, logy, positive_values, all_values):
        if logy:
            ax.set_yscale('log')
            if positive_values:
                ymin = max(min(positive_values) * 0.8, np.finfo(np.float64).tiny)
                ymax = max(positive_values) * 1.25
                ax.set_ylim(ymin, ymax)
        elif all_values:
            ax.set_ylim(0.0, max(all_values) * 1.08)
    def _apply_xtick_rotation(ax, rotation):
        if rotation is None:
            return
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
    def _format_k(value, _pos):
        """Tick label formatter: 0 -> '0', 10000 -> '10\\mathrm{k}', etc."""
        if value == 0:
            return '0'
        if abs(value) >= 1000 and abs(value) % 1000 == 0:
            return f'${int(value // 1000)}\\mathrm{{k}}$'
        return f'{value:g}'
    def _apply_xtick_k(ax, xtick_values=None):
        """Force xticks to ``xtick_values`` and label them in '0 / 10k / 20k' style.

        If ``xtick_values`` is None, this is a no-op.
        """
        if xtick_values is None:
            return
        from matplotlib.ticker import FixedLocator
        ax.xaxis.set_major_locator(FixedLocator(list(xtick_values)))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
    def _apply_log10_exponent_ticks(ax):
        """For a log-scale y-axis, replace tick labels with the integer exponent.

        e.g. 10^{-1} -> '-1', 10^{-2} -> '-2'. Only labels integer powers of 10.
        """
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))

        def _fmt(value, _pos):
            if value <= 0:
                return ''
            e = np.log10(value)
            if abs(e - round(e)) < 1e-06:
                return f'{int(round(e))}'
            return ''
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ''))
    class _FixedOomFormatter(ScalarFormatter):
        """ScalarFormatter that pins the order-of-magnitude offset to a fixed exponent."""

        def __init__(self, order_of_magnitude, useMathText=True):
            super().__init__(useMathText=useMathText)
            self._fixed_oom = int(order_of_magnitude)
            self.set_scientific(True)

        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self._fixed_oom

        def _set_orderOfMagnitude(self, *args, **kwargs):
            self.orderOfMagnitude = self._fixed_oom
    def _apply_fixed_oom(ax, exponent, inset_box=False, inset_xy=(0.02, 0.97), inset_fontsize=None, inset_box_facecolor='white', inset_box_alpha=0.85, inset_box_edgecolor='none'):
        fmt = _FixedOomFormatter(exponent)
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(exponent, exponent))
        if inset_box:
            ax.yaxis.get_offset_text().set_visible(False)
            text_kwargs = dict(transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor=inset_box_facecolor, alpha=inset_box_alpha, edgecolor=inset_box_edgecolor, boxstyle='round,pad=0.25'))
            if inset_fontsize is not None:
                text_kwargs['fontsize'] = inset_fontsize
            ax.text(inset_xy[0], inset_xy[1], f'$\\times 10^{{{exponent}}}$', **text_kwargs)
    def plot_line_grid(metric_key, ylabel, output_name, logy=False, sharey=True, legend_bbox=(0.5, 0.01), layout_rect=(0.08, 0.08, 0.98, 0.95), supylabel_xy=(0.11, 0.55), y_offset_exponent=None, y_offset_inset_box=False, y_offset_inset_xy=(0.02, 0.97), y_offset_inset_fontsize=None, title_fontsize=None, xlabel_fontsize=None, ylabel_fontsize=None, supylabel_fontsize=17, xtick_fontsize=None, ytick_fontsize=None, xtick_rotation=None, xtick_values=None, legend_fontsize=None, legend_labelspacing=0.5):
        fig, axes = plt.subplots(2, 4, figsize=(15, 5), sharex=False, sharey=sharey)
        positive_values = []
        all_values = []
        for row_idx, scenario in enumerate(SCENARIO_ORDER):
            for col_idx, case_id in enumerate(CASE_ORDER):
                ax = axes[row_idx, col_idx]
                if row_idx == 0:
                    ax.set_title(f'Case {case_id}', fontsize=title_fontsize)
                if col_idx == 0:
                    ax.set_ylabel(SCENARIO_ROW_LABEL[scenario], rotation=90, labelpad=14, fontsize=ylabel_fontsize)
                for n in N_VALUES:
                    row = _row(scenario, case_id, n)
                    if row is None:
                        continue
                    y, y_sd = _draw_curve(ax, row, metric_key, N_COLORS[n])
                    positive_values.extend(y[y > 0].tolist())
                    all_values.extend((y + y_sd).tolist())
                ax.grid(alpha=0.2, axis='both')
                if row_idx == 1:
                    ax.set_xlabel('$m$', fontsize=xlabel_fontsize)
                else:
                    ax.set_xlabel('')
                    ax.tick_params(labelbottom=False)
                if col_idx != 0:
                    ax.set_ylabel('')
                    if sharey:
                        ax.tick_params(labelleft=False)
                if xtick_fontsize is not None:
                    ax.tick_params(axis='x', labelsize=xtick_fontsize)
                if ytick_fontsize is not None:
                    ax.tick_params(axis='y', labelsize=ytick_fontsize)
        if sharey:
            for ax in axes.flat:
                _apply_yscale(ax, logy, positive_values, all_values)
        elif logy:
            for ax in axes.flat:
                ax.set_yscale('log')
        if y_offset_exponent is not None and (not logy):
            for ax in axes.flat:
                _apply_fixed_oom(ax, y_offset_exponent, inset_box=y_offset_inset_box, inset_xy=y_offset_inset_xy, inset_fontsize=y_offset_inset_fontsize)
        for ax in axes.flat:
            _apply_xtick_k(ax, xtick_values)
            _apply_xtick_rotation(ax, xtick_rotation)
        leg_kwargs = dict(handles=_n_legend_handles(prefix=True), loc='lower center', ncol=len(N_VALUES), frameon=False, columnspacing=1.4, handlelength=2.0, labelspacing=legend_labelspacing, bbox_to_anchor=legend_bbox)
        if legend_fontsize is not None:
            leg_kwargs['fontsize'] = legend_fontsize
        fig.legend(**leg_kwargs)
        sup_kwargs = dict(x=supylabel_xy[0], y=supylabel_xy[1])
        if supylabel_fontsize is not None:
            sup_kwargs['fontsize'] = supylabel_fontsize
        fig.supylabel(ylabel, **sup_kwargs)
        fig.tight_layout(rect=layout_rect, w_pad=0.7, h_pad=0.8)
        out_path = OUTPUT_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)
        print('Saved', out_path)
    def _draw_legend_in_axes(target_ax, *, headroom_fraction, loc='upper right', bbox=(1.0, 1.0), borderaxespad=0.0, ncol=1, frameon=True, framealpha=0.9, edgecolor='0.4', labelspacing=0.3, fontsize=12, handlelength=1.6, handletextpad=0.5, columnspacing=2.0, legend_prefix=True, legend_pow2=False):
        """Add room at the top of `target_ax` and draw the n-curve legend inside it."""
        if target_ax.get_yscale() == 'log':
            ymin, ymax = target_ax.get_ylim()
            target_ax.set_ylim(ymin, ymax * (1.0 + headroom_fraction * 4))
        else:
            ymin, ymax = target_ax.get_ylim()
            span = max(ymax - ymin, 1e-12)
            target_ax.set_ylim(ymin, ymax + headroom_fraction * span)
        target_ax.legend(handles=_n_legend_handles(prefix=legend_prefix, pow2=legend_pow2), loc=loc, ncol=ncol, frameon=frameon, framealpha=framealpha, edgecolor=edgecolor, bbox_to_anchor=bbox, bbox_transform=target_ax.transAxes, borderaxespad=borderaxespad, labelspacing=labelspacing, handlelength=handlelength, handletextpad=handletextpad, columnspacing=columnspacing, fontsize=fontsize)
    def plot_short_scenario(scenario, case_id, output_name, figsize=(6, 3), are_ylabel='$\\log_{10}(\\mathrm{ARE}(m))$', unique_ylabel="Unique $S$'s", unique_y_offset_exponent=4, unique_y_offset_inset_xy=(0.03, 0.97), unique_y_offset_inset_fontsize=11, w_pad=0.5, legend_inside=True, legend_inside_panel=0, legend_inside_loc='upper right', legend_inside_bbox=(1.0, 1.0), legend_inside_borderaxespad=0.0, legend_inside_y_headroom=0.2, legend_inside_frameon=True, legend_inside_framealpha=0.9, legend_inside_edgecolor='0.4', legend_inside_labelspacing=0.3, legend_inside_ncol=1, legend_inside_columnspacing=2.0, legend_pow2=True, legend_fontsize=12, legend_handlelength=1.6, legend_handletextpad=0.5, layout_rect=(0.0, 0.04, 1.0, 0.94), suptitle_y=0.99, suptitle_fontsize=14, xlabel_fontsize=None, ylabel_fontsize=None, xtick_fontsize=None, ytick_fontsize=None, xtick_rotation=None, xtick_values=None):
        """1 x 2 figure for a fixed (scenario, case_id):
            [ARE, Unique-S].

        The figure-level suptitle is the scenario name (e.g. "Scenario 1");
        no per-panel titles are drawn.

        The left panel shows ARE on a log y-axis (tick labels are integer
        exponents, ylabel ``$\\log_{10}(\\mathrm{ARE}(m))$``). The right panel
        shows the Unique-subsets count on a linear y-axis with a fixed-exponent
        inset box (``unique_y_offset_exponent``).

        If ``legend_inside`` is True, the n-curve legend is drawn inside
        ``axes[legend_inside_panel]`` (default the ARE panel) at the top-right
        corner. ``legend_inside_ncol`` controls the number of legend columns
        (1 = vertical stack, len(N_VALUES) = single horizontal row, etc.).
        ``legend_inside_columnspacing`` adjusts the horizontal spacing between
        columns. When ``legend_pow2`` is True (default), n values are rendered
        as ``$n=2^{k}$``.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
        ax_are, ax_uniq = (axes[0], axes[1])
        fig.suptitle(SCENARIO_ROW_LABEL[scenario], y=suptitle_y, fontsize=suptitle_fontsize)
        are_pos, are_all = ([], [])
        uniq_pos, uniq_all = ([], [])
        for n in N_VALUES:
            row = _row(scenario, case_id, n)
            if row is None:
                continue
            y_a, y_a_sd = _draw_curve(ax_are, row, 'are_curve', N_COLORS[n])
            y_u, y_u_sd = _draw_curve(ax_uniq, row, 'unique_curve', N_COLORS[n])
            are_pos.extend(y_a[y_a > 0].tolist())
            are_all.extend((y_a + y_a_sd).tolist())
            uniq_pos.extend(y_u[y_u > 0].tolist())
            uniq_all.extend((y_u + y_u_sd).tolist())
        for ax in (ax_are, ax_uniq):
            ax.set_xlabel('$m$', fontsize=xlabel_fontsize)
            ax.grid(alpha=0.2, axis='both')
            if xtick_fontsize is not None:
                ax.tick_params(axis='x', labelsize=xtick_fontsize)
            if ytick_fontsize is not None:
                ax.tick_params(axis='y', labelsize=ytick_fontsize)
            _apply_xtick_k(ax, xtick_values)
            _apply_xtick_rotation(ax, xtick_rotation)
        ax_are.set_ylabel(are_ylabel, fontsize=ylabel_fontsize)
        ax_uniq.set_ylabel(unique_ylabel, fontsize=ylabel_fontsize)
        _apply_yscale(ax_are, True, are_pos, are_all)
        _apply_log10_exponent_ticks(ax_are)
        _apply_yscale(ax_uniq, False, uniq_pos, uniq_all)
        if unique_y_offset_exponent is not None:
            _apply_fixed_oom(ax_uniq, unique_y_offset_exponent, inset_box=True, inset_xy=unique_y_offset_inset_xy, inset_fontsize=unique_y_offset_inset_fontsize)
        if legend_inside:
            _draw_legend_in_axes(axes[legend_inside_panel % 2], headroom_fraction=legend_inside_y_headroom, loc=legend_inside_loc, bbox=legend_inside_bbox, borderaxespad=legend_inside_borderaxespad, ncol=legend_inside_ncol, columnspacing=legend_inside_columnspacing, frameon=legend_inside_frameon, framealpha=legend_inside_framealpha, edgecolor=legend_inside_edgecolor, labelspacing=legend_inside_labelspacing, fontsize=legend_fontsize, handlelength=legend_handlelength, handletextpad=legend_handletextpad, legend_pow2=legend_pow2)
        fig.tight_layout(rect=layout_rect, w_pad=w_pad)
        out_path = OUTPUT_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)
        print('Saved', out_path)
    plot_line_grid(metric_key='are_curve', ylabel='$\\mathrm{ARE}(m)$', output_name='accuracy_are_full.pdf', logy=True, sharey=True, legend_bbox=(0.57, 0.01), layout_rect=(0.08, 0.08, 0.98, 0.95), supylabel_xy=(0.11, 0.55), title_fontsize=16, xlabel_fontsize=14, ylabel_fontsize=14, supylabel_fontsize=17, xtick_fontsize=14, ytick_fontsize=14, xtick_rotation=0, xtick_values=[0, 10000, 20000], legend_fontsize=15, legend_labelspacing=0.5)
    plot_line_grid(metric_key='unique_curve', ylabel="Unique $S$'s", output_name='accuracy_unique_full.pdf', logy=False, sharey=False, legend_bbox=(0.57, 0.01), layout_rect=(0.08, 0.08, 0.98, 0.95), supylabel_xy=(0.11, 0.55), y_offset_exponent=6, y_offset_inset_box=False, y_offset_inset_xy=(0.02, 0.97), y_offset_inset_fontsize=12, title_fontsize=16, xlabel_fontsize=14, ylabel_fontsize=14, supylabel_fontsize=17, xtick_fontsize=14, ytick_fontsize=14, xtick_rotation=0, xtick_values=[0, 10000, 20000], legend_fontsize=15, legend_labelspacing=0.5)
    plot_short_scenario(scenario='scenario1', case_id=2, output_name='accuracy_short_scenario1.pdf', figsize=(8, 3.8), are_ylabel='$\\log_{10}(\\mathrm{ARE}(m))$', unique_ylabel="Unique $S$'s", unique_y_offset_exponent=4, unique_y_offset_inset_xy=(0.03, 0.95), unique_y_offset_inset_fontsize=15, w_pad=-0, legend_inside=True, legend_inside_panel=0, legend_inside_loc='upper right', legend_inside_bbox=(1.0, 1.0), legend_inside_borderaxespad=0.0, legend_inside_y_headroom=0.2, legend_inside_frameon=True, legend_inside_framealpha=0.9, legend_inside_edgecolor='0.4', legend_inside_labelspacing=-0.01, legend_inside_ncol=2, legend_inside_columnspacing=0.01, legend_fontsize=16, legend_handlelength=0.6, legend_handletextpad=0.2, layout_rect=(0.0, 0.04, 1.0, 0.94), suptitle_y=0.82, suptitle_fontsize=20, xlabel_fontsize=18, ylabel_fontsize=18, xtick_fontsize=18, ytick_fontsize=18, xtick_rotation=0, xtick_values=[0, 10000, 20000])
    plot_short_scenario(scenario='scenario2', case_id=2, output_name='accuracy_short_scenario2.pdf', figsize=(8, 3.8), are_ylabel='$\\log_{10}(\\mathrm{ARE}(m))$', unique_ylabel="Unique $S$'s", unique_y_offset_exponent=6, unique_y_offset_inset_xy=(0.03, 0.95), unique_y_offset_inset_fontsize=15, w_pad=-0.7, legend_inside=False, legend_inside_ncol=1, legend_inside_columnspacing=2.0, layout_rect=(0.0, 0.04, 1.0, 0.94), suptitle_y=0.82, suptitle_fontsize=20, xlabel_fontsize=18, ylabel_fontsize=18, xtick_fontsize=18, ytick_fontsize=18, xtick_rotation=0, xtick_values=[0, 10000, 20000])
    records = []
    for scenario in SCENARIO_ORDER:
        for case_id in CASE_ORDER:
            for n in N_VALUES:
                row = _row(scenario, case_id, n)
                if row is None:
                    continue
                records.append({'scenario': scenario, 'scenario_label': SCENARIO_ROW_LABEL[scenario], 'case': case_id, 'n': int(n), 'aucc_mean': row['aucc_mean'], 'aucc_std': row['aucc_std'], 'runtime_mean': row['runtime_mean'], 'runtime_std': row['runtime_std']})
    table_df = pd.DataFrame(records)
    aucc_csv = OUTPUT_DIR / 'table1_aucc_by_case_scenario.csv'
    runtime_csv = OUTPUT_DIR / 'table2_runtime_by_case_scenario.csv'
    table_df[['scenario', 'scenario_label', 'case', 'n', 'aucc_mean', 'aucc_std']].to_csv(aucc_csv, index=False)
    table_df[['scenario', 'scenario_label', 'case', 'n', 'runtime_mean', 'runtime_std']].to_csv(runtime_csv, index=False)
    print('Saved:', aucc_csv)
    print('Saved:', runtime_csv)


if __name__ == "__main__":
    _run()
