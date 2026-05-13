import argparse
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 15
    EXP1_ROOT = Path('.')
    SAVE_TAG = 'default'
    SAVE_PATH = EXP1_ROOT / 'save' / 'sweep' / f'sweep_{SAVE_TAG}.pkl'
    OUTPUT_DIR = EXP1_ROOT / 'figure' / 'sweep'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PASTEL_PALETTE_5 = ['#f768a1', '#dd3497', '#ae017e', '#7a0177', '#49006a', 'black']
    LEGEND_TITLE_BETA = '$\\beta$:'
    def _beta_label(b):
        bf = float(b)
        return f'$\\beta={int(bf)}$' if bf.is_integer() else f'$\\beta={bf:g}$'
    def _beta_short_label(b):
        bf = float(b)
        return f'$\\beta={int(bf)}$' if bf.is_integer() else f'$\\beta={bf:g}$'
    def _build_method_style(beta_grid):
        method_order = [('gpasv', float(b)) for b in beta_grid] + [('pasv', None)]
        palette = list(PASTEL_PALETTE_5)
        if len(method_order) > len(palette):
            extra = sns.color_palette('pastel', len(method_order)).as_hex()
            palette = extra[:len(method_order) - 1] + [palette[-1]]
        style = {}
        for idx, mk in enumerate(method_order):
            method, beta = mk
            if method == 'pasv':
                label = '$\\beta\\to\\infty$ (PASV)'
                short = '\n'.join(['$\\beta\\to\\infty$', '(PASV)'])
                ls = 'dashed'
            else:
                label, short, ls = (_beta_label(beta), _beta_short_label(beta), 'solid')
            style[mk] = {'label': label, 'short_label': short, 'color': palette[idx], 'linestyle': ls, 'zorder': 3 + idx}
        return (method_order, style)
    def _load_pickle(path):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        with open(path, 'rb') as f:
            return pickle.load(f)
    PAYLOAD = _load_pickle(SAVE_PATH)
    CONFIG = PAYLOAD['config']
    RESULTS = PAYLOAD['results']
    N_VALUE = int(CONFIG['n'])
    GROUP_SIZE = int(CONFIG['group_size'])
    P_GRID = [float(p) for p in CONFIG['p_grid']]
    LAM_BAR_GRID = [float(x) for x in CONFIG['lam_bar_grid']]
    BETA_GRID = [float(x) for x in CONFIG['beta_grid']]
    METHOD_ORDER, METHOD_STYLE = _build_method_style(BETA_GRID)
    ROWS = []
    for _, r in RESULTS.items():
        ROWS.append({'method': str(r['method']), 'p': float(r['p']), 'beta': float(r['beta']), 'lam_bar': float(r['lam_bar']), 'rbar_G_mean': float(r['rbar_G_mean_rep_mean']), 'rbar_G_sd': float(r['rbar_G_mean_rep_sd']), 'within_sd_G_mean': float(r['within_sd_G_mean_rep_mean']), 'within_sd_G_sd': float(r['within_sd_G_mean_rep_sd']), 'value_sum_unanimity_G_mean': float(r.get('value_sum_unanimity_G_mean_rep_mean', np.nan)), 'value_sum_unanimity_G_sd': float(r.get('value_sum_unanimity_G_mean_rep_sd', np.nan)), 'value_sum_race_G_mean': float(r.get('value_sum_race_G_mean_rep_mean', np.nan)), 'value_sum_race_G_sd': float(r.get('value_sum_race_G_mean_rep_sd', np.nan))})
    DF = pd.DataFrame(ROWS).sort_values(['method', 'p', 'beta', 'lam_bar']).reset_index(drop=True)
    print('Save path       :', SAVE_PATH)
    print('Output directory:', OUTPUT_DIR)
    print(f'n = {N_VALUE}  (|G| = {GROUP_SIZE})')
    print('p grid          :', P_GRID)
    print('lam_bar grid    :', LAM_BAR_GRID)
    print('beta grid       :', BETA_GRID)
    print('rows            :', len(DF))
    def _curves(df, mean_key, sd_key, p):
        """Return {method_key: (x_log2, mean, sd)} for one (df, panel, p)."""
        out = {}
        x_log2 = np.log2(np.asarray(LAM_BAR_GRID, dtype=np.float64))
        for mk in METHOD_ORDER:
            method, beta = mk
            means, sds = ([], [])
            for lam_bar in LAM_BAR_GRID:
                mask = (df['method'] == method) & (df['p'] == float(p)) & (df['lam_bar'] == float(lam_bar))
                if method == 'pasv':
                    mask = mask & df['beta'].isna()
                else:
                    mask = mask & (df['beta'] == float(beta))
                sub = df.loc[mask]
                if len(sub) == 0:
                    means.append(np.nan)
                    sds.append(np.nan)
                else:
                    means.append(float(sub.iloc[0][mean_key]))
                    sds.append(float(sub.iloc[0][sd_key]))
            out[mk] = (x_log2, np.asarray(means, dtype=np.float64), np.asarray(sds, dtype=np.float64))
        return out
    def _draw_methods(ax, curves, line_width=2.2, pasv_line_width=1.2, marker_size=5.0, band_alpha=0.28, line_alpha=1.0):
        """Plot all methods with their styled bands; return (lo, hi) of band extent."""
        lo, hi = (np.inf, -np.inf)
        for mk in METHOD_ORDER:
            x, m, s = curves[mk]
            style = METHOD_STYLE[mk]
            is_pasv = mk[0] == 'pasv'
            ax.plot(x, m, color=style['color'], linewidth=pasv_line_width if is_pasv else line_width, linestyle=style['linestyle'], marker='o', markersize=marker_size, markeredgecolor='white', markeredgewidth=0.7, alpha=line_alpha, zorder=style['zorder'] + 1)
            ax.fill_between(x, m - s, m + s, color=style['color'], alpha=band_alpha, linewidth=0.0, zorder=style['zorder'])
            finite = np.isfinite(m) & np.isfinite(s)
            if finite.any():
                lo = min(lo, float(np.min((m - s)[finite])))
                hi = max(hi, float(np.max((m + s)[finite])))
        return (lo, hi)
    def _setup_xaxis(ax, xlabel_fontsize=None, xtick_fontsize=12, show_xlabel=True):
        ax.set_xticks(np.log2(np.asarray(LAM_BAR_GRID, dtype=np.float64)))
        ax.set_xticklabels([f'$2^{{{int(np.log2(x))}}}$' for x in LAM_BAR_GRID], fontsize=xtick_fontsize)
        if show_xlabel:
            ax.set_xlabel('$\\lambda_0$', fontsize=xlabel_fontsize)
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
    def _legend_handles(label_field='label'):
        """Return Line2D handles using either 'label' or 'short_label' from METHOD_STYLE."""
        return [Line2D([0], [0], color=METHOD_STYLE[mk]['color'], linewidth=2.6, linestyle=METHOD_STYLE[mk]['linestyle'], label=METHOD_STYLE[mk][label_field]) for mk in METHOD_ORDER]
    def _legend_handles_full():
        return _legend_handles('label')
    def _legend_handles_short():
        return _legend_handles('short_label')
    def _add_inline_title_legend(target, handles, legend_title=LEGEND_TITLE_BETA, legend_title_fontsize=None, **legend_kwargs):

        if legend_title:
            title_handle = Line2D([0], [0], linestyle='', marker='', label=legend_title)
            all_handles = [title_handle] + list(handles)
        else:
            all_handles = list(handles)
        leg = target.legend(handles=all_handles, **legend_kwargs)
        if legend_title and legend_title_fontsize is not None:
            leg.get_texts()[0].set_fontsize(legend_title_fontsize)
        if legend_title:
            leg.legend_handles[0].set_visible(False)
        return leg
    def _add_legend_hline(fig, ax, leg, y_frac, color='0.4', linewidth=0.8, linestyle=(0, (3, 2)), x_inset=0.0):

        if y_frac is None:
            return None
        fig.canvas.draw()
        bbox = leg.get_window_extent()
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        width = x1 - x0
        xa = x0 + x_inset * width
        xb = x1 - x_inset * width
        yy = y0 + y_frac * (y1 - y0)
        line = plt.Line2D([xa, xb], [yy, yy], color=color, linewidth=linewidth, linestyle=linestyle, transform=fig.transFigure, zorder=leg.get_zorder() + 1, clip_on=False)
        fig.add_artist(line)
        return line
    ROW_SPEC = [{'ylabel': 'Mean Rank', 'mean_key': 'rbar_G_mean', 'sd_key': 'rbar_G_sd', 'midline': lambda: (N_VALUE + 1) / 2.0}, {'ylabel': 'Value-Sum (SOU)', 'mean_key': 'value_sum_unanimity_G_mean', 'sd_key': 'value_sum_unanimity_G_sd', 'midline': None}, {'ylabel': 'Value-Sum (SOR)', 'mean_key': 'value_sum_race_G_mean', 'sd_key': 'value_sum_race_G_sd', 'midline': None}]
    def plot_full_grid(output_name, figsize=None, legend_bbox=(0.5, 0.04), layout_rect=(0.04, 0.07, 0.99, 0.97), title_fontsize=None, ylabel_fontsize=None, ylabel_pad=8, xlabel_fontsize=None, xtick_fontsize=12, ytick_fontsize=None, legend_fontsize=None, legend_title=LEGEND_TITLE_BETA, legend_title_fontsize=None, line_width=2.2, pasv_line_width=1.2, marker_size=5.0, band_alpha=0.28, line_alpha=1.0):
        """3 x len(P_GRID) grid: one row per ROW_SPEC entry, columns = p values."""
        nrow = len(ROW_SPEC)
        ncol = len(P_GRID)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (3.3 * ncol, 3.0 * nrow), sharex=False, sharey='row')
        row_lo = [np.inf] * nrow
        row_hi = [-np.inf] * nrow
        for row_idx, spec in enumerate(ROW_SPEC):
            for col_idx, p in enumerate(P_GRID):
                ax = axes[row_idx, col_idx]
                if row_idx == 0:
                    ax.set_title(f'$p_{{\\mathrm{{edge}}}}={p}$', fontsize=title_fontsize)
                if col_idx == 0:
                    ax.set_ylabel(spec['ylabel'], rotation=90, labelpad=ylabel_pad, fontsize=ylabel_fontsize)
                curves = _curves(DF, spec['mean_key'], spec['sd_key'], p)
                lo, hi = _draw_methods(ax, curves, line_width=line_width, pasv_line_width=pasv_line_width, marker_size=marker_size, band_alpha=band_alpha, line_alpha=line_alpha)
                row_lo[row_idx] = min(row_lo[row_idx], lo)
                row_hi[row_idx] = max(row_hi[row_idx], hi)
                if spec['midline'] is not None:
                    ax.axhline(spec['midline'](), color='0.55', linestyle=(0, (2, 2)), linewidth=1.1, zorder=1)
                ax.grid(alpha=0.2, axis='both')
                _setup_xaxis(ax, xlabel_fontsize=xlabel_fontsize, xtick_fontsize=xtick_fontsize, show_xlabel=row_idx == nrow - 1)
                if col_idx != 0:
                    ax.tick_params(labelleft=False)
                if ytick_fontsize is not None:
                    ax.tick_params(axis='y', labelsize=ytick_fontsize)
        for row_idx in range(nrow):
            lo, hi = (row_lo[row_idx], row_hi[row_idx])
            if np.isfinite(lo) and np.isfinite(hi):
                span = max(hi - lo, 1e-06)
                pad = 0.06 * span
                for col_idx in range(ncol):
                    axes[row_idx, col_idx].set_ylim(lo - pad, hi + pad)
        fig.align_ylabels(axes[:, 0])
        n_entries = len(METHOD_ORDER) + (1 if legend_title else 0)
        leg_kwargs = dict(loc='lower center', ncol=n_entries, frameon=False, columnspacing=1.6, handlelength=2.4, handletextpad=0.5, bbox_to_anchor=legend_bbox)
        if legend_fontsize is not None:
            leg_kwargs['fontsize'] = legend_fontsize
        _add_inline_title_legend(fig, _legend_handles_full(), legend_title=legend_title, legend_title_fontsize=legend_title_fontsize, **leg_kwargs)
        fig.tight_layout(rect=layout_rect, w_pad=0.8, h_pad=0.9)
        out_path = OUTPUT_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)
        print('Saved', out_path)
    def plot_short_triptych(output_name, p=0.5, figsize=None, legend_inside=True, legend_inside_panel=-1, legend_inside_loc='upper right', legend_inside_bbox=(0.98, 0.98), legend_inside_ncol=1, legend_inside_y_headroom=0.18, legend_inside_frameon=True, legend_inside_framealpha=0.9, legend_inside_edgecolor='0.4', legend_inside_borderaxespad=0.0, legend_inside_labelspacing=0.3, legend_bbox=(0.5, 0.03), layout_rect=(0.0, 0.08, 1.0, 0.98), ylabel_fontsize=None, xlabel_fontsize=None, xtick_fontsize=12, ytick_fontsize=None, legend_fontsize=None, legend_title=LEGEND_TITLE_BETA, legend_title_fontsize=None, line_width=2.2, pasv_line_width=1.2, marker_size=5.0, band_alpha=0.28, line_alpha=1.0, legend_hline_y=None, legend_hline_color='0.4', legend_hline_width=0.8, legend_hline_linestyle=(0, (3, 2)), legend_hline_x_inset=0.0):

        fig, axes = plt.subplots(1, len(ROW_SPEC), figsize=figsize or (2.5 * len(ROW_SPEC), 3.2), sharex=True, sharey=False)
        inside_panel_idx = legend_inside_panel % len(ROW_SPEC) if legend_inside else None
        for col_idx, spec in enumerate(ROW_SPEC):
            ax = axes[col_idx]
            ax.set_ylabel(spec['ylabel'], fontsize=ylabel_fontsize)
            curves = _curves(DF, spec['mean_key'], spec['sd_key'], p)
            lo, hi = _draw_methods(ax, curves, line_width=line_width, pasv_line_width=pasv_line_width, marker_size=marker_size, band_alpha=band_alpha, line_alpha=line_alpha)
            if spec['midline'] is not None:
                ax.axhline(spec['midline'](), color='0.55', linestyle=(0, (2, 2)), linewidth=1.1, zorder=1)
            if np.isfinite(lo) and np.isfinite(hi):
                span = max(hi - lo, 1e-06)
                pad = 0.06 * span
                top_pad = pad
                if inside_panel_idx is not None and col_idx == inside_panel_idx:
                    top_pad = max(pad, legend_inside_y_headroom * span)
                ax.set_ylim(lo - pad, hi + top_pad)
            ax.grid(alpha=0.2, axis='both')
            _setup_xaxis(ax, xlabel_fontsize=xlabel_fontsize, xtick_fontsize=xtick_fontsize, show_xlabel=True)
            if ytick_fontsize is not None:
                ax.tick_params(axis='y', labelsize=ytick_fontsize)
        leg = None
        target_ax = None
        if legend_inside:
            target_ax = axes[inside_panel_idx]
            leg_kwargs = dict(loc=legend_inside_loc, ncol=legend_inside_ncol, frameon=legend_inside_frameon, framealpha=legend_inside_framealpha, edgecolor=legend_inside_edgecolor, columnspacing=1.0, handlelength=2.2, handletextpad=0.5, labelspacing=legend_inside_labelspacing, borderaxespad=legend_inside_borderaxespad, bbox_to_anchor=legend_inside_bbox, bbox_transform=target_ax.transAxes)
            if legend_fontsize is not None:
                leg_kwargs['fontsize'] = legend_fontsize
            leg = _add_inline_title_legend(target_ax, _legend_handles_short(), legend_title=legend_title, legend_title_fontsize=legend_title_fontsize, **leg_kwargs)
        else:
            n_entries = len(METHOD_ORDER) + (1 if legend_title else 0)
            leg_kwargs = dict(loc='lower center', ncol=n_entries, frameon=False, columnspacing=1.6, handlelength=2.4, handletextpad=0.5, bbox_to_anchor=legend_bbox)
            if legend_fontsize is not None:
                leg_kwargs['fontsize'] = legend_fontsize
            leg = _add_inline_title_legend(fig, _legend_handles_short(), legend_title=legend_title, legend_title_fontsize=legend_title_fontsize, **leg_kwargs)
        fig.tight_layout(rect=layout_rect, w_pad=1.0)
        if legend_hline_y is not None and leg is not None:
            _add_legend_hline(fig, target_ax if target_ax is not None else fig, leg, legend_hline_y, color=legend_hline_color, linewidth=legend_hline_width, linestyle=legend_hline_linestyle, x_inset=legend_hline_x_inset)
        out_path = OUTPUT_DIR / output_name
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)
        print('Saved', out_path)
    plot_full_grid(output_name='sweep_full.pdf', figsize=(4 * len(P_GRID), 3.0 * len(ROW_SPEC)), legend_bbox=(0.5, 0.04), layout_rect=(0.04, 0.07, 0.99, 0.97), title_fontsize=16, ylabel_fontsize=14, ylabel_pad=8, xlabel_fontsize=14, xtick_fontsize=12, ytick_fontsize=14, legend_fontsize=15, legend_title_fontsize=15, line_width=2.2, pasv_line_width=1.2, marker_size=5.0, band_alpha=0.28, line_alpha=1.0)
    plot_short_triptych(output_name='sweep_short.pdf', p=0.5, figsize=(2.8 * len(ROW_SPEC), 3), legend_inside=True, legend_inside_panel=-1, legend_inside_loc='upper right', legend_inside_bbox=(1.0, 1.0), legend_inside_borderaxespad=0.0, legend_inside_ncol=1, legend_inside_y_headroom=0.2, legend_inside_frameon=True, legend_inside_framealpha=0.9, legend_inside_edgecolor='0.4', legend_inside_labelspacing=0.1, layout_rect=(0.0, 0.04, 1.0, 0.98), ylabel_fontsize=14, xlabel_fontsize=14, xtick_fontsize=12, ytick_fontsize=12, legend_fontsize=11.7, legend_title=None, line_width=1.7, pasv_line_width=1.2, marker_size=3.5, band_alpha=0.7, line_alpha=1.0, legend_hline_y=0.395, legend_hline_color='0.4', legend_hline_width=0.9, legend_hline_linestyle=(0, (3, 2)), legend_hline_x_inset=0.0)


if __name__ == "__main__":
    _run()
