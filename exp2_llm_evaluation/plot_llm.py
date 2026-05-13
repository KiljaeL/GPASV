# ========= data options (pickle layout) =========
USE_SNIS = True             # True -> *_snis.pkl, False -> *.pkl (--no-ess-reuse)
SAVE_ROOT = 'save/value'
QUESTIONS = list(range(1, 81))

# Sweep grid; must match what main_value.py was run with.
SWEEP_VALUES = [0, 1, 2, 4, 8, 16, 32]

# Proprietary / paid API players (the rest are open-source).
PAID_API_PLAYERS = [
    'gpt-4', 'gpt-3.5-turbo', 'claude-v1', 'claude-instant-v1', 'palm-2',
]

# Output dir for PDFs/CSVs.
from pathlib import Path
PLOT_SUBDIR = Path('figure')
PLOT_SUBDIR.mkdir(parents=True, exist_ok=True)
print('Output dir:', PLOT_SUBDIR.resolve())

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='white')

PAID_SET = set(PAID_API_PLAYERS)
_save_root = Path(SAVE_ROOT)
_suffix = '_snis' if USE_SNIS else ''

def model_group(model: str) -> str:
    return 'paid' if model in PAID_SET else 'open-source'


def _capitalize_group(group: str) -> str:
    return group[:1].upper() + group[1:]


def _format_pow2_tick(value: float) -> str:
    if value <= 0:
        return '0'
    log2_val = math.log2(value)
    if abs(log2_val - round(log2_val)) < 1e-9:
        exponent = int(round(log2_val))
        return rf'$2^{{{exponent}}}$'
    return f'{value:g}'


def _format_param_tag(alpha: float, beta: float) -> str:
    return f'a{alpha:g}_b{beta:g}'.replace('/', '_')


def _stored_sweep_for_case(alpha: float, beta: float):
    a, b = float(alpha), float(beta)
    if a == 0 and b == 0:
        return 'alpha', 0.0, 0.0
    if b == 0:
        return 'alpha', a, 0.0
    if a == 0:
        return 'beta', 0.0, b
    if a == b:
        return 'both', a, b
    raise ValueError(f'No stored sweep for off-path case (alpha,beta)=({alpha},{beta}).')


def _value_pickle_path(q_one_based: int, sweep_name: str, alpha: float, beta: float) -> Path:
    tag = _format_param_tag(alpha, beta)
    return _save_root / str(q_one_based) / f'{sweep_name}_{tag}{_suffix}.pkl'


def _load_final_value_row(q_one_based: int, sweep_name: str, alpha: float, beta: float) -> pd.Series:
    path = _value_pickle_path(q_one_based, sweep_name, alpha, beta)
    if not path.exists():
        raise FileNotFoundError(f'Missing value pickle: {path}')
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f'Invalid or empty value trajectory: {path}')
    return df.iloc[-1].astype(float)


def _mean_value_across_questions(sweep_name: str, alpha: float, beta: float) -> pd.Series:
    rows = [_load_final_value_row(q, sweep_name, alpha, beta) for q in QUESTIONS]
    return pd.concat(rows, axis=1).mean(axis=1)


def mean_value_for_case(alpha: float, beta: float) -> pd.Series:
    """Mean (across QUESTIONS) Shapley value vector for the case (alpha, beta)."""
    sweep_name, aa, bb = _stored_sweep_for_case(alpha, beta)
    return _mean_value_across_questions(sweep_name, aa, bb)


def texttt_label(text: str) -> str:
    escaped = (
        str(text)
        .replace('\\', r'\\')
        .replace('_', r'\_')
        .replace('-', r'{-}')
        .replace('.', r'{.}')
    )
    return rf'$\mathtt{{{escaped}}}$'

# Figure 8: top-K bar grid (1 baseline + 6 priority regimes).

DEFAULT_GROUP_COLORS = {
    'paid':        '#a1c9f4',
    'open-source': '#ffb482',
}


def _draw_topk_bar(ax, values: pd.Series, title: str, *,
                   top_k: int, group_colors: dict,
                   title_fontsize: float, axis_label_fontsize: float,
                   tick_label_fontsize: float, value_label_fontsize: float,
                   model_label_fontsize: float,
                   show_xlabel: bool, xlabel: str):
    values = values.sort_values(ascending=False).head(top_k).sort_values(ascending=True)
    y = np.arange(len(values))
    colors = [group_colors[model_group(m)] for m in values.index]
    ax.barh(y, values.values, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([texttt_label(m) for m in values.index], fontsize=model_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.set_xlabel(xlabel if show_xlabel else '', fontsize=axis_label_fontsize)
    ax.tick_params(axis='x', labelsize=tick_label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)
    ax.axvline(0, color='0.25', linewidth=0.8)

    xmin = min(0.0, float(values.min()))
    xmax = max(0.0, float(values.max()))
    span = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(xmin - 0.05 * span, xmax + 0.24 * span)
    for yi, val in zip(y, values.values):
        if val >= 0:
            ax.text(val + 0.025 * span, yi, f'{val:.2f}', va='center', ha='left',
                    fontsize=value_label_fontsize)
        else:
            ax.text(val - 0.025 * span, yi, f'{val:.2f}', va='center', ha='right',
                    fontsize=value_label_fontsize)
    sns.despine(ax=ax)


def plot_topk_grid(
    output_name: str = '8_topk.pdf',
    *,
    weak_strength: Optional[float] = None,
    strong_strength: Optional[float] = None,
    top_k: int = 8,
    figsize: Tuple[float, float] = (16, 7),
    title_alpha: str = 'Open-Source Prior',
    title_beta: str = 'Graph-Structure Prior',
    title_joint: str = 'Joint Prior',
    title_baseline: str = 'No Prior (SV)',
    show_param_in_title: bool = True,
    strong_row_title_mode: str = 'param_only',
    title_fontsize: float = 19,
    axis_label_fontsize: float = 16,
    tick_label_fontsize: float = 13,
    value_label_fontsize: float = 12,
    model_label_fontsize: float = 10,
    legend_fontsize: float = 15,
    xlabel: str = 'GPASV',
    width_ratios: Tuple[float, float, float, float] = (1.15, 1.0, 1.0, 1.0),
    wspace: float = 1.18,
    hspace: float = 0.34,
    subplots_adjust: dict = None,
    # ---- Legend placement ----
    # 'inside_baseline' : draw legend inside the baseline (leftmost) panel.
    # 'figure'          : figure-level legend; loc/bbox in *figure* coordinates
    #                     (e.g. loc='upper left', bbox=(0.01, 0.99) for top-left).
    legend_mode: str = 'inside_baseline',
    legend_loc: str = 'upper left',
    legend_bbox: Tuple[float, float] = (0.02, 0.98),
    legend_borderaxespad: float = 0.0,
    legend_ncol: int = 1,
    legend_frameon: bool = True,
    legend_framealpha: float = 0.9,
    legend_edgecolor: str = '0.4',
    legend_labelspacing: float = 0.3,
    legend_handlelength: float = 1.4,
    legend_handletextpad: float = 0.5,
    legend_columnspacing: float = 1.5,
    group_colors: Optional[dict] = None,
    save_csv: bool = True,
):
    """Draw the 2x4 top-K grid: baseline + weak/strong x (alpha, beta, joint).

    Per-panel titles use ``title_*`` plus an optional regime tag.
    See ``strong_row_title_mode`` and ``legend_mode`` for layout knobs.
    """
    if group_colors is None:
        group_colors = DEFAULT_GROUP_COLORS
    if strong_row_title_mode not in ('full', 'param_only', 'none'):
        raise ValueError("strong_row_title_mode must be 'full', 'param_only', or 'none'.")
    if legend_mode not in ('inside_baseline', 'figure'):
        raise ValueError("legend_mode must be 'inside_baseline' or 'figure'.")

    positive_sweep = [v for v in SWEEP_VALUES if v > 0]
    if not positive_sweep:
        raise ValueError('SWEEP_VALUES must include at least one positive strength.')
    weak = float(weak_strength if weak_strength is not None else (1 if 1 in positive_sweep else min(positive_sweep)))
    strong = float(strong_strength if strong_strength is not None else (32 if 32 in set(SWEEP_VALUES) else max(SWEEP_VALUES)))

    def _ptag(a: float, b: float) -> str:
        if not show_param_in_title:
            return ''
        return rf'$(\alpha,\beta)=({a:g},{b:g})$'

    def _title(base: str, a: float, b: float) -> str:
        tag = _ptag(a, b)
        if not tag:
            return base
        return f'{base}\n{tag}'

    def _strong_title(base: str, a: float, b: float) -> str:
        if strong_row_title_mode == 'none':
            return ''
        if strong_row_title_mode == 'param_only':
            return _ptag(a, b)
        return _title(base, a, b)

    cases = [
        {'slot': 'baseline',     'alpha': 0.0,    'beta': 0.0,    'title': _title(title_baseline, 0.0, 0.0)},
        {'slot': 'weak_alpha',   'alpha': weak,   'beta': 0.0,    'title': _title(title_alpha, weak, 0.0)},
        {'slot': 'weak_beta',    'alpha': 0.0,    'beta': weak,   'title': _title(title_beta, 0.0, weak)},
        {'slot': 'weak_joint',   'alpha': weak,   'beta': weak,   'title': _title(title_joint, weak, weak)},
        {'slot': 'strong_alpha', 'alpha': strong, 'beta': 0.0,    'title': _strong_title(title_alpha, strong, 0.0)},
        {'slot': 'strong_beta',  'alpha': 0.0,    'beta': strong, 'title': _strong_title(title_beta, 0.0, strong)},
        {'slot': 'strong_joint', 'alpha': strong, 'beta': strong, 'title': _strong_title(title_joint, strong, strong)},
    ]

    # Compute regime vectors (and optional CSV).
    regime_vectors = {}
    records = []
    for case in cases:
        a, b = case['alpha'], case['beta']
        values = mean_value_for_case(a, b)
        ranks = values.rank(ascending=False, method='min').astype(int)
        regime_vectors[(a, b)] = values
        for model, value in values.items():
            records.append({
                'slot': case['slot'], 'alpha': a, 'beta': b,
                'model': model, 'group': model_group(model),
                'value': float(value), 'rank': int(ranks[model]),
            })
    if save_csv:
        csv_path = PLOT_SUBDIR / (Path(output_name).stem + '.csv')
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print('CSV saved:', csv_path)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, width_ratios=list(width_ratios), wspace=wspace, hspace=hspace)
    axes_by_slot = {
        'baseline':     fig.add_subplot(gs[:, 0]),
        'weak_alpha':   fig.add_subplot(gs[0, 1]),
        'weak_beta':    fig.add_subplot(gs[0, 2]),
        'weak_joint':   fig.add_subplot(gs[0, 3]),
        'strong_alpha': fig.add_subplot(gs[1, 1]),
        'strong_beta':  fig.add_subplot(gs[1, 2]),
        'strong_joint': fig.add_subplot(gs[1, 3]),
    }
    for case in cases:
        slot = case['slot']
        show_xlabel = not slot.startswith('weak_')
        _draw_topk_bar(
            axes_by_slot[slot], regime_vectors[(case['alpha'], case['beta'])],
            case['title'], top_k=top_k, group_colors=group_colors,
            title_fontsize=title_fontsize, axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize, value_label_fontsize=value_label_fontsize,
            model_label_fontsize=model_label_fontsize,
            show_xlabel=show_xlabel, xlabel=xlabel,
        )

    sa = subplots_adjust if subplots_adjust is not None else dict(
        left=0.06, right=0.99, top=0.92, bottom=0.16, wspace=0.82, hspace=0.38,
    )
    fig.subplots_adjust(**sa)
    # Recenter the baseline panel vertically between the two rows.
    baseline_ax = axes_by_slot['baseline']
    row_ax = axes_by_slot['weak_alpha']
    base_pos = baseline_ax.get_position()
    row_pos = row_ax.get_position()
    baseline_ax.set_position([
        base_pos.x0,
        base_pos.y0 + 0.5 * (base_pos.height - row_pos.height),
        base_pos.width,
        row_pos.height,
    ])

    legend_handles = [
        Patch(facecolor=group_colors[g], edgecolor='white', label=_capitalize_group(g))
        for g in ['paid', 'open-source']
    ]
    if legend_mode == 'inside_baseline':
        leg = baseline_ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            bbox_transform=baseline_ax.transAxes,
            ncol=legend_ncol,
            frameon=legend_frameon,
            framealpha=legend_framealpha,
            edgecolor=legend_edgecolor,
            borderaxespad=legend_borderaxespad,
            labelspacing=legend_labelspacing,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            columnspacing=legend_columnspacing,
            prop={'size': legend_fontsize},
        )
        baseline_ax.add_artist(leg)
    else:  # 'figure'
        leg = fig.legend(
            handles=legend_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncol,
            frameon=legend_frameon,
            framealpha=legend_framealpha,
            edgecolor=legend_edgecolor,
            borderaxespad=legend_borderaxespad,
            labelspacing=legend_labelspacing,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            columnspacing=legend_columnspacing,
            prop={'size': legend_fontsize},
        )
    out_path = PLOT_SUBDIR / output_name
    fig.savefig(out_path, bbox_inches='tight', bbox_extra_artists=(leg,))
    plt.show()
    plt.close(fig)
    print('Saved', out_path)
    return out_path


# Figure 9: 1x3 group-sum sweeps over alpha, beta, alpha=beta.

DEFAULT_GROUP_STYLES = {
    'paid':        '--',
    'open-source': '-',
}


def plot_group_sum_sweep(
    output_name: str = '9_group_sum_sweep.pdf',
    *,
    title_alpha: str = r'Sweeping $\alpha$ ($\beta=0$)',
    title_beta: str = r'Sweeping $\beta$ ($\alpha=0$)',
    title_joint: str = r'Sweeping $\alpha=\beta$',
    xlabel_alpha: str = r'$\alpha$',
    xlabel_beta: str = r'$\beta$',
    xlabel_joint: str = r'$\alpha=\beta$',
    ylabel: str = 'Value-Sum',
    figsize: Tuple[float, float] = (6, 3),
    title_fontsize: float = 13,
    axis_label_fontsize: float = 11,
    tick_label_fontsize: float = 11,
    legend_fontsize: float = 11,
    line_width: float = 2.2,
    marker_size: float = 4,
    line_alpha: float = 1.0,
    tick_length: float = 4.0,
    tick_width: float = 1.0,
    tick_direction: str = 'out',
    show_top_spine: bool = False,
    show_right_spine: bool = False,
    show_bottom_spine: bool = True,
    show_left_spine: bool = True,
    # ---- Legend placement ----
    # 'inside_first'  : draw legend inside axes[0] (first panel).
    # 'figure_bottom' : keep the original figure-level lower-center legend.
    legend_mode: str = 'inside_first',
    legend_panel_idx: int = 0,
    legend_loc: str = 'center right',
    legend_bbox: Tuple[float, float] = (0.98, 0.5),
    legend_borderaxespad: float = 0.0,
    legend_ncol: int = 1,
    legend_frameon: bool = True,
    legend_framealpha: float = 0.9,
    legend_edgecolor: str = '0.4',
    legend_labelspacing: float = 0.3,
    legend_handlelength: float = 1.6,
    legend_handletextpad: float = 0.5,
    # Used only when legend_mode == 'figure_bottom'.
    legend_figure_bbox: Tuple[float, float] = (0.5, 0.12),
    layout_rect: Tuple[float, float, float, float] = (0, 0.17, 1, 0.98),
    zero_hline: bool = True,
    group_colors: Optional[dict] = None,
    group_styles: Optional[dict] = None,
    save_csv: bool = True,
):
    """1x3 group-sum panel: alpha-only, beta-only, joint sweeps.

    X-axis is log_2 with sweep value 0 placed at 0.5 so the baseline shows
    as the leftmost point. See ``legend_mode`` for legend placement options.
    """
    if group_colors is None:
        group_colors = DEFAULT_GROUP_COLORS
    if group_styles is None:
        group_styles = DEFAULT_GROUP_STYLES
    if legend_mode not in ('inside_first', 'figure_bottom'):
        raise ValueError("legend_mode must be 'inside_first' or 'figure_bottom'.")

    base_players = list(mean_value_for_case(0.0, 0.0).index)
    paid_players = [m for m in base_players if m in PAID_SET]
    open_players = [m for m in base_players if m not in PAID_SET]
    if not paid_players or not open_players:
        raise ValueError('Need at least one paid and one open-source player.')

    panels = [
        ('alpha', title_alpha, xlabel_alpha, [(a, 0.0) for a in SWEEP_VALUES]),
        ('beta',  title_beta,  xlabel_beta,  [(0.0, b) for b in SWEEP_VALUES]),
        ('both',  title_joint, xlabel_joint, [(v, v) for v in SWEEP_VALUES]),
    ]
    records = []
    for sweep_name, panel_title, xlabel, points in panels:
        for alpha, beta in points:
            values = mean_value_for_case(alpha, beta)
            sweep_value = max(float(alpha), float(beta))
            for group, players in [('paid', paid_players), ('open-source', open_players)]:
                records.append({
                    'sweep_name': sweep_name, 'panel': panel_title,
                    'sweep_value': sweep_value, 'alpha': float(alpha), 'beta': float(beta),
                    'group': group, 'group_sum': float(values.loc[players].sum()),
                    'total_value': float(values.sum()),
                })
    table = pd.DataFrame(records)
    if save_csv:
        csv_path = PLOT_SUBDIR / (Path(output_name).stem + '.csv')
        table.to_csv(csv_path, index=False)
        print('CSV saved:', csv_path)

    plot_x = [0.5 if x == 0 else x for x in SWEEP_VALUES]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, (sweep_name, panel_title, xlabel, _pts) in zip(axes, panels):
        panel_df = table[table['sweep_name'] == sweep_name]
        for group in ['paid', 'open-source']:
            gdf = panel_df[panel_df['group'] == group].sort_values('sweep_value')
            x = [0.5 if v == 0 else v for v in gdf['sweep_value']]
            ax.plot(
                x, gdf['group_sum'],
                marker='o', markersize=marker_size, linewidth=line_width,
                linestyle=group_styles[group], color=group_colors[group],
                alpha=line_alpha, label=_capitalize_group(group),
            )
        ax.set_title(panel_title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(FixedLocator(plot_x))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p, vals=SWEEP_VALUES, xs=plot_x: (
            _format_pow2_tick(vals[xs.index(v)]) if v in xs else ''
        )))
        ax.tick_params(
            axis='x', which='both',
            length=tick_length, width=tick_width, direction=tick_direction,
            bottom=True, top=False, labelsize=tick_label_fontsize,
        )
        ax.tick_params(
            axis='y', which='both',
            length=tick_length, width=tick_width, direction=tick_direction,
            left=True, right=False, labelsize=tick_label_fontsize,
        )
        if zero_hline:
            ax.axhline(0, color='0.25', linewidth=0.9, linestyle='--')
        ax.spines['top'].set_visible(show_top_spine)
        ax.spines['right'].set_visible(show_right_spine)
        ax.spines['bottom'].set_visible(show_bottom_spine)
        ax.spines['left'].set_visible(show_left_spine)

    axes[0].set_ylabel(ylabel, fontsize=axis_label_fontsize)

    # Remove any per-axes legends matplotlib auto-draws from `label=` kwargs.
    for ax in axes:
        leg_ = ax.get_legend()
        if leg_ is not None:
            leg_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if legend_mode == 'inside_first':
        target_ax = axes[legend_panel_idx % len(axes)]
        leg = target_ax.legend(
            handles, labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            bbox_transform=target_ax.transAxes,
            ncol=legend_ncol,
            frameon=legend_frameon,
            framealpha=legend_framealpha,
            edgecolor=legend_edgecolor,
            borderaxespad=legend_borderaxespad,
            labelspacing=legend_labelspacing,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            fontsize=legend_fontsize,
        )
        target_ax.add_artist(leg)
    else:  # 'figure_bottom'
        leg = fig.legend(
            handles, labels,
            loc='lower center', bbox_to_anchor=legend_figure_bbox,
            ncol=2, frameon=legend_frameon, fontsize=legend_fontsize,
        )

    fig.tight_layout(rect=list(layout_rect))
    out_path = PLOT_SUBDIR / output_name
    fig.savefig(out_path, bbox_inches='tight', bbox_extra_artists=(leg,))
    plt.show()
    plt.close(fig)
    print('Saved', out_path)
    return out_path


# ========= appendix figure config =========
# Player line palette (for the per-player sweep figure).
PALETTE = [
    '#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b',
    '#d0bbff', '#debb9b', '#cfcfcf', '#fab0e4',
    '#c984c3', '#fde68a', '#99f6e4',
    '#ccebc5', '#b3d9ff', '#ffd9b3',
    '#e6ccff', '#fff2b3', '#f7d6e0',
    '#d9f0c7', '#cfe8d6', '#e6d8c3',
]


def paid_first(items):
    items = list(items)
    return [p for p in items if p in PAID_SET] + [p for p in items if p not in PAID_SET]


# Apply log1p to the per-player sweep curves? (rank heatmap is unaffected.)
APPENDIX_SWEEP_LOG1P = False


# Appendix Figure A: per-player value trajectories along the three sweeps.

def plot_per_player_sweep(
    output_name: str = 'llm_player_sweep.pdf',
    *,
    figsize: Tuple[float, float] = (18, 6),
    title_alpha: str = r'Sweeping $\alpha$ ($\beta=0$)',
    title_beta: str = r'Sweeping $\beta$ ($\alpha=0$)',
    title_joint: str = r'Sweeping $\alpha=\beta$',
    xlabel_alpha: str = r'$\alpha$',
    xlabel_beta: str = r'$\beta$',
    xlabel_joint: str = r'$\alpha=\beta$',
    title_fontsize: float = 16,
    axis_label_fontsize: float = 16,
    tick_label_fontsize: float = 13,
    line_width: float = 2.0,
    marker_size: float = 5,
    legend_player_fontsize: float = 11,
    legend_style_fontsize: float = 11,
    legend_player_title: str = 'player',
    palette: Optional[Sequence[str]] = None,
    use_log1p: Optional[bool] = None,
    save_csv: bool = True,
):
    """3-panel line plot: each panel shows one sweep, one line per model.

    Solid = open-source, dashed = paid."""
    if palette is None:
        palette = PALETTE
    if use_log1p is None:
        use_log1p = APPENDIX_SWEEP_LOG1P

    panels = [
        (title_alpha, xlabel_alpha, [(a, 0.0) for a in SWEEP_VALUES]),
        (title_beta,  xlabel_beta,  [(0.0, b) for b in SWEEP_VALUES]),
        (title_joint, xlabel_joint, [(v, v) for v in SWEEP_VALUES]),
    ]

    # Build per-panel value frames (rows = sweep value, cols = models).
    frames = []
    for title, xlabel, points in panels:
        rows = [mean_value_for_case(a, b) for (a, b) in points]
        df = pd.DataFrame(rows)
        df.index = [max(a, b) for (a, b) in points]
        df.index.name = 'sweep_value'
        frames.append((title, xlabel, df))

    if use_log1p:
        for k, (title, xlabel, df) in enumerate(frames):
            min_val = float(df.min().min())
            if min_val <= -1.0:
                raise ValueError(
                    f'use_log1p requires values > -1, but found min={min_val:.6f} in {title}.'
                )
            frames[k] = (title, xlabel, np.log1p(df))

    players = paid_first(list(frames[0][2].columns))
    if len(players) > len(palette):
        raise ValueError(f'Need {len(players)} colors but PALETTE has {len(palette)}.')
    palette = list(palette)[: len(players)]
    frames = [(title, xlabel, df.loc[:, players]) for title, xlabel, df in frames]

    # log_2 axis with 0 placed at 0.5
    plot_x = [0.5 if x == 0 else x for x in SWEEP_VALUES]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    for ax, (title, xlabel, df) in zip(axes, frames):
        for color, player in zip(palette, players):
            sns.lineplot(
                x=plot_x,
                y=df[player].values,
                ax=ax,
                color=color,
                linestyle='--' if player in PAID_SET else '-',
                marker='o',
                markersize=marker_size,
                linewidth=line_width,
            )
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_xscale('log', base=2)
        ax.set_xticks(plot_x)
        ax.set_xticklabels([str(x) for x in SWEEP_VALUES])
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)
        ax.grid(False)

    axes[0].set_ylabel('log(1 + GPASV)' if use_log1p else 'GPASV', fontsize=axis_label_fontsize)

    # Per-player legend (right of figure)
    handles = [
        plt.Line2D([0], [0], color=color, lw=line_width, marker='o',
                   linestyle='--' if player in PAID_SET else '-')
        for color, player in zip(palette, players)
    ]
    leg = fig.legend(
        handles, players,
        loc='center left', bbox_to_anchor=(1.01, 0.5),
        frameon=False, title=legend_player_title, fontsize=legend_player_fontsize,
    )

    # Solid/dashed style legend (bottom center)
    style_handles = [
        plt.Line2D([0], [0], color='0.25', lw=line_width, linestyle='--'),
        plt.Line2D([0], [0], color='0.25', lw=line_width, linestyle='-'),
    ]
    style_leg = fig.legend(
        style_handles, ['paid', 'open-source'],
        loc='lower center', bbox_to_anchor=(0.5, -0.04),
        ncol=2, frameon=False, fontsize=legend_style_fontsize,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = PLOT_SUBDIR / output_name
    plt.savefig(out_path, bbox_inches='tight',
                bbox_extra_artists=(leg, style_leg), pad_inches=0.2)
    plt.show()
    print(f'Saved: {out_path}')

    if save_csv:
        long_rows = []
        for title, _, df in frames:
            for sweep_val, row in df.iterrows():
                for player, val in row.items():
                    long_rows.append({'sweep': title, 'sweep_value': sweep_val,
                                      'player': player, 'value': float(val)})
        csv_path = out_path.with_suffix('.csv')
        pd.DataFrame(long_rows).to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}')


# Appendix Figure B: per-player rank heatmap on the (alpha, beta) 19-cell grid.

# Darker title colors (saturated) used to color the per-panel model name in the heatmap.
DEFAULT_HEATMAP_TITLE_COLORS = {
    'paid':        '#1f4e79',  # dark navy
    'open-source': '#bf5700',  # burnt orange
}


def plot_per_player_rank_heatmap(
    output_name: str = 'llm_rank_heatmap.pdf',
    *,
    # ---- Layout ----
    ncols: int = 5,
    panel_w: float = 3.6,
    panel_h: float = 3.3,
    figsize: Optional[Tuple[float, float]] = None,   # overrides (panel_w * ncols, panel_h * nrows)
    wspace: float = 0.25,
    hspace: float = 0.35,
    subplots_right: float = 0.9,
    subplots_top: float = 0.93,
    # ---- Colorbar position (figure coords: left, bottom, width, height) ----
    cbar_left: float = 0.92,
    cbar_bottom: float = 0.20,
    cbar_width: float = 0.015,
    cbar_height: float = 0.65,
    # ---- Fonts ----
    title_fontsize: float = 11,
    axis_label_fontsize: float = 9,
    tick_label_fontsize: float = 9,
    cbar_label_fontsize: float = 12,
    cbar_tick_fontsize: float = 11,
    # ---- Title styling by group ----
    title_colors: Optional[dict] = None,
    title_fontweight: str = 'bold',
    # ---- Heatmap appearance ----
    cmap: str = 'YlGnBu_r',
    cbar_label: str = 'Rank (1 = Best)',
    cbar_n_ticks: int = 5,
    cell_linewidth: float = 0.35,
    cell_linecolor: str = '0.88',
    # ---- Misc ----
    save_csv: bool = True,
):
    """For each model, draw a (beta, alpha) rank heatmap over the 19 supported (a, b) cells.
    Rank is computed across all models at each (alpha, beta) cell; off-support cells are masked.
    Per-panel title color is set by the paid/open-source group via `title_colors`."""
    if title_colors is None:
        title_colors = DEFAULT_HEATMAP_TITLE_COLORS

    GRID = list(SWEEP_VALUES)
    n_grid = len(GRID)

    vec_cache = {}
    def mean_vec_cached(alpha, beta):
        key = (float(alpha), float(beta))
        if key not in vec_cache:
            try:
                vec_cache[key] = mean_value_for_case(alpha, beta)
            except ValueError:
                vec_cache[key] = None
        return vec_cache[key]

    if 0 not in SWEEP_VALUES:
        raise ValueError('SWEEP_VALUES must include 0 for the (0,0) corner.')
    v00 = mean_vec_cached(0, 0)
    if v00 is None:
        raise RuntimeError('Could not load mean vector at (0, 0).')
    players = paid_first(list(v00.index))
    n_all = len(players)

    rank_mats = {p: np.full((n_grid, n_grid), np.nan, dtype=float) for p in players}
    for i, beta in enumerate(GRID):
        for j, alpha in enumerate(GRID):
            vec = mean_vec_cached(alpha, beta)
            if vec is None:
                continue
            r_all = vec.rank(ascending=False, method='min')
            for p in players:
                rank_mats[p][i, j] = r_all[p]

    nrows = int(np.ceil(len(players) / ncols))
    if figsize is None:
        fig_w, fig_h = panel_w * ncols, panel_h * nrows
    else:
        fig_w, fig_h = figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(players), len(axes)):
        axes[k].set_visible(False)

    for idx, (ax, player) in enumerate(zip(axes, players)):
        sns.heatmap(
            np.flipud(rank_mats[player]),
            ax=ax,
            xticklabels=GRID,
            yticklabels=list(reversed(GRID)),
            cmap=cmap,
            vmin=1, vmax=n_all,
            mask=np.flipud(np.isnan(rank_mats[player])),
            square=True, cbar=False,
            linewidths=cell_linewidth, linecolor=cell_linecolor,
        )
        title_color = title_colors['paid' if player in PAID_SET else 'open-source']
        ax.set_title(player, fontsize=title_fontsize, color=title_color,
                     fontweight=title_fontweight)
        row, col = idx // ncols, idx % ncols
        ax.set_xlabel(r'$\alpha$' if row == nrows - 1 else '', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\beta$' if col == 0 else '', fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize)

    fig.subplots_adjust(right=subplots_right, top=subplots_top,
                        hspace=hspace, wspace=wspace)
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_all))
    sm.set_array(np.linspace(1, n_all, 256))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(cbar_label, fontsize=cbar_label_fontsize)
    ticks = np.linspace(1, n_all, num=min(cbar_n_ticks, n_all), dtype=float)
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(int(round(t))) for t in ticks])
    cb.ax.tick_params(labelsize=cbar_tick_fontsize)

    out_path = PLOT_SUBDIR / output_name
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.15)
    plt.show()
    print(f'Saved: {out_path}')

    if save_csv:
        long_rows = []
        for i, beta in enumerate(GRID):
            for j, alpha in enumerate(GRID):
                vec = mean_vec_cached(alpha, beta)
                if vec is None:
                    continue
                r_all = vec.rank(ascending=False, method='min')
                for p in players:
                    long_rows.append({'alpha': alpha, 'beta': beta, 'player': p,
                                      'value': float(vec[p]), 'rank': int(r_all[p])})
        csv_path = out_path.with_suffix('.csv')
        pd.DataFrame(long_rows).to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}')


# Appendix Figure C: top-K bars across all 19 priority regimes.
# Layout is TRANSPOSED: 7 rows (baseline + 6 temperatures) x 3 cols (alpha, beta, joint sweeps).

def plot_topk_full_grid(
    output_name: str = 'llm_top8_all19.pdf',
    *,
    # ---- Layout ----
    top_k: int = 8,
    panel_w: float = 4.0,
    panel_h: float = 2.6,
    figsize: Optional[Tuple[float, float]] = None,    # overrides (panel_w * ncols, panel_h * nrows)
    wspace: float = 0.85,
    hspace: float = 0.55,
    subplots_left: float = 0.08,
    subplots_right: float = 0.99,
    subplots_top: float = 0.96,
    subplots_bottom: float = 0.06,
    # ---- Column / row strip headers ----
    col_titles: Sequence[str] = (
        r'$\alpha$-sweep ($\beta=0$)',
        r'$\beta$-sweep ($\alpha=0$)',
        r'joint $\alpha=\beta$',
    ),
    col_title_fontsize: float = 16,
    col_title_offset_pts: float = 36,    # vertical offset above the top row
    # ---- Per-panel title text (the (alpha,beta)=(...) tag) ----
    title_fontsize: float = 12,
    # ---- Bar / axis / value labels ----
    axis_label_fontsize: float = 10,
    tick_label_fontsize: float = 9,
    value_label_fontsize: float = 8,
    model_label_fontsize: float = 9,
    xlabel: str = 'GPASV',
    # ---- Legend (bottom-center, simple) ----
    legend_fontsize: float = 13,
    legend_loc: str = 'lower center',
    legend_bbox: Tuple[float, float] = (0.5, -0.005),
    legend_ncol: int = 2,
    legend_frameon: bool = True,
    legend_framealpha: float = 0.9,
    legend_edgecolor: str = '0.4',
    legend_handletextpad: float = 0.5,
    legend_columnspacing: float = 1.5,
    # ---- Misc ----
    group_colors: Optional[dict] = None,
    save_csv: bool = True,
):
    """7 rows (temperatures: baseline + 1, 2, 4, 8, 16, 32) x 3 cols (sweeps).

    Cell at (r, c) is the top-K bar chart for sweep c at temperature SWEEP_VALUES[r].
    Legend is drawn at the bottom-center of the figure."""
    if group_colors is None:
        group_colors = DEFAULT_GROUP_COLORS

    sweep_specs = [
        ('alpha', lambda s: (s, 0.0)),
        ('beta',  lambda s: (0.0, s)),
        ('both',  lambda s: (s, s)),
    ]
    nrows, ncols = len(SWEEP_VALUES), len(sweep_specs)
    if figsize is None:
        fig_w, fig_h = panel_w * ncols, panel_h * nrows
    else:
        fig_w, fig_h = figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    csv_rows = []
    for r, s in enumerate(SWEEP_VALUES):
        for c, (sweep_label, point_fn) in enumerate(sweep_specs):
            ax = axes[r, c]
            alpha, beta = point_fn(s)
            vals = mean_value_for_case(alpha, beta)
            title = rf'$(\alpha,\beta)=({alpha:g},{beta:g})$'
            _draw_topk_bar(
                ax, vals, title,
                top_k=top_k, group_colors=group_colors,
                title_fontsize=title_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                tick_label_fontsize=tick_label_fontsize,
                value_label_fontsize=value_label_fontsize,
                model_label_fontsize=model_label_fontsize,
                show_xlabel=(r == nrows - 1),
                xlabel=xlabel,
            )
            if r == 0:
                ax.annotate(col_titles[c], xy=(0.5, 1.0),
                            xytext=(0, col_title_offset_pts),
                            xycoords='axes fraction',
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=col_title_fontsize)
            if save_csv:
                top_vals = vals.sort_values(ascending=False).head(top_k)
                for rank, (p, v) in enumerate(top_vals.items(), start=1):
                    csv_rows.append({'sweep': sweep_label, 'alpha': alpha, 'beta': beta,
                                     'rank': rank, 'player': p, 'value': float(v),
                                     'group': model_group(p)})

    legend_handles = [
        Patch(facecolor=group_colors['paid'], edgecolor='white', label='paid'),
        Patch(facecolor=group_colors['open-source'], edgecolor='white', label='open-source'),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc=legend_loc, bbox_to_anchor=legend_bbox,
        ncol=legend_ncol, frameon=legend_frameon,
        framealpha=legend_framealpha, edgecolor=legend_edgecolor,
        fontsize=legend_fontsize,
        handletextpad=legend_handletextpad,
        columnspacing=legend_columnspacing,
    )

    fig.subplots_adjust(wspace=wspace, hspace=hspace,
                        left=subplots_left, right=subplots_right,
                        top=subplots_top, bottom=subplots_bottom)

    out_path = PLOT_SUBDIR / output_name
    plt.savefig(out_path, bbox_inches='tight',
                bbox_extra_artists=(leg,), pad_inches=0.15)
    plt.show()
    print(f'Saved: {out_path}')

    if save_csv:
        csv_path = out_path.with_suffix('.csv')
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}')



def _run_top8():
    # Top-K bar grid (Figure 8).
    plot_topk_grid(
        output_name='llm_top8.pdf',
        weak_strength=1,
        strong_strength=32,
        top_k=8,
        figsize=(16, 7),
        # Title strings: pick from the suggested sets below, or write your own.
        title_alpha='Favoring Open-Source Models',     # alt: 'Open-Source Boost', r'$\alpha$-Sweep (Open-Source Pref.)'
        title_beta='Favoring Human Preferences',  # alt: 'Edge-Density Boost', r'$\beta$-Sweep (Graph Pref.)'
        title_joint='Favoring Both',           # alt: 'Combined Boost', r'$\alpha=\beta$ Joint Sweep'
        title_baseline='Equal Treatment (SV)',
        show_param_in_title=True,            # adds '(\alpha,\beta)=(a,b)' under each base title
        # Second-row (strong_*) title:
        #   'full'       -> base name + (alpha,beta) tag (same as first row)
        #   'param_only' -> drop base name, keep only (alpha,beta) tag (default)
        #   'none'       -> no title at all
        strong_row_title_mode='param_only',
        title_fontsize=19,
        axis_label_fontsize=16,
        tick_label_fontsize=14,
        value_label_fontsize=14,
        model_label_fontsize=14,
        legend_fontsize=16,
        xlabel='GPASV',
        width_ratios=(1.15, 1.0, 1.0, 1.0),
        wspace=1.18,
        hspace=0.34,
        subplots_adjust=dict(left=0.06, right=0.99, top=0.92, bottom=0.16, wspace=0.82, hspace=0.38),
        # Legend: figure-level, top-left of the whole figure, single horizontal row.
        legend_mode='figure',
        legend_loc='upper left',
        legend_bbox=(-0.05, 0.995),               # figure coords: a touch in from the corner
        legend_borderaxespad=0.0,
        legend_ncol=2,                          # one row (paid + open-source side by side)
        legend_frameon=True,
        legend_framealpha=0.9,
        legend_edgecolor='0.4',
        legend_labelspacing=0.3,
        legend_handlelength=1.4,
        legend_handletextpad=0.5,
        legend_columnspacing=1.5,
        save_csv=True,
    )


def _run_group():
    # Group-sum sweep (Figure 9).
    plot_group_sum_sweep(
        output_name='llm_group.pdf',
        title_alpha=r'Sweep $\alpha$ ($\beta=0$)',
        title_beta=r'Sweep $\beta$ ($\alpha=0$)',
        title_joint=r'$\alpha=\beta$ Sweep',
        xlabel_alpha=r'$\alpha$',
        xlabel_beta=r'$\beta$',
        xlabel_joint=r'$\alpha=\beta$',
        ylabel='Value-Sum',
        figsize=(7.6, 3.5),
        title_fontsize=16,
        axis_label_fontsize=16,
        tick_label_fontsize=16,
        legend_fontsize=13,
        line_width=2.2,
        marker_size=4,
        line_alpha=1.0,
        # Tick marks (length>0 = visible ticks on the axis edge).
        tick_length=4.0,
        tick_width=1.0,
        tick_direction='out',
        # Spine visibility.
        show_top_spine=False,
        show_right_spine=False,
        show_bottom_spine=True,
        show_left_spine=True,
        # Legend: inside the first panel, center-right (axes coords), with frame.
        legend_mode='inside_first',          # or 'figure_bottom'
        legend_panel_idx=0,                  # 0 = first panel
        legend_loc='center right',
        legend_bbox=(0.98, 0.5),             # axes coords of the chosen panel
        legend_borderaxespad=0.0,
        legend_ncol=1,
        legend_frameon=True,
        legend_framealpha=0.9,
        legend_edgecolor='0.4',
        legend_labelspacing=0.3,
        legend_handlelength=1.6,
        legend_handletextpad=0.5,
        layout_rect=(0, 0.04, 1, 0.98),      # no bottom strip needed when legend is inside
        zero_hline=True,
        save_csv=True,
    )


def _run_player_sweep():
    # Appendix Figure A: per-player sweep trajectories.
    plot_per_player_sweep(
        output_name='llm_player_sweep.pdf',
        figsize=(13, 5),
        title_alpha=r'Sweep $\alpha$ ($\beta=0$)',
        title_beta=r'Sweep $\beta$ ($\alpha=0$)',
        title_joint=r'Sweep $\alpha=\beta$',
        title_fontsize=20,
        axis_label_fontsize=20,
        tick_label_fontsize=20,
        line_width=2.0,
        marker_size=5,
        legend_player_fontsize=13,
        legend_style_fontsize=18,
        legend_player_title='Model',
        use_log1p=False,
        save_csv=True,
    )


def _run_rank_heatmap():
    # Appendix Figure B: per-player rank heatmap.
    plot_per_player_rank_heatmap(
        output_name='llm_rank_heatmap.pdf',
        # ---- Layout ----
        ncols=5,
        panel_w=3.6,
        panel_h=3.3,
        figsize=(20, 14),                  # set e.g. (20, 14) to override panel_w * ncols
        wspace=0.25,
        hspace=0.35,
        subplots_right=0.9,
        subplots_top=0.93,
        # ---- Colorbar position (figure coords) ----
        cbar_left=0.92,
        cbar_bottom=0.20,
        cbar_width=0.015,
        cbar_height=0.65,
        # ---- Fonts ----
        title_fontsize=20,
        axis_label_fontsize=20,
        tick_label_fontsize=18,
        cbar_label_fontsize=20,
        cbar_tick_fontsize=20,
        # ---- Title styling (per-panel model name) ----
        title_colors={'paid': '#1f4e79', 'open-source': '#bf5700'},
        title_fontweight='bold',
        # ---- Heatmap appearance ----
        cmap='YlGnBu_r',
        cbar_label='Rank (1 = Best)',
        cbar_n_ticks=5,
        cell_linewidth=0.35,
        cell_linecolor='0.88',
        save_csv=True,
    )


def _run_top8_all19():
    # Appendix Figure C: top-8 across all 19 regimes (7 rows x 3 cols, transposed).
    plot_topk_full_grid(
        output_name='llm_top8_all19.pdf',
        # ---- Layout ----
        top_k=8,
        panel_w=3.0,
        panel_h=2.0,
        figsize=None,                       # override (panel_w * 3, panel_h * 7) if needed
        wspace=1.5,
        hspace=0.55,
        subplots_left=0.08,
        subplots_right=0.99,
        subplots_top=0.96,
        subplots_bottom=0.06,
        # ---- Column / row strip headers ----
        col_titles=(
            r'Sweep $\alpha$ ($\beta=0$)',
            r'Sweep $\beta$ ($\alpha=0$)',
            r'Sweep $\alpha=\beta$',
        ),
        col_title_fontsize=18,
        col_title_offset_pts=36,
        # ---- Per-panel title (the (alpha,beta)=(...) tag) ----
        title_fontsize=13,
        # ---- Bar / axis / value labels ----
        axis_label_fontsize=12,
        tick_label_fontsize=11,
        value_label_fontsize=10,
        model_label_fontsize=11,
        xlabel='GPASV',
        # ---- Legend (bottom-center) ----
        legend_fontsize=14,
        legend_loc='lower center',
        legend_bbox=(0.5, -0.01),
        legend_ncol=2,
        legend_frameon=False,
        legend_framealpha=0.9,
        legend_edgecolor='0.4',
        legend_handletextpad=0.5,
        legend_columnspacing=1.5,
        save_csv=True,
    )


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--which",
        default="all",
        choices=["all", "top8", "group", "player_sweep", "rank_heatmap", "top8_all19"],
        help=("Which figure(s) to render. 'all' renders the two main-text figures "
              "(top8, group) and the three appendix figures (player_sweep, "
              "rank_heatmap, top8_all19)."),
    )
    return p.parse_args()


_RUNNERS = {
    "top8": _run_top8,
    "group": _run_group,
    "player_sweep": _run_player_sweep,
    "rank_heatmap": _run_rank_heatmap,
    "top8_all19": _run_top8_all19,
}


if __name__ == "__main__":
    args = _parse_args()
    if args.which == "all":
        for name, fn in _RUNNERS.items():
            print(f"--- rendering {name} ---")
            fn()
    else:
        _RUNNERS[args.which]()
