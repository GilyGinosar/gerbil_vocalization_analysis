"""Library helpers for call-transition analysis.

Public entry points (used by scripts/run_transitions.py):
    - compute_and_save_arena_transitions
    - plot_transition_matrices
    - collect_inter_call_gaps
    - collect_self_inter_call_gaps
    - compute_shared_log_count_max

Inputs are CSVs with the columns produced by scripts/run_transitions.add_exp_times:
file_num, channel, event_type, start_time_experiment_sec, stop_time_experiment_sec,
start_time_real, stop_time_real.

This module no longer carries a top-level runner block; importing it has no
side effects. To run an analysis, use scripts/run_transitions.py.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CALL_TYPE_ORDER = [
    'pup', 'low','half-mnt', 'mnt', 'hat', 'ufm', 'tilda', 'warble',
    'alarm','dense-stack', 'dfm-stack'
]

CALL_TYPE_CANONICAL_MAP = {
    'tild': 'tilda',
    'tilda': 'tilda',
    'dfm-stack-dense-stack': 'dfm-stack',
    'dfm_stack_dense_stack': 'dfm-stack',
    'dfm-stack': 'dfm-stack',
    'dense-stack': 'dense-stack',
    'dense_stack': 'dense-stack',
    'half_mnt': 'half-mnt',
    'halfmnt': 'half-mnt',
}


def _call_color(call_type):
    call = str(call_type).lower()
    if call in {'high-freq'}:
        return '#2A9D8F'
    if call in {'tilda', 'ufm', 'warble'}:
        return '#2A9D8F'
    if call in {'half-mnt', 'mnt', 'half_mnt', 'halfmnt'}:
        return '#E9C46A'
    if call in {'low', 'dfm', 'stack', 'dense-stack', 'dfm-stack-dense-stack'}:
        return '#457B9D'
    if call == 'alarm':
        return '#E63946'
    return '#8D99AE'


def _canonicalize_call_type(value):
    key = str(value).strip().lower()
    return CALL_TYPE_CANONICAL_MAP.get(key, key)


def _effective_call_type_order(base_order, call_group_map):
    if not call_group_map:
        return list(base_order)
    order = []
    for call in base_order:
        grouped = call_group_map.get(call, call)
        if grouped not in order:
            order.append(grouped)
    return order


def collect_inter_call_gaps(csv_files):
    if isinstance(csv_files, str):
        csv_paths = [csv_files]
    else:
        csv_paths = list(csv_files)

    gaps = []
    for csv_path in csv_paths:
        source_name = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        df['_source_exp'] = source_name
        df = df.dropna(subset=['event_type', 'start_time_experiment_sec', 'stop_time_experiment_sec']).copy()
        df = df[df['event_type'] != 'noise']
        if df.empty:
            continue

        for (_, channel_df) in df.groupby(['_source_exp', 'channel']):
            rows = channel_df.sort_values('start_time_experiment_sec').to_dict('records')
            for i in range(len(rows) - 1):
                gap = rows[i + 1]['start_time_experiment_sec'] - rows[i]['stop_time_experiment_sec']
                if pd.notna(gap) and gap >= 0:
                    gaps.append(float(gap))
    return np.array(gaps, dtype=float)


def collect_self_inter_call_gaps(csv_files, target_call_type, call_group_map=None):
    if isinstance(csv_files, str):
        csv_paths = [csv_files]
    else:
        csv_paths = list(csv_files)

    target = str(target_call_type).strip().lower()
    gaps = []
    for csv_path in csv_paths:
        source_name = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        df['_source_exp'] = source_name
        df = df.dropna(subset=['event_type', 'start_time_experiment_sec', 'stop_time_experiment_sec']).copy()
        df['event_type'] = df['event_type'].map(_canonicalize_call_type)
        if call_group_map:
            df['event_type'] = df['event_type'].map(lambda x: call_group_map.get(x, x))
        df = df[df['event_type'] == target]
        if df.empty:
            continue

        for (_, channel_df) in df.groupby(['_source_exp', 'channel']):
            rows = channel_df.sort_values('start_time_experiment_sec').to_dict('records')
            for i in range(len(rows) - 1):
                gap = rows[i + 1]['start_time_experiment_sec'] - rows[i]['stop_time_experiment_sec']
                if pd.notna(gap) and gap >= 0:
                    gaps.append(float(gap))
    return np.array(gaps, dtype=float)


def plot_inter_call_gap_distribution(
    gaps,
    thresholds,
    output_path,
    title='',
    n_bins=100,
    xmax_sec=1000.0,
):
    """Wide single-axis histogram of inter-call gaps for threshold inspection.

    Plots all positive gaps on a log-x scale with vertical dashed lines at the
    given thresholds, plus a cumulative-fraction overlay so it's easy to read
    off what fraction of gaps fall below each threshold.

    Args:
        gaps: 1-D array of gap durations in seconds.
        thresholds: iterable of threshold values (sec) to mark with vlines.
        output_path: where to save the PNG.
        title: optional figure title.
        n_bins: number of geometric (log-spaced) bins for the histogram.
        xmax_sec: x-axis upper bound. Bins and view are capped here; gaps
            beyond this are excluded from the histogram (but remain in the
            CDF), and the count is reported in the subtitle.
    """
    gaps = np.asarray(gaps, dtype=float)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        print("plot_inter_call_gap_distribution: no positive gaps to plot.")
        return None

    xmin = max(1e-3, float(positive.min()))
    xmax = min(float(positive.max()), float(xmax_sec))
    n_beyond = int((positive > xmax_sec).sum())
    bins = np.geomspace(xmin, xmax, n_bins)

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.hist(positive, bins=bins, color='#6C757D', edgecolor='white', linewidth=0.4)
    ax.set_xscale('log')
    ax.set_xlabel('Inter-call interval (sec)')
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.set_xlim(xmin, xmax)

    # CDF overlay on a secondary axis.
    sorted_gaps = np.sort(positive)
    cdf = np.arange(1, sorted_gaps.size + 1) / sorted_gaps.size
    ax_cdf = ax.twinx()
    ax_cdf.plot(sorted_gaps, cdf, color='#457B9D', linewidth=1.4, alpha=0.85)
    ax_cdf.set_ylabel('Cumulative fraction', color='#457B9D')
    ax_cdf.tick_params(axis='y', labelcolor='#457B9D')
    ax_cdf.set_ylim(0, 1.0)
    ax_cdf.spines['top'].set_visible(False)

    # Threshold markers + annotations with fraction-below.
    y_top = ax.get_ylim()[1]
    for thresh in thresholds:
        ax.axvline(thresh, color='#E63946', linestyle='--', linewidth=1.2, alpha=0.85)
        frac_below = float((positive <= thresh).mean())
        ax.text(
            thresh, y_top * 0.97,
            f' {thresh}s ({frac_below*100:.1f}% ≤)',
            rotation=90, va='top', ha='left', fontsize=9, color='#E63946',
        )

    base_title = title or f'Inter-call interval distribution (n={positive.size:,})'
    if n_beyond:
        base_title = f"{base_title}  —  {n_beyond:,} gaps > {xmax_sec:g}s not shown"
    ax.set_title(base_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved gap-distribution plot: {output_path}")
    return output_path


def compute_shared_log_count_max(output_dirs, arena_names=('arena', 'underground')):
    shared_max = 0.0
    for out_item in output_dirs:
        if isinstance(out_item, (list, tuple)) and len(out_item) == 2:
            out_dir, file_tag = out_item
        else:
            out_dir, file_tag = out_item, ''
        tag_part = f"_{file_tag}" if file_tag else ""
        for arena in arena_names:
            count_path = os.path.join(out_dir, f'counts_{arena}{tag_part}.csv')
            if os.path.exists(count_path):
                cdf = pd.read_csv(count_path, index_col=0)
                shared_max = max(shared_max, float(np.log1p(cdf.values).max()))
    return shared_max


def compute_and_save_arena_transitions(
    csv_files,
    inter_call_interval_sec,
    output_dir='transition_results',
    min_inter_call_interval_sec=None,
    call_group_map=None,
    call_type_order=None,
    file_tag=''
):
    """
    Parses one or more consolidated CSVs by channel, computes transitions,
    and saves results to a specified directory.

    Notes:
    - `csv_files` can be a single path or a list of paths.
    - Transitions are computed within each source CSV independently
      (no transitions are counted across experiments).
    - Day/night filtering must be applied upstream by the caller; this function
      does not look at clock time.
    """
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Channel mapping and output grouping
    channel_map = {10: 'arena_1', 20: 'arena_2', 30: 'underground'}
    output_groups = {'arena': [10, 20], 'underground': [30]}

    try:
        if isinstance(csv_files, str):
            csv_paths = [csv_files]
        else:
            csv_paths = list(csv_files)

        if not csv_paths:
            raise ValueError("No CSV files provided.")

        all_dfs = []
        for csv_path in csv_paths:
            source_name = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
            src_df = pd.read_csv(csv_path)
            src_df['_source_exp'] = source_name
            all_dfs.append(src_df)

        df = pd.concat(all_dfs, ignore_index=True)

        # 2. Cleaning: drop missing events, canonicalize labels, drop 'noise'.
        df = df.dropna(subset=['event_type', 'start_time_experiment_sec']).copy()
        df['event_type'] = df['event_type'].map(_canonicalize_call_type)
        if call_group_map:
            df['event_type'] = df['event_type'].map(lambda x: call_group_map.get(x, x))
        df = df[df['event_type'] != 'noise']
        if call_type_order is None:
            call_type_order = _effective_call_type_order(CALL_TYPE_ORDER, call_group_map or {})
        df = df[df['event_type'].isin(call_type_order)]
        if df.empty:
            raise ValueError("No rows remain after cleaning input.")

        # Use fixed call-type order for all outputs, even if some are absent.
        all_call_types = list(call_type_order)

        # 4. Compute per-channel transition matrices first
        channel_matrices = {}
        channel_call_counts = {}
        for channel_id, arena_name in channel_map.items():
            # Filter for this channel across all source CSVs
            arena_data = df[df['channel'] == channel_id].copy()

            # Initialize empty count matrix
            matrix = pd.DataFrame(0, index=all_call_types, columns=all_call_types)

            # 5. Compute transitions within each source experiment only
            for _, exp_data in arena_data.groupby('_source_exp'):
                rows = exp_data.sort_values('start_time_experiment_sec').to_dict('records')
                for i in range(len(rows) - 1):
                    curr, nxt = rows[i], rows[i + 1]

                    # Gap between end of current and start of next
                    gap = nxt['start_time_experiment_sec'] - curr['stop_time_experiment_sec']

                    if min_inter_call_interval_sec is not None:
                        lower_ok = gap > min_inter_call_interval_sec
                    else:
                        lower_ok = gap >= 0   # reject overlapping calls (negative gap)
                    if lower_ok and gap <= inter_call_interval_sec:
                        matrix.loc[curr['event_type'], nxt['event_type']] += 1

            channel_matrices[channel_id] = matrix
            channel_call_counts[channel_id] = (
                arena_data['event_type']
                .value_counts()
                .reindex(all_call_types, fill_value=0)
                .astype(int)
            )

        # 6. Combine and save requested outputs
        tag_part = f"_{file_tag}" if file_tag else ""
        for output_name, channel_ids in output_groups.items():
            combined = pd.DataFrame(0, index=all_call_types, columns=all_call_types)
            combined_call_counts = pd.Series(0, index=all_call_types, dtype=int)
            for channel_id in channel_ids:
                combined = combined.add(channel_matrices[channel_id], fill_value=0)
                combined_call_counts = combined_call_counts.add(channel_call_counts[channel_id], fill_value=0)
            combined = combined.astype(int)
            combined_call_counts = combined_call_counts.astype(int)

            # Save raw counts
            count_path = os.path.join(output_dir, f'counts_{output_name}{tag_part}.csv')
            combined.to_csv(count_path)

            # Save per-call-type totals
            calls_path = os.path.join(output_dir, f'call_counts_{output_name}{tag_part}.csv')
            combined_call_counts.rename('n_calls').to_csv(calls_path, header=True)

            # Save probabilities (row-normalized)
            prob_matrix = combined.div(combined.sum(axis=1), axis=0).fillna(0)
            prob_path = os.path.join(output_dir, f'probabilities_{output_name}{tag_part}.csv')
            prob_matrix.to_csv(prob_path)

        print(f"Success! Transition matrices saved in: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_transition_matrices(
    output_dir_left,
    output_dir_mid,
    output_dir_right,
    arena_names=('arena', 'underground'),
    save_name='transition_matrices_overview.png',
    plot_note=None,
    row_y_shift=None,
    interval_left=None,
    interval_mid=None,
    interval_right=None,
    interval_left_label=None,
    interval_mid_label=None,
    interval_right_label=None,
    inter_call_gaps=None,
    self_ici_gaps_by_type=None,
    self_ici_call_types=None,
    shared_log_count_max=None,
    hist_full_bins=None,
    hist_full_ymax=None,
    hist_zoom_ymax=None,
    self_ici_ymax_by_type=None,
    thresholds=None,
    hist_full_xmax_sec=1000.0,
    call_type_order=None,
    file_tag_left='',
    file_tag_mid='',
    file_tag_right='',
    figure_title=None,
):
    """
    Row 1: call proportions (2 panels, unchanged style).
    Row 2: transition counts (6 panels: 3 interval groups x arena/underground).
    Row 3: transition probabilities (same 6-panel layout).
    """
    if len(arena_names) != 2:
        raise ValueError("Compare layout expects exactly two arena names.")
    if call_type_order is None:
        call_type_order = CALL_TYPE_ORDER

    # Layout (11 rows):
    #   0: within-location proportions (cols 0-1)
    #   1: across-locations proportions (cols 0-1) — pushed slightly down
    #   2: ICI full (wide, log-x, threshold lines + CDF)
    #   3: ICI zoom (wide, 0-3s linear)
    #   4..7: per-call-type self-ICI (one wide row each)
    #   8: count matrices (9 cols)
    #   9: probability matrices (row-normalized) (9 cols)
    #   10: joint matrices (9 cols)
    # 18-column gridspec so we can size things in finer-than-1/9 increments:
    # matrix panels span 2 cols each (× 9 panels = 18 cols, same widths as before),
    # histograms span 3 cols on the right (1/6 of figure width — about half the
    # width they had at 3/9 = 1/3), and proportion panels span 3 cols each (a
    # bit wider than before).
    fig = plt.figure(figsize=(52, 40), constrained_layout=False)
    gs = fig.add_gridspec(
        11, 18,
        height_ratios=[0.65, 0.65, 0.45, 0.45, 0.30, 0.30, 0.30, 0.30, 1.0, 1.0, 1.0],
        hspace=0.85, wspace=0.24,
    )
    fig.subplots_adjust(top=0.95)

    top_axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
    top_axes_global = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    # Histograms: stacked vertically on the right side of the figure.
    hist_axes = [fig.add_subplot(gs[2, 15:18]), fig.add_subplot(gs[3, 15:18])]   # [full, zoom]
    extra_hist_axes = [fig.add_subplot(gs[4 + i, 15:18]) for i in range(4)]
    count_axes = [fig.add_subplot(gs[8, 2*c:2*c+2]) for c in range(9)]
    prob_axes = [fig.add_subplot(gs[9, 2*c:2*c+2]) for c in range(9)]
    prob_global_axes = [fig.add_subplot(gs[10, 2*c:2*c+2]) for c in range(9)]

    if row_y_shift is None:
        row_y_shift = {
            0: 0.0,
            1: -0.07,    # push global proportions further down — row 0 is now taller

            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            # Differential shifts on matrix rows: each successive row pushed
            # further down so the inter-row gap (where row super-titles live)
            # is wider and tick labels stop occluding the row above.
            8: -0.02,
            9: -0.07,
            10: -0.12,
        }
    for r, row_axes in {
        0: top_axes,
        1: top_axes_global,
        2: [hist_axes[0]],
        3: [hist_axes[1]],
        4: [extra_hist_axes[0]],
        5: [extra_hist_axes[1]],
        6: [extra_hist_axes[2]],
        7: [extra_hist_axes[3]],
        8: count_axes,
        9: prob_axes,
        10: prob_global_axes,
    }.items():
        dy = float(row_y_shift.get(r, 0.0))
        if dy != 0.0:
            for ax in row_axes:
                p = ax.get_position()
                ax.set_position([p.x0, p.y0 + dy, p.width, p.height])

    # Decrease horizontal gaps within each triplet (Arena, Underground, Diff).
    triplet_dx_2nd = -0.01
    triplet_dx_3rd = -0.02
    for row_axes in (count_axes, prob_axes, prob_global_axes):
        for base in (0, 3, 6):
            p2 = row_axes[base + 1].get_position()
            row_axes[base + 1].set_position([p2.x0 + triplet_dx_2nd, p2.y0, p2.width, p2.height])
            p3 = row_axes[base + 2].get_position()
            row_axes[base + 2].set_position([p3.x0 + triplet_dx_3rd, p3.y0, p3.width, p3.height])

    call_count_series = {}
    count_left = {}
    count_mid = {}
    prob_left = {}
    prob_mid = {}
    count_right = {}
    prob_right = {}
    global_log_count_max = 0.0

    for arena in arena_names:
        tag_left = f"_{file_tag_left}" if file_tag_left else ""
        tag_mid = f"_{file_tag_mid}" if file_tag_mid else ""
        tag_right = f"_{file_tag_right}" if file_tag_right else ""
        call_count_series[arena] = pd.read_csv(
            os.path.join(output_dir_left, f'call_counts_{arena}{tag_left}.csv'), index_col=0
        )['n_calls']
        count_left[arena] = pd.read_csv(os.path.join(output_dir_left, f'counts_{arena}{tag_left}.csv'), index_col=0)
        prob_left[arena] = pd.read_csv(os.path.join(output_dir_left, f'probabilities_{arena}{tag_left}.csv'), index_col=0)
        count_mid[arena] = pd.read_csv(os.path.join(output_dir_mid, f'counts_{arena}{tag_mid}.csv'), index_col=0)
        prob_mid[arena] = pd.read_csv(os.path.join(output_dir_mid, f'probabilities_{arena}{tag_mid}.csv'), index_col=0)
        count_right[arena] = pd.read_csv(os.path.join(output_dir_right, f'counts_{arena}{tag_right}.csv'), index_col=0)
        prob_right[arena] = pd.read_csv(os.path.join(output_dir_right, f'probabilities_{arena}{tag_right}.csv'), index_col=0)
        global_log_count_max = max(global_log_count_max, np.log1p(count_left[arena].values).max())
        global_log_count_max = max(global_log_count_max, np.log1p(count_mid[arena].values).max())
        global_log_count_max = max(global_log_count_max, np.log1p(count_right[arena].values).max())
    if shared_log_count_max is not None:
        global_log_count_max = float(shared_log_count_max)

    # Total calls across all locations — denominator for the global-norm row.
    grand_total_calls = sum(
        int(call_count_series[a].reindex(call_type_order, fill_value=0).sum())
        for a in arena_names
    )

    for i, arena in enumerate(arena_names):
        area_title = 'Arena' if arena == 'arena' else 'Underground'
        calls_s = call_count_series[arena].reindex(call_type_order, fill_value=0)
        total_calls = int(calls_s.sum())
        call_props = calls_s / total_calls if total_calls > 0 else calls_s.astype(float)

        x = np.arange(len(calls_s.index))
        bars = top_axes[i].bar(x, call_props.values, color='#6C757D')
        top_axes[i].set_title(f'{area_title} (N_calls = {total_calls})')
        top_axes[i].set_ylim(0, 1)
        top_axes[i].set_xticks(x)
        top_axes[i].set_xticklabels(calls_s.index, rotation=90)
        top_axes[i].set_ylabel('Proportion')
        top_axes[i].spines['top'].set_visible(False)
        top_axes[i].spines['right'].set_visible(False)
        for b, n_calls in zip(bars, calls_s.values):
            top_axes[i].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'n={int(n_calls)}',
                ha='center', va='bottom', fontsize=8, rotation=90, clip_on=False
            )

        # Same bars but normalized by total calls across ALL locations.
        if grand_total_calls > 0:
            call_props_global = calls_s / grand_total_calls
        else:
            call_props_global = calls_s.astype(float) * 0.0

        bars_g = top_axes_global[i].bar(x, call_props_global.values, color='#6C757D')
        loc_share_pct = (total_calls / grand_total_calls * 100) if grand_total_calls > 0 else 0.0
        top_axes_global[i].set_title(
            f'{area_title} (relative to all {grand_total_calls:,} calls — '
            f'{area_title.lower()} = {loc_share_pct:.1f}%)'
        )
        top_axes_global[i].set_ylim(0, 1)
        top_axes_global[i].set_xticks(x)
        top_axes_global[i].set_xticklabels(calls_s.index, rotation=90)
        top_axes_global[i].set_ylabel('Proportion of total')
        top_axes_global[i].spines['top'].set_visible(False)
        top_axes_global[i].spines['right'].set_visible(False)
        for b, n_calls in zip(bars_g, calls_s.values):
            top_axes_global[i].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'n={int(n_calls)}',
                ha='center', va='bottom', fontsize=8, rotation=90, clip_on=False
            )

    # Wide ICI histograms: full (log-x, with threshold markers + CDF overlay)
    # and zoom (0-3 s linear). Each spans the full figure width.
    if inter_call_gaps is not None and len(inter_call_gaps) > 0:
        positive_gaps = inter_call_gaps[inter_call_gaps > 0]
        hist_ax = hist_axes[0]
        if len(positive_gaps) > 0:
            # Shared bins across variants when provided.
            if hist_full_bins is not None:
                bins = hist_full_bins
            else:
                xmax = min(float(positive_gaps.max()), float(hist_full_xmax_sec))
                bins = np.geomspace(max(1e-3, float(positive_gaps.min())), xmax, 100)
            hist_ax.hist(positive_gaps, bins=bins, color='#6C757D', edgecolor='white', linewidth=0.4)
            hist_ax.set_xscale('log')
            xmin = float(bins[0])
            xmax = float(bins[-1])
            hist_ax.set_xlim(xmin, xmax)
        else:
            hist_ax.hist(inter_call_gaps, bins=30, color='#6C757D', edgecolor='white', linewidth=0.4)

        n_beyond = int((positive_gaps > hist_full_xmax_sec).sum()) if len(positive_gaps) else 0
        title = f'Inter-call interval (full, n={len(positive_gaps):,})'
        if n_beyond:
            title += f'  —  {n_beyond:,} > {hist_full_xmax_sec:g}s not shown'
        hist_ax.set_title(title, fontsize=12)
        hist_ax.set_xlabel('Interval (sec)', fontsize=11)
        hist_ax.set_ylabel('Count', fontsize=11)
        hist_ax.spines['top'].set_visible(False)
        if hist_full_ymax is not None and hist_full_ymax > 0:
            hist_ax.set_ylim(0, hist_full_ymax * 1.05)

        # Cumulative-fraction overlay on a secondary axis.
        if len(positive_gaps) > 0:
            sorted_gaps = np.sort(positive_gaps)
            cdf = np.arange(1, sorted_gaps.size + 1) / sorted_gaps.size
            ax_cdf = hist_ax.twinx()
            ax_cdf.plot(sorted_gaps, cdf, color='#457B9D', linewidth=1.4, alpha=0.85)
            ax_cdf.set_ylabel('Cumulative fraction', color='#457B9D', fontsize=11)
            ax_cdf.tick_params(axis='y', labelcolor='#457B9D')
            ax_cdf.set_ylim(0, 1.0)
            ax_cdf.spines['top'].set_visible(False)

        # Threshold markers (gap-band boundaries) with annotation.
        if thresholds and len(positive_gaps) > 0:
            y_top = hist_ax.get_ylim()[1]
            for thresh in thresholds:
                hist_ax.axvline(thresh, color='#E63946', linestyle='--', linewidth=1.4, alpha=0.85)
                frac_below = float((positive_gaps <= thresh).mean())
                hist_ax.text(
                    thresh, y_top * 0.97,
                    f' {thresh}s ({frac_below*100:.1f}% ≤)',
                    rotation=90, va='top', ha='left', fontsize=10, color='#E63946',
                )

        zoom_ax = hist_axes[1]
        zoom_min = 0.05
        zoom_max = 3.0
        zoom_bins = np.geomspace(zoom_min, zoom_max, 30)
        zoom_data = inter_call_gaps[(inter_call_gaps >= zoom_min) & (inter_call_gaps <= zoom_max)]
        if len(zoom_data) > 0:
            zoom_ax.hist(zoom_data, bins=zoom_bins, color='#ADB5BD', edgecolor='white', linewidth=0.4)
        zoom_ax.set_xscale('log')
        zoom_ax.set_xlim(zoom_min, zoom_max)
        if hist_zoom_ymax is not None and hist_zoom_ymax > 0:
            zoom_ax.set_ylim(0, hist_zoom_ymax * 1.05)
        zoom_ticks = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
        zoom_ax.set_xticks(zoom_ticks)
        zoom_ax.set_xticklabels([f'{t:g}' for t in zoom_ticks], fontsize=10)
        zoom_ax.set_title('Inter-call interval (0.05-3 s)', fontsize=12)
        zoom_ax.set_xlabel('Interval (sec)', fontsize=11)
        zoom_ax.set_ylabel('Count', fontsize=11)
        zoom_ax.spines['top'].set_visible(False)
        zoom_ax.spines['right'].set_visible(False)
        # Mark thresholds that fall in the zoom range.
        if thresholds:
            for thresh in thresholds:
                if zoom_min <= thresh <= zoom_max:
                    zoom_ax.axvline(thresh, color='#E63946', linestyle='--', linewidth=1.2, alpha=0.85)
    else:
        for ax in hist_axes:
            ax.set_axis_off()

    # Optional self-ICI histograms per selected call type.
    if self_ici_call_types is None:
        self_ici_call_types = []
    if self_ici_gaps_by_type is None:
        self_ici_gaps_by_type = {}
    for ax, call_type in zip(extra_hist_axes, self_ici_call_types[:len(extra_hist_axes)]):
        gaps = self_ici_gaps_by_type.get(call_type, np.array([]))
        zoom_min = 0.05
        zoom_max = 3.0
        zoom_bins = np.geomspace(zoom_min, zoom_max, 30)
        zoom_data = gaps[(gaps >= zoom_min) & (gaps <= zoom_max)] if len(gaps) > 0 else np.array([])
        if len(zoom_data) > 0:
            ax.hist(zoom_data, bins=zoom_bins, color='#CED4DA', edgecolor='white', linewidth=0.4)
        ax.set_xscale('log')
        ax.set_xlim(zoom_min, zoom_max)
        if self_ici_ymax_by_type is not None:
            shared_y = self_ici_ymax_by_type.get(call_type, 0)
            if shared_y > 0:
                ax.set_ylim(0, shared_y * 1.05)
        zoom_ticks = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
        ax.set_xticks(zoom_ticks)
        ax.set_xticklabels([f'{t:g}' for t in zoom_ticks], fontsize=10)
        n_self = len(zoom_data)
        ax.set_title(f'Self-ICI: {call_type} -> {call_type}  (n={n_self:,})', fontsize=12)
        ax.set_xlabel('Interval (sec)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if thresholds:
            for thresh in thresholds:
                if zoom_min <= thresh <= zoom_max:
                    ax.axvline(thresh, color='#E63946', linestyle='--', linewidth=1.0, alpha=0.7)
    for ax in extra_hist_axes[len(self_ici_call_types[:len(extra_hist_axes)]):]:
        ax.set_axis_off()

    interval_specs = [
        (count_left, prob_left, interval_left),
        (count_mid, prob_mid, interval_mid),
        (count_right, prob_right, interval_right),
    ]

    count_main_axes = []
    count_diff_axes = []
    prob_main_axes = []
    prob_diff_axes = []
    w_main_axes = []
    w_diff_axes = []
    count_diff_imgs = []
    prob_diff_imgs = []
    w_main_imgs = []
    w_diff_imgs = []
    count_diff_max = 0.0
    prob_diff_max = 0.0
    weighted_prob_max = 0.0
    weighted_diff_max = 0.0

    for g_idx, (cdict, pdict, _interval_val) in enumerate(interval_specs):
        c_arena = cdict['arena']
        c_und = cdict['underground']
        c_diff = c_arena - c_und
        p_arena = pdict['arena']
        p_und = pdict['underground']
        p_diff = p_arena - p_und

        def joint(cdf):
            """Joint probability of transition: count(a->b) / total_transitions.
            Sums to 1 over the whole matrix when there are any transitions."""
            total = float(cdf.values.sum())
            if total > 0:
                return cdf.astype(float) / total
            return pd.DataFrame(0.0, index=cdf.index, columns=cdf.columns)

        w_arena = joint(c_arena)
        w_und = joint(c_und)
        w_diff = w_arena - w_und

        count_diff_max = max(count_diff_max, float(np.abs(c_diff.values).max()))
        prob_diff_max = max(prob_diff_max, float(np.abs(p_diff.values).max()))
        weighted_prob_max = max(weighted_prob_max, float(w_arena.values.max()), float(w_und.values.max()))
        weighted_diff_max = max(weighted_diff_max, float(np.abs(w_diff.values).max()))

        col0 = g_idx * 3
        panels = [
            ('Arena', c_arena, p_arena, w_arena),
            ('Underground', c_und, p_und, w_und),
            ('Diff (A-U)', c_diff, p_diff, w_diff),
        ]
        for local_idx, (subtitle, cdf, pdf, wdf) in enumerate(panels):
            col_idx = col0 + local_idx
            cax = count_axes[col_idx]
            pax = prob_axes[col_idx]
            wax = prob_global_axes[col_idx]

            cimg = cax.imshow(np.sign(cdf.values) * np.log1p(np.abs(cdf.values)) if local_idx == 2 else np.log1p(cdf.values),
                              cmap='coolwarm' if local_idx == 2 else 'viridis',
                              vmin=-1 if local_idx == 2 else 0,
                              vmax=1 if local_idx == 2 else global_log_count_max,
                              aspect='equal')
            cax.set_title(subtitle)
            cax.set_xticks(range(len(cdf.columns)))
            cax.set_xticklabels(cdf.columns, rotation=90)
            cax.set_yticks(range(len(cdf.index)))
            cax.set_yticklabels(cdf.index if col_idx in (0, 3, 6) else [])

            pimg = pax.imshow(pdf.values, cmap='coolwarm' if local_idx == 2 else 'magma',
                              vmin=-1 if local_idx == 2 else 0,
                              vmax=1, aspect='equal')
            pax.set_title(subtitle)
            pax.set_xticks(range(len(pdf.columns)))
            pax.set_xticklabels(pdf.columns, rotation=90)
            pax.set_yticks(range(len(pdf.index)))
            pax.set_yticklabels(pdf.index if col_idx in (0, 3, 6) else [])

            wimg = wax.imshow(wdf.values, cmap='coolwarm' if local_idx == 2 else 'cividis',
                              vmin=-1 if local_idx == 2 else 0,
                              vmax=1, aspect='equal')
            wax.set_title(subtitle)
            wax.set_xticks(range(len(wdf.columns)))
            wax.set_xticklabels(wdf.columns, rotation=90)
            wax.set_yticks(range(len(wdf.index)))
            wax.set_yticklabels(wdf.index if col_idx in (0, 3, 6) else [])

            if local_idx == 2:
                count_diff_axes.append(cax)
                prob_diff_axes.append(pax)
                w_diff_axes.append(wax)
                count_diff_imgs.append(cimg)
                prob_diff_imgs.append(pimg)
                w_diff_imgs.append(wimg)
            else:
                count_main_axes.append(cax)
                prob_main_axes.append(pax)
                w_main_axes.append(wax)
                w_main_imgs.append(wimg)

    cdiff_vmax = count_diff_max if count_diff_max > 0 else 1.0
    pdiff_vmax = prob_diff_max if prob_diff_max > 0 else 1.0
    wmain_vmax = weighted_prob_max if weighted_prob_max > 0 else 1.0
    wdiff_vmax = weighted_diff_max if weighted_diff_max > 0 else 1.0
    for img in count_diff_imgs:
        img.set_clim(-np.log1p(cdiff_vmax), np.log1p(cdiff_vmax))
    for img in prob_diff_imgs:
        img.set_clim(-pdiff_vmax, pdiff_vmax)
    for img in w_main_imgs:
        img.set_clim(0, wmain_vmax)
    for img in w_diff_imgs:
        img.set_clim(-wdiff_vmax, wdiff_vmax)

    # Place all matrix colorbars in the far-right margin (main + diff per row).
    def _row_y_bounds(row_axes):
        return min(ax.get_position().y0 for ax in row_axes), max(ax.get_position().y1 for ax in row_axes)

    c_y0, c_y1 = _row_y_bounds(count_axes)
    p_y0, p_y1 = _row_y_bounds(prob_axes)
    w_y0, w_y1 = _row_y_bounds(prob_global_axes)
    cb_w = 0.007
    cb_gap = 0.006
    cb_right = 0.922
    cb_x_main = cb_right - cb_w
    cb_x_diff = cb_x_main - cb_gap - cb_w - 0.009

    cax_count_main = fig.add_axes([cb_x_main, c_y0, cb_w, c_y1 - c_y0])
    cax_count_diff = fig.add_axes([cb_x_diff, c_y0, cb_w, c_y1 - c_y0])
    cax_prob_main = fig.add_axes([cb_x_main, p_y0, cb_w, p_y1 - p_y0])
    cax_prob_diff = fig.add_axes([cb_x_diff, p_y0, cb_w, p_y1 - p_y0])
    cax_w_main = fig.add_axes([cb_x_main, w_y0, cb_w, w_y1 - w_y0])
    cax_w_diff = fig.add_axes([cb_x_diff, w_y0, cb_w, w_y1 - w_y0])

    count_cbar = fig.colorbar(count_main_axes[0].images[0], cax=cax_count_main)
    ticks = count_cbar.get_ticks()
    count_cbar.set_ticks(ticks)
    count_cbar.set_ticklabels([f'log1p={t:g}, n_calls~{int(np.expm1(t)):,}' for t in ticks])
    count_diff_cbar = fig.colorbar(count_diff_axes[0].images[0], cax=cax_count_diff)
    count_diff_cbar.set_label('Diff (Arena - Underground), log-scaled')

    prob_cbar = fig.colorbar(prob_main_axes[0].images[0], cax=cax_prob_main)
    prob_cbar.set_label('Transition probability')
    prob_diff_cbar = fig.colorbar(prob_diff_axes[0].images[0], cax=cax_prob_diff)
    prob_diff_cbar.set_label('Diff (Arena - Underground)')

    prob_global_cbar = fig.colorbar(w_main_axes[0].images[0], cax=cax_w_main)
    prob_global_cbar.set_label('Joint transition probability  count(a->b) / total transitions')
    prob_global_diff_cbar = fig.colorbar(w_diff_axes[0].images[0], cax=cax_w_diff)
    prob_global_diff_cbar.set_label('Joint diff (Arena - Underground)')

    all_axes = top_axes + top_axes_global + hist_axes + extra_hist_axes + count_axes + prob_axes + prob_global_axes
    grid_left = min(ax.get_position().x0 for ax in all_axes)
    grid_right = max(ax.get_position().x1 for ax in all_axes)
    cx = (grid_left + grid_right) / 2.0
    top_left = min(ax.get_position().x0 for ax in top_axes)
    top_right = max(ax.get_position().x1 for ax in top_axes)
    top_cx = (top_left + top_right) / 2.0
    r0_top = max(ax.get_position().y1 for ax in top_axes)
    r0g_top = max(ax.get_position().y1 for ax in top_axes_global)
    r1_top = max(ax.get_position().y1 for ax in count_axes)
    r2_top = max(ax.get_position().y1 for ax in prob_axes)
    r3_top = max(ax.get_position().y1 for ax in prob_global_axes)
    r0_h = np.mean([ax.get_position().height for ax in top_axes])
    r0g_h = np.mean([ax.get_position().height for ax in top_axes_global])
    r1_h = np.mean([ax.get_position().height for ax in count_axes])
    r2_h = np.mean([ax.get_position().height for ax in prob_axes])
    r3_h = np.mean([ax.get_position().height for ax in prob_global_axes])

    # Main row super-titles
    fig.text(top_cx, r0_top + 0.4 * r0_h, 'Call proportions (within location)', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(top_cx, r0g_top + 0.4 * r0g_h, 'Call proportions (across all locations)', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r1_top + 0.2 * r1_h, 'Transition counts', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r2_top + 0.2 * r2_h, 'Transition probabilities (row-normalized)', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r3_top + 0.2 * r3_h, 'Joint transition probability (count(a->b) / total transitions)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Interval super-titles per row: three interval groups (left/mid/right)
    c_left_cx = (count_axes[0].get_position().x0 + count_axes[2].get_position().x1) / 2.0
    c_mid_cx = (count_axes[3].get_position().x0 + count_axes[5].get_position().x1) / 2.0
    c_right_cx = (count_axes[6].get_position().x0 + count_axes[8].get_position().x1) / 2.0
    p_left_cx = (prob_axes[0].get_position().x0 + prob_axes[2].get_position().x1) / 2.0
    p_mid_cx = (prob_axes[3].get_position().x0 + prob_axes[5].get_position().x1) / 2.0
    p_right_cx = (prob_axes[6].get_position().x0 + prob_axes[8].get_position().x1) / 2.0
    pg_left_cx = (prob_global_axes[0].get_position().x0 + prob_global_axes[2].get_position().x1) / 2.0
    pg_mid_cx = (prob_global_axes[3].get_position().x0 + prob_global_axes[5].get_position().x1) / 2.0
    pg_right_cx = (prob_global_axes[6].get_position().x0 + prob_global_axes[8].get_position().x1) / 2.0
    left_txt = interval_left_label or (f'{interval_left}sec inter-call-interval' if interval_left is not None else 'inter-call-interval')
    mid_txt = interval_mid_label or (f'{interval_mid}sec inter-call-interval' if interval_mid is not None else 'inter-call-interval')
    right_txt = interval_right_label or (f'{interval_right}sec inter-call-interval' if interval_right is not None else 'inter-call-interval')
    fig.text(c_left_cx, r1_top + 0.08 * r1_h, left_txt, ha='center', va='center', fontsize=10)
    fig.text(c_mid_cx, r1_top + 0.08 * r1_h, mid_txt, ha='center', va='center', fontsize=10)
    fig.text(c_right_cx, r1_top + 0.08 * r1_h, right_txt, ha='center', va='center', fontsize=10)
    fig.text(p_left_cx, r2_top + 0.08 * r2_h, left_txt, ha='center', va='center', fontsize=10)
    fig.text(p_mid_cx, r2_top + 0.08 * r2_h, mid_txt, ha='center', va='center', fontsize=10)
    fig.text(p_right_cx, r2_top + 0.08 * r2_h, right_txt, ha='center', va='center', fontsize=10)
    fig.text(pg_left_cx, r3_top + 0.08 * r3_h, left_txt, ha='center', va='center', fontsize=10)
    fig.text(pg_mid_cx, r3_top + 0.08 * r3_h, mid_txt, ha='center', va='center', fontsize=10)
    fig.text(pg_right_cx, r3_top + 0.08 * r3_h, right_txt, ha='center', va='center', fontsize=10)

    fig.suptitle(figure_title or '', fontsize=22, fontweight='bold', y=0.995)
    if plot_note:
        fig.text(0.5, 0.972, plot_note, ha='center', va='top', fontsize=10)
    plot_path = os.path.join(output_dir_left, save_name)
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to: {os.path.abspath(plot_path)}")
