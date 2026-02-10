import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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


def _build_group_map(group_call_types=False, extra_group_map=None):
    group_map = {}
    if group_call_types:
        group_map.update({
            'ufm': 'high-freq',
            'tilda': 'high-freq',
            'hat': 'high-freq',
        })
    if extra_group_map:
        group_map.update(extra_group_map)
    return group_map


def _effective_call_type_order(base_order, call_group_map):
    if not call_group_map:
        return list(base_order)
    order = []
    for call in base_order:
        grouped = call_group_map.get(call, call)
        if grouped not in order:
            order.append(grouped)
    return order


def _filter_by_time_window(df, time_window='all'):
    if time_window == 'all':
        return df
    if time_window not in {'day', 'night'}:
        raise ValueError(f"Unsupported time_window: {time_window}")
    if 'start_time_real' not in df.columns:
        raise ValueError("`start_time_real` column is required for day/night filtering.")

    filtered = df.copy()
    filtered['_start_time_real_dt'] = pd.to_datetime(filtered['start_time_real'], errors='coerce')
    filtered = filtered.dropna(subset=['_start_time_real_dt'])
    hours = filtered['_start_time_real_dt'].dt.hour

    if time_window == 'day':
        # Daytime: 08:00 to 19:59
        filtered = filtered[(hours >= 8) & (hours < 20)].copy()
    else:
        # Nighttime: 20:00 to 07:59
        filtered = filtered[(hours >= 20) | (hours < 8)].copy()

    return filtered


def estimate_total_hours(csv_files, time_window='all'):
    if isinstance(csv_files, str):
        csv_paths = [csv_files]
    else:
        csv_paths = list(csv_files)

    def overlap_seconds(a_start, a_end, b_start, b_end):
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        return max(0.0, (end - start).total_seconds())

    def span_seconds_in_window(span_start, span_end, window):
        if window == 'all':
            return max(0.0, (span_end - span_start).total_seconds())
        if window not in {'day', 'night'}:
            raise ValueError(f"Unsupported time_window: {window}")

        total = 0.0
        # Iterate day by day across this experiment span.
        day = span_start.date()
        last_day = span_end.date()
        while day <= last_day:
            day_start = datetime.combine(day, datetime.min.time())
            next_day = day_start + timedelta(days=1)
            if window == 'day':
                # 08:00-20:00
                w_start = day_start + timedelta(hours=8)
                w_end = day_start + timedelta(hours=20)
                total += overlap_seconds(span_start, span_end, w_start, w_end)
            else:
                # Night is 20:00-24:00 and 00:00-08:00
                w1_start = day_start
                w1_end = day_start + timedelta(hours=8)
                w2_start = day_start + timedelta(hours=20)
                w2_end = next_day
                total += overlap_seconds(span_start, span_end, w1_start, w1_end)
                total += overlap_seconds(span_start, span_end, w2_start, w2_end)
            day = day + timedelta(days=1)
        return total

    total_seconds = 0.0
    for csv_path in csv_paths:
        exp_df = pd.read_csv(csv_path)
        if not {'start_time_real', 'stop_time_real'}.issubset(exp_df.columns):
            continue

        starts = pd.to_datetime(exp_df['start_time_real'], errors='coerce')
        stops = pd.to_datetime(exp_df['stop_time_real'], errors='coerce')
        valid = (~starts.isna()) & (~stops.isna()) & (stops >= starts)
        if not valid.any():
            continue

        span_start = starts[valid].min().to_pydatetime()
        span_end = stops[valid].max().to_pydatetime()
        total_seconds += span_seconds_in_window(span_start, span_end, time_window)

    return total_seconds / 3600.0


def collect_inter_call_gaps(csv_files, time_window='all'):
    if isinstance(csv_files, str):
        csv_paths = [csv_files]
    else:
        csv_paths = list(csv_files)

    gaps = []
    for csv_path in csv_paths:
        source_name = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        df['_source_exp'] = source_name
        df = _filter_by_time_window(df, time_window=time_window)
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


def compute_shared_log_count_max(output_dirs, arena_names=('arena', 'underground')):
    shared_max = 0.0
    for out_dir in output_dirs:
        for arena in arena_names:
            count_path = os.path.join(out_dir, f'counts_{arena}.csv')
            if os.path.exists(count_path):
                cdf = pd.read_csv(count_path, index_col=0)
                shared_max = max(shared_max, float(np.log1p(cdf.values).max()))
    return shared_max


def compute_and_save_arena_transitions(
    csv_files,
    inter_call_interval_sec,
    output_dir='transition_results',
    time_window='all',
    min_inter_call_interval_sec=None,
    call_group_map=None,
    call_type_order=None
):
    """
    Parses one or more consolidated step4 CSVs by channel, computes transitions,
    and saves results to a specified directory.

    Notes:
    - `csv_files` can be a single path or a list of paths.
    - Transitions are computed within each source CSV independently
      (no transitions are counted across experiment-file boundaries).
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

        # 2. Optional daytime/nighttime filtering
        df = _filter_by_time_window(df, time_window=time_window)

        # 3. Cleaning: Remove noise and handle time scales
        df = df.dropna(subset=['event_type', 'start_time_experiment_sec']).copy()
        df['event_type'] = df['event_type'].map(_canonicalize_call_type)
        if call_group_map:
            df['event_type'] = df['event_type'].map(lambda x: call_group_map.get(x, x))
        df = df[df['event_type'] != 'noise']
        if call_type_order is None:
            call_type_order = _effective_call_type_order(CALL_TYPE_ORDER, call_group_map or {})
        df = df[df['event_type'].isin(call_type_order)]
        if df.empty:
            raise ValueError(f"No rows remain after applying time filter '{time_window}'.")

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

                    lower_ok = True if min_inter_call_interval_sec is None else (gap > min_inter_call_interval_sec)
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
        for output_name, channel_ids in output_groups.items():
            combined = pd.DataFrame(0, index=all_call_types, columns=all_call_types)
            combined_call_counts = pd.Series(0, index=all_call_types, dtype=int)
            for channel_id in channel_ids:
                combined = combined.add(channel_matrices[channel_id], fill_value=0)
                combined_call_counts = combined_call_counts.add(channel_call_counts[channel_id], fill_value=0)
            combined = combined.astype(int)
            combined_call_counts = combined_call_counts.astype(int)

            # Save raw counts
            count_path = os.path.join(output_dir, f'counts_{output_name}.csv')
            combined.to_csv(count_path)

            # Save per-call-type totals
            calls_path = os.path.join(output_dir, f'call_counts_{output_name}.csv')
            combined_call_counts.rename('n_calls').to_csv(calls_path, header=True)

            # Save probabilities (row-normalized)
            prob_matrix = combined.div(combined.sum(axis=1), axis=0).fillna(0)
            prob_path = os.path.join(output_dir, f'probabilities_{output_name}.csv')
            prob_matrix.to_csv(prob_path)

        print(f"Success! Transition matrices saved in: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_transition_matrices(
    output_dir_left,
    output_dir_right,
    arena_names=('arena', 'underground'),
    save_name='transition_matrices_overview.png',
    plot_note=None,
    row_y_shift=None,
    interval_left=None,
    interval_right=None,
    interval_left_label=None,
    interval_right_label=None,
    inter_call_gaps=None,
    shared_log_count_max=None,
    call_type_order=None
):
    """
    Row 1: call proportions (2 panels, unchanged style).
    Row 2: transition counts (4 panels: 2 for left interval, 2 for right interval).
    Row 3: transition probabilities (same 4-panel layout).
    """
    if len(arena_names) != 2:
        raise ValueError("Compare layout expects exactly two arena names.")
    if call_type_order is None:
        call_type_order = CALL_TYPE_ORDER

    fig = plt.figure(figsize=(22, 16), constrained_layout=False)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.2, 1.0, 1.0], hspace=0.03, wspace=0.22)
    fig.subplots_adjust(top=0.93)

    # Keep top row in left half of the page (same practical width as pre-compare layout).
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    hist_gs = gs[0, 2:4].subgridspec(1, 2, wspace=0.25)
    hist_axes = [fig.add_subplot(hist_gs[0, 0]), fig.add_subplot(hist_gs[0, 1])]
    count_axes = [fig.add_subplot(gs[1, c]) for c in range(4)]
    prob_axes = [fig.add_subplot(gs[2, c]) for c in range(4)]

    if row_y_shift is None:
        row_y_shift = {0: -0.08, 1: -0.13, 2: -0.1}
    for r, row_axes in {0: top_axes + hist_axes, 1: count_axes, 2: prob_axes}.items():
        dy = float(row_y_shift.get(r, 0.0))
        if dy != 0.0:
            for ax in row_axes:
                p = ax.get_position()
                ax.set_position([p.x0, p.y0 + dy, p.width, p.height])

    # Bring each Underground panel a bit closer to its paired Arena panel.
    pair_dx = -0.03
    for ax in (count_axes[1], count_axes[3], prob_axes[1], prob_axes[3]):
        p = ax.get_position()
        ax.set_position([p.x0 + pair_dx, p.y0, p.width, p.height])

    call_count_series = {}
    count_left = {}
    prob_left = {}
    count_right = {}
    prob_right = {}
    global_log_count_max = 0.0

    for arena in arena_names:
        call_count_series[arena] = pd.read_csv(
            os.path.join(output_dir_left, f'call_counts_{arena}.csv'), index_col=0
        )['n_calls']
        count_left[arena] = pd.read_csv(os.path.join(output_dir_left, f'counts_{arena}.csv'), index_col=0)
        prob_left[arena] = pd.read_csv(os.path.join(output_dir_left, f'probabilities_{arena}.csv'), index_col=0)
        count_right[arena] = pd.read_csv(os.path.join(output_dir_right, f'counts_{arena}.csv'), index_col=0)
        prob_right[arena] = pd.read_csv(os.path.join(output_dir_right, f'probabilities_{arena}.csv'), index_col=0)
        global_log_count_max = max(global_log_count_max, np.log1p(count_left[arena].values).max())
        global_log_count_max = max(global_log_count_max, np.log1p(count_right[arena].values).max())
    if shared_log_count_max is not None:
        global_log_count_max = float(shared_log_count_max)

    for i, arena in enumerate(arena_names):
        area_title = 'Arena' if arena == 'arena' else 'Underground'
        calls_s = call_count_series[arena].reindex(call_type_order, fill_value=0)
        total_calls = int(calls_s.sum())
        call_props = calls_s / total_calls if total_calls > 0 else calls_s.astype(float)

        x = np.arange(len(calls_s.index))
        bars = top_axes[i].bar(x, call_props.values, color=[_call_color(ct) for ct in calls_s.index])
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

    # Top-right: two inter-call-interval histograms from the input data for this plot.
    if inter_call_gaps is not None and len(inter_call_gaps) > 0:
        positive_gaps = inter_call_gaps[inter_call_gaps > 0]
        hist_ax = hist_axes[0]
        if len(positive_gaps) > 0:
            bins = np.geomspace(max(1e-3, positive_gaps.min()), positive_gaps.max(), 40)
            hist_ax.hist(positive_gaps, bins=bins, color='#6C757D', edgecolor='white', linewidth=0.4)
            hist_ax.set_xscale('log')
        else:
            hist_ax.hist(inter_call_gaps, bins=30, color='#6C757D', edgecolor='white', linewidth=0.4)
        hist_ax.set_title('Inter-call-interval (full)')
        hist_ax.set_xlabel('Interval (sec)')
        hist_ax.set_ylabel('Count')
        hist_ax.spines['top'].set_visible(False)
        hist_ax.spines['right'].set_visible(False)

        zoom_ax = hist_axes[1]
        zoom_min = 0.05
        zoom_max = 3.0
        zoom_bin_w = 0.1
        zoom_data = inter_call_gaps[(inter_call_gaps >= 0.0) & (inter_call_gaps <= zoom_max)]
        if len(zoom_data) > 0:
            # Fixed 0.1s bin edges aligned to 0 so first bin is [0.0, 0.1).
            zoom_bins = np.arange(0.0, zoom_max + zoom_bin_w, zoom_bin_w)
            zoom_ax.hist(zoom_data, bins=zoom_bins, color='#ADB5BD', edgecolor='white', linewidth=0.4)
        zoom_ax.set_xlim(zoom_min, zoom_max)
        zoom_ticks = np.arange(0.1, zoom_max + 1e-9, 0.1)
        zoom_tick_labels = [f'{t:.1f}' if abs((t / 0.5) - round(t / 0.5)) < 1e-9 else '' for t in zoom_ticks]
        zoom_ax.set_xticks(zoom_ticks)
        zoom_ax.set_xticklabels(zoom_tick_labels)
        zoom_ax.set_title('Inter-call-interval (0.05-3s)')
        zoom_ax.set_xlabel('Interval (sec)')
        zoom_ax.set_ylabel('Count')
        zoom_ax.spines['top'].set_visible(False)
        zoom_ax.spines['right'].set_visible(False)
    else:
        for ax in hist_axes:
            ax.set_axis_off()

    bottom_specs = [
        (count_axes[0], prob_axes[0], 'arena', count_left, prob_left, interval_left),
        (count_axes[1], prob_axes[1], 'underground', count_left, prob_left, interval_left),
        (count_axes[2], prob_axes[2], 'arena', count_right, prob_right, interval_right),
        (count_axes[3], prob_axes[3], 'underground', count_right, prob_right, interval_right),
    ]

    for col_idx, (count_ax, prob_ax, arena, cdict, pdict, interval_val) in enumerate(bottom_specs):
        area_title = 'Arena' if arena == 'arena' else 'Underground'
        subtitle = area_title
        cdf = cdict[arena]
        pdf = pdict[arena]

        count_img = count_ax.imshow(np.log1p(cdf.values), cmap='viridis', vmin=0, vmax=global_log_count_max, aspect='equal')
        count_ax.set_title(subtitle)
        count_ax.set_xticks(range(len(cdf.columns)))
        count_ax.set_xticklabels(cdf.columns, rotation=90)
        count_ax.set_yticks(range(len(cdf.index)))
        count_ax.set_yticklabels(cdf.index if col_idx in (0, 2) else [])

        prob_img = prob_ax.imshow(pdf.values, cmap='magma', vmin=0, vmax=1, aspect='equal')
        prob_ax.set_title(subtitle)
        prob_ax.set_xticks(range(len(pdf.columns)))
        prob_ax.set_xticklabels(pdf.columns, rotation=90)
        prob_ax.set_yticks(range(len(pdf.index)))
        prob_ax.set_yticklabels(pdf.index if col_idx in (0, 2) else [])

    count_cbar = fig.colorbar(count_img, ax=count_axes, fraction=0.012, pad=0.01, shrink=0.82)
    ticks = count_cbar.get_ticks()
    count_cbar.set_ticks(ticks)
    count_cbar.set_ticklabels([f'log1p={t:g}, n_calls~{int(np.expm1(t)):,}' for t in ticks])

    prob_cbar = fig.colorbar(prob_img, ax=prob_axes, fraction=0.012, pad=0.01, shrink=0.82)
    prob_cbar.set_label('Transition probability')

    all_axes = top_axes + hist_axes + count_axes + prob_axes
    grid_left = min(ax.get_position().x0 for ax in all_axes)
    grid_right = max(ax.get_position().x1 for ax in all_axes)
    cx = (grid_left + grid_right) / 2.0
    top_left = min(ax.get_position().x0 for ax in top_axes)
    top_right = max(ax.get_position().x1 for ax in top_axes)
    top_cx = (top_left + top_right) / 2.0
    r0_top = max(ax.get_position().y1 for ax in top_axes)
    r1_top = max(ax.get_position().y1 for ax in count_axes)
    r2_top = max(ax.get_position().y1 for ax in prob_axes)
    r0_h = np.mean([ax.get_position().height for ax in top_axes])
    r1_h = np.mean([ax.get_position().height for ax in count_axes])
    r2_h = np.mean([ax.get_position().height for ax in prob_axes])

    # Main row super-titles
    fig.text(top_cx, r0_top + 0.4 * r0_h, 'Call proportions', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r1_top + 0.2 * r1_h, 'Transition counts', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r2_top + 0.2 * r2_h, 'Transition probabilities (normalized per row)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Interval super-titles per row: left pair (300s) vs right pair (0.1s)
    c_left_cx = (count_axes[0].get_position().x0 + count_axes[1].get_position().x1) / 2.0
    c_right_cx = (count_axes[2].get_position().x0 + count_axes[3].get_position().x1) / 2.0
    p_left_cx = (prob_axes[0].get_position().x0 + prob_axes[1].get_position().x1) / 2.0
    p_right_cx = (prob_axes[2].get_position().x0 + prob_axes[3].get_position().x1) / 2.0
    left_txt = interval_left_label or (f'{interval_left}sec inter-call-interval' if interval_left is not None else 'inter-call-interval')
    right_txt = interval_right_label or (f'{interval_right}sec inter-call-interval' if interval_right is not None else 'inter-call-interval')
    fig.text(c_left_cx, r1_top + 0.08 * r1_h, left_txt, ha='center', va='center', fontsize=10)
    fig.text(c_right_cx, r1_top + 0.08 * r1_h, right_txt, ha='center', va='center', fontsize=10)
    fig.text(p_left_cx, r2_top + 0.08 * r2_h, left_txt, ha='center', va='center', fontsize=10)
    fig.text(p_right_cx, r2_top + 0.08 * r2_h, right_txt, ha='center', va='center', fontsize=10)

    fig.suptitle('', fontsize=14, y=0.995)
    if plot_note:
        fig.text(0.5, 0.972, plot_note, ha='center', va='top', fontsize=10)
    plot_path = os.path.join(output_dir_left, save_name)
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to: {os.path.abspath(plot_path)}")


# Run the script with full input/output paths
date_exp = '2025_10'
exp_list = [273,275,276]  # e.g., [275, 276, 277]
short_gap_sec = 0.3
long_gap_sec = 300
group_call_types = True
# Add future consolidations here, e.g. {'warble': 'high-freq'} or {'mnt': 'mnt-family'}
extra_call_group_map = {}
call_group_map = _build_group_map(group_call_types=group_call_types, extra_group_map=extra_call_group_map)
effective_call_type_order = _effective_call_type_order(CALL_TYPE_ORDER, call_group_map)

input_csvs = [
    fr"Z:\ginosar\Processed_data\Audio\{date_exp}\{exp}\step4_assigned_location.csv"
    for exp in exp_list
]

exp_names = ", ".join(str(e) for e in exp_list)

# All data (keeps your current plot behavior)
output_folder_all_left = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_300s"
output_folder_all_right = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_0p1s"
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=long_gap_sec,
    output_dir=output_folder_all_left,
    time_window='all',
    min_inter_call_interval_sec=short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=short_gap_sec,
    output_dir=output_folder_all_right,
    time_window='all',
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order
)
hours_all = estimate_total_hours(input_csvs, time_window='all')
gaps_all = collect_inter_call_gaps(input_csvs, time_window='all')
plot_note_all = (
    f"Exps: {exp_names} "
    f"| Total analyzed duration: {hours_all:.2f} h"
)

# Daytime only (08:00-20:00)
output_folder_day_left = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_daytime_300s"
output_folder_day_right = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_daytime_0p1s"
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=long_gap_sec,
    output_dir=output_folder_day_left,
    time_window='day',
    min_inter_call_interval_sec=short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=short_gap_sec,
    output_dir=output_folder_day_right,
    time_window='day',
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order
)
hours_day = estimate_total_hours(input_csvs, time_window='day')
gaps_day = collect_inter_call_gaps(input_csvs, time_window='day')
plot_note_day = (
    f"Exps: {exp_names} "
    f"| Daytime (08:00-20:00) | Total analyzed duration: {hours_day:.2f} h"
)

# Nighttime only (20:00-08:00)
output_folder_night_left = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_nighttime_300s"
output_folder_night_right = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_nighttime_0p1s"
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=long_gap_sec,
    output_dir=output_folder_night_left,
    time_window='night',
    min_inter_call_interval_sec=short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=short_gap_sec,
    output_dir=output_folder_night_right,
    time_window='night',
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order
)
hours_night = estimate_total_hours(input_csvs, time_window='night')
gaps_night = collect_inter_call_gaps(input_csvs, time_window='night')
plot_note_night = (
    f"Exps: {exp_names} "
    f"| Nighttime (20:00-08:00) | Total analyzed duration: {hours_night:.2f} h"
)

# Shared color scale across all 3 figures (all/day/night) for the counts row.
shared_log_count_max = compute_shared_log_count_max([
    output_folder_all_left,
    output_folder_all_right,
    output_folder_day_left,
    output_folder_day_right,
    output_folder_night_left,
    output_folder_night_right
], arena_names=('arena', 'underground'))

plot_transition_matrices(
    output_folder_all_left,
    output_folder_all_right,
    save_name='transition_matrices_overview.png',
    plot_note=plot_note_all,
    interval_left=long_gap_sec,
    interval_right=short_gap_sec,
    interval_left_label=f'{short_gap_sec}s < inter-call-interval <= {long_gap_sec}s',
    interval_right_label=f'inter-call-interval <= {short_gap_sec}s',
    inter_call_gaps=gaps_all,
    shared_log_count_max=shared_log_count_max,
    call_type_order=effective_call_type_order
)
plot_transition_matrices(
    output_folder_day_left,
    output_folder_day_right,
    save_name='transition_matrices_daytime.png',
    plot_note=plot_note_day,
    interval_left=long_gap_sec,
    interval_right=short_gap_sec,
    interval_left_label=f'{short_gap_sec}s < inter-call-interval <= {long_gap_sec}s',
    interval_right_label=f'inter-call-interval <= {short_gap_sec}s',
    inter_call_gaps=gaps_day,
    shared_log_count_max=shared_log_count_max,
    call_type_order=effective_call_type_order
)
plot_transition_matrices(
    output_folder_night_left,
    output_folder_night_right,
    save_name='transition_matrices_nighttime.png',
    plot_note=plot_note_night,
    interval_left=long_gap_sec,
    interval_right=short_gap_sec,
    interval_left_label=f'{short_gap_sec}s < inter-call-interval <= {long_gap_sec}s',
    interval_right_label=f'inter-call-interval <= {short_gap_sec}s',
    inter_call_gaps=gaps_night,
    shared_log_count_max=shared_log_count_max,
    call_type_order=effective_call_type_order
)
