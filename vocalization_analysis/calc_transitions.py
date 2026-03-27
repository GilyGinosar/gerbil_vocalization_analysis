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


def collect_self_inter_call_gaps(csv_files, target_call_type, time_window='all', call_group_map=None):
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
        df = _filter_by_time_window(df, time_window=time_window)
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
    time_window='all',
    min_inter_call_interval_sec=None,
    call_group_map=None,
    call_type_order=None,
    file_tag=''
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
    call_type_order=None,
    file_tag_left='',
    file_tag_mid='',
    file_tag_right=''
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

    fig = plt.figure(figsize=(52, 28), constrained_layout=False)
    gs = fig.add_gridspec(4, 9, height_ratios=[0.2, 1.0, 1.0, 1.0], hspace=0.10, wspace=0.24)
    fig.subplots_adjust(top=0.93)

    # Top row: manual placement so it keeps prior sizing and does not follow matrix columns.
    top_axes = [
        fig.add_axes([0.06, 0.856, 0.15, 0.094]),   # Arena proportions
        fig.add_axes([0.25, 0.856, 0.15, 0.094])    # Underground proportions
    ]
    hist_axes = [
        fig.add_axes([0.47, 0.856, 0.07, 0.094]),   # ICI full
        fig.add_axes([0.57, 0.856, 0.07, 0.094])    # ICI zoom
    ]
    extra_hist_axes = [
        fig.add_axes([0.67, 0.856, 0.055, 0.094]),
        fig.add_axes([0.735, 0.856, 0.055, 0.094]),
        fig.add_axes([0.80, 0.856, 0.055, 0.094]),
        fig.add_axes([0.865, 0.856, 0.055, 0.094]),
    ]
    count_axes = [fig.add_subplot(gs[1, c]) for c in range(9)]
    prob_axes = [fig.add_subplot(gs[2, c]) for c in range(9)]
    prob_global_axes = [fig.add_subplot(gs[3, c]) for c in range(9)]

    if row_y_shift is None:
        row_y_shift = {0: -0.08, 1: -0.13, 2: -0.1, 3: -0.08}
    for r, row_axes in {0: top_axes + hist_axes + extra_hist_axes, 1: count_axes, 2: prob_axes, 3: prob_global_axes}.items():
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

    # Optional self-ICI histograms per selected call type.
    if self_ici_call_types is None:
        self_ici_call_types = []
    if self_ici_gaps_by_type is None:
        self_ici_gaps_by_type = {}
    for ax, call_type in zip(extra_hist_axes, self_ici_call_types[:len(extra_hist_axes)]):
        gaps = self_ici_gaps_by_type.get(call_type, np.array([]))
        zoom_min = 0.05
        zoom_max = 3.0
        zoom_bin_w = 0.1
        zoom_data = gaps[(gaps >= 0.0) & (gaps <= zoom_max)] if len(gaps) > 0 else np.array([])
        if len(zoom_data) > 0:
            zoom_bins = np.arange(0.0, zoom_max + zoom_bin_w, zoom_bin_w)
            ax.hist(zoom_data, bins=zoom_bins, color='#CED4DA', edgecolor='white', linewidth=0.4)
        ax.set_xlim(zoom_min, zoom_max)
        zoom_ticks = np.arange(0.1, zoom_max + 1e-9, 0.1)
        zoom_tick_labels = [f'{t:.1f}' if abs((t / 0.5) - round(t / 0.5)) < 1e-9 else '' for t in zoom_ticks]
        ax.set_xticks(zoom_ticks)
        ax.set_xticklabels(zoom_tick_labels, fontsize=7)
        ax.set_title(f'{call_type} -> {call_type}', fontsize=8)
        ax.set_xlabel('sec', fontsize=7)
        ax.set_ylabel('n', fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
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

        def weighted(pdf, arena_key):
            src_calls = call_count_series[arena_key].reindex(pdf.index, fill_value=0).astype(float)
            total_calls = float(src_calls.sum())
            if total_calls > 0:
                return pdf.mul(src_calls / total_calls, axis=0)
            return pd.DataFrame(0.0, index=pdf.index, columns=pdf.columns)

        w_arena = weighted(p_arena, 'arena')
        w_und = weighted(p_und, 'underground')
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
    prob_global_cbar.set_label('Row probability x source prevalence')
    prob_global_diff_cbar = fig.colorbar(w_diff_axes[0].images[0], cax=cax_w_diff)
    prob_global_diff_cbar.set_label('Weighted diff (Arena - Underground)')

    all_axes = top_axes + hist_axes + extra_hist_axes + count_axes + prob_axes + prob_global_axes
    grid_left = min(ax.get_position().x0 for ax in all_axes)
    grid_right = max(ax.get_position().x1 for ax in all_axes)
    cx = (grid_left + grid_right) / 2.0
    top_left = min(ax.get_position().x0 for ax in top_axes)
    top_right = max(ax.get_position().x1 for ax in top_axes)
    top_cx = (top_left + top_right) / 2.0
    r0_top = max(ax.get_position().y1 for ax in top_axes)
    r1_top = max(ax.get_position().y1 for ax in count_axes)
    r2_top = max(ax.get_position().y1 for ax in prob_axes)
    r3_top = max(ax.get_position().y1 for ax in prob_global_axes)
    r0_h = np.mean([ax.get_position().height for ax in top_axes])
    r1_h = np.mean([ax.get_position().height for ax in count_axes])
    r2_h = np.mean([ax.get_position().height for ax in prob_axes])
    r3_h = np.mean([ax.get_position().height for ax in prob_global_axes])

    # Main row super-titles
    fig.text(top_cx, r0_top + 0.4 * r0_h, 'Call proportions', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r1_top + 0.2 * r1_h, 'Transition counts', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r2_top + 0.2 * r2_h, 'Transition probabilities (row-normalized)', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(cx, r3_top + 0.2 * r3_h, 'Transition probabilities (row-normalized x call prob.)', ha='center', va='center', fontsize=12, fontweight='bold')

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
very_short_gap_sec = 0.05
short_gap_sec = 0.3
long_gap_sec = 300
group_call_types = True
# Add future consolidations here, e.g. {'warble': 'high-freq'} or {'mnt': 'mnt-family'}
extra_call_group_map = {}
call_group_map = _build_group_map(group_call_types=group_call_types, extra_group_map=extra_call_group_map)
effective_call_type_order = _effective_call_type_order(CALL_TYPE_ORDER, call_group_map)
self_ici_call_types = ['high-freq', 'warble', 'alarm', 'dense-stack']

input_csvs = [
    fr"Z:\ginosar\Processed_data\Audio\{date_exp}\{exp}\step4_assigned_location.csv"
    for exp in exp_list
]

exp_names = ", ".join(str(e) for e in exp_list)

# All CSVs and figures go into one folder; each run uses a unique file tag.
output_folder = fr"Z:\ginosar\Processed_data\Audio\{date_exp}\combined_transitions_outputs"

tag_all_left = "all_gt0p3_le300"
tag_all_mid = "all_gt0p05_le0p3"
tag_all_right = "all_le0p05"
tag_day_left = "day_gt0p3_le300"
tag_day_mid = "day_gt0p05_le0p3"
tag_day_right = "day_le0p05"
tag_night_left = "night_gt0p3_le300"
tag_night_mid = "night_gt0p05_le0p3"
tag_night_right = "night_le0p05"

# All data
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=long_gap_sec,
    output_dir=output_folder,
    time_window='all',
    min_inter_call_interval_sec=short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_all_left
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=short_gap_sec,
    output_dir=output_folder,
    time_window='all',
    min_inter_call_interval_sec=very_short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_all_mid
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=very_short_gap_sec,
    output_dir=output_folder,
    time_window='all',
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_all_right
)
hours_all = estimate_total_hours(input_csvs, time_window='all')
gaps_all = collect_inter_call_gaps(input_csvs, time_window='all')
self_ici_gaps_all = {ct: collect_self_inter_call_gaps(input_csvs, ct, time_window='all', call_group_map=call_group_map) for ct in self_ici_call_types}
plot_note_all = f"Exps: {exp_names} | Total analyzed duration: {hours_all:.2f} h"

# Daytime only
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=long_gap_sec,
    output_dir=output_folder,
    time_window='day',
    min_inter_call_interval_sec=short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_day_left
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=short_gap_sec,
    output_dir=output_folder,
    time_window='day',
    min_inter_call_interval_sec=very_short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_day_mid
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=very_short_gap_sec,
    output_dir=output_folder,
    time_window='day',
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_day_right
)
hours_day = estimate_total_hours(input_csvs, time_window='day')
gaps_day = collect_inter_call_gaps(input_csvs, time_window='day')
self_ici_gaps_day = {ct: collect_self_inter_call_gaps(input_csvs, ct, time_window='day', call_group_map=call_group_map) for ct in self_ici_call_types}
plot_note_day = f"Exps: {exp_names} | Daytime (08:00-20:00) | Total analyzed duration: {hours_day:.2f} h"

# Nighttime only
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=long_gap_sec,
    output_dir=output_folder,
    time_window='night',
    min_inter_call_interval_sec=short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_night_left
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=short_gap_sec,
    output_dir=output_folder,
    time_window='night',
    min_inter_call_interval_sec=very_short_gap_sec,
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_night_mid
)
compute_and_save_arena_transitions(
    input_csvs,
    inter_call_interval_sec=very_short_gap_sec,
    output_dir=output_folder,
    time_window='night',
    call_group_map=call_group_map,
    call_type_order=effective_call_type_order,
    file_tag=tag_night_right
)
hours_night = estimate_total_hours(input_csvs, time_window='night')
gaps_night = collect_inter_call_gaps(input_csvs, time_window='night')
self_ici_gaps_night = {ct: collect_self_inter_call_gaps(input_csvs, ct, time_window='night', call_group_map=call_group_map) for ct in self_ici_call_types}
plot_note_night = f"Exps: {exp_names} | Nighttime (20:00-08:00) | Total analyzed duration: {hours_night:.2f} h"

# Shared count color scale across all 3 figures.
shared_log_count_max = compute_shared_log_count_max([
    (output_folder, tag_all_left), (output_folder, tag_all_mid), (output_folder, tag_all_right),
    (output_folder, tag_day_left), (output_folder, tag_day_mid), (output_folder, tag_day_right),
    (output_folder, tag_night_left), (output_folder, tag_night_mid), (output_folder, tag_night_right),
], arena_names=('arena', 'underground'))

plot_transition_matrices(
    output_folder, output_folder, output_folder,
    save_name='transition_matrices_overview.png',
    plot_note=plot_note_all,
    interval_left=long_gap_sec,
    interval_mid=short_gap_sec,
    interval_right=very_short_gap_sec,
    interval_left_label=f'{short_gap_sec}s < inter-call-interval <= {long_gap_sec}s',
    interval_mid_label=f'{very_short_gap_sec}s < inter-call-interval <= {short_gap_sec}s',
    interval_right_label=f'inter-call-interval <= {very_short_gap_sec}s',
    inter_call_gaps=gaps_all,
    self_ici_gaps_by_type=self_ici_gaps_all,
    self_ici_call_types=self_ici_call_types,
    shared_log_count_max=shared_log_count_max,
    call_type_order=effective_call_type_order,
    file_tag_left=tag_all_left,
    file_tag_mid=tag_all_mid,
    file_tag_right=tag_all_right
)
plot_transition_matrices(
    output_folder, output_folder, output_folder,
    save_name='transition_matrices_daytime.png',
    plot_note=plot_note_day,
    interval_left=long_gap_sec,
    interval_mid=short_gap_sec,
    interval_right=very_short_gap_sec,
    interval_left_label=f'{short_gap_sec}s < inter-call-interval <= {long_gap_sec}s',
    interval_mid_label=f'{very_short_gap_sec}s < inter-call-interval <= {short_gap_sec}s',
    interval_right_label=f'inter-call-interval <= {very_short_gap_sec}s',
    inter_call_gaps=gaps_day,
    self_ici_gaps_by_type=self_ici_gaps_day,
    self_ici_call_types=self_ici_call_types,
    shared_log_count_max=shared_log_count_max,
    call_type_order=effective_call_type_order,
    file_tag_left=tag_day_left,
    file_tag_mid=tag_day_mid,
    file_tag_right=tag_day_right
)
plot_transition_matrices(
    output_folder, output_folder, output_folder,
    save_name='transition_matrices_nighttime.png',
    plot_note=plot_note_night,
    interval_left=long_gap_sec,
    interval_mid=short_gap_sec,
    interval_right=very_short_gap_sec,
    interval_left_label=f'{short_gap_sec}s < inter-call-interval <= {long_gap_sec}s',
    interval_mid_label=f'{very_short_gap_sec}s < inter-call-interval <= {short_gap_sec}s',
    interval_right_label=f'inter-call-interval <= {very_short_gap_sec}s',
    inter_call_gaps=gaps_night,
    self_ici_gaps_by_type=self_ici_gaps_night,
    self_ici_call_types=self_ici_call_types,
    shared_log_count_max=shared_log_count_max,
    call_type_order=effective_call_type_order,
    file_tag_left=tag_night_left,
    file_tag_mid=tag_night_mid,
    file_tag_right=tag_night_right
)
