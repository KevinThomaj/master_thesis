import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Matplotlib global aesthetic settings
plt.rcParams['figure.max_open_warning'] = 150
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7


def get_palette():
    colors = {
        "exp_1":  "#D62728", # Bright Red: S(hist) + S(inf)
        "exp_2":  "#8B0000", # Dark Crimson: S+P(hist) + S(inf)
        "exp_3":  "#1F77B4", # Strong Blue: S(hist) + S(stream)
        "exp_4":  "#17BECF", # Cyan: S(hist) + S+P(stream)
        "exp_5":  "#2CA02C", # Forest Green: S+P(hist) + S+P(stream)
        "exp_6":  "#8CD17D", # Light Sage Green: S+P(hist) + S(stream, P frozen)
        "exp_7":  "#FF7F0E", # Vivid Orange: S+P(hist) + S(stream, no P)
        "exp_8":  "#9467BD", # Purple: S(hist) + EMA Teacher
        "exp_9":  "#00B4D8", # Sky Blue: S+P(hist) + S+P(stream) [Stop 2k]
        "exp_10": "#03045E", # Midnight Navy: S+P(hist) + S+P(stream) [Stop 4k]
        "exp_11": "#8C564B"  # Saddle Brown: S+P(hist) + S+P(stream) [Stop 8k]
    }
    
    labels = {
        "exp_1":  "Exp 1: S(hist) + S(inf)",
        "exp_2":  "Exp 2: S+P(hist) + S(inf)",
        "exp_3":  "Exp 3: S(hist) + S(stream)",
        "exp_4":  "Exp 4: S(hist) + S+P(stream)",
        "exp_5":  "Exp 5: S+P(hist) + S+P(stream)",
        "exp_6":  "Exp 6: S+P(hist) + S(stream, P frozen)",
        "exp_7":  "Exp 7: S+P(hist) + S(stream, no P)",
        "exp_8":  "Exp 8: S(hist) + EMA Teacher",
        "exp_9":  "Exp 9: S+P(hist) + S+P(stream) [Stop @ 2k/drift]",
        "exp_10": "Exp 10: S+P(hist) + S+P(stream) [Stop @ 4k/drift]",
        "exp_11": "Exp 11: S+P(hist) + S+P(stream) [Stop @ 8k/drift]"
    }
    return colors, labels


def get_exp_num(k):
    try:
        return int(k.split('_')[1])
    except:
        return 999


def format_exp_title(label_str):
    if ": " in label_str:
        prefix, desc = label_str.split(": ", 1)
        return f"{prefix}:\n{desc}"
    return label_str


def plot_cl_matrix(cl_matrix, title, ax):
    if not cl_matrix:
        return
        
    train_concepts = [entry['train_concept'] for entry in cl_matrix]
    eval_concepts = []
    for entry in cl_matrix:
        for c in entry['evaluations'].keys():
            if c not in eval_concepts:
                eval_concepts.append(c)
                
    matrix = np.zeros((len(train_concepts), len(eval_concepts)))
    for i, entry in enumerate(cl_matrix):
        for j, c in enumerate(eval_concepts):
            matrix[i, j] = entry['evaluations'].get(c, np.nan)
            
    t_labels = [f"C{i}" for i in range(len(train_concepts))]
    e_labels = [f"C{j}" for j in range(len(eval_concepts))]
    
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", 
                xticklabels=e_labels, yticklabels=t_labels, ax=ax,
                cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Evaluated Concept", fontsize=10, fontweight='bold')
    ax.set_ylabel("Trained Concept", fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)


def plot_forgetting(cl_matrix, title, ax):
    if not cl_matrix or len(cl_matrix) < 2:
        return
        
    train_concepts = [entry['train_concept'] for entry in cl_matrix]
    immediate_acc = []
    final_acc = []
    
    for i, concept in enumerate(train_concepts):
        eval_key = concept.replace('_recurrent', '') if '_recurrent' in concept else concept
        immediate_acc.append(cl_matrix[i]['evaluations'].get(eval_key, 0))
        final_acc.append(cl_matrix[-1]['evaluations'].get(eval_key, 0))
        
    x = np.arange(len(train_concepts))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, immediate_acc, width, label='Right after learning', color='#5DADE2', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, final_acc, width, label='At end of stream', color='#EC7063', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(x)
    compact_labels = [f"C{i}" for i in range(len(train_concepts))]
    ax.set_xticklabels(compact_labels, rotation=0, fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=9, facecolor='white', framealpha=0.95)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    ax.bar_label(rects1, padding=2, fmt='%.1f', fontsize=8)
    ax.bar_label(rects2, padding=2, fmt='%.1f', fontsize=8)


def export_detailed_metrics_table(config_key, config_data, labels, output_dir="."):
    records = []
    for exp_key, data in config_data.items():
        if not exp_key.startswith('exp_'): continue
        det_met = data.get('history', {}).get('detailed_metrics', {})
        exp_name = labels.get(exp_key, exp_key)
        
        concept_keys = [k for k in det_met.keys() if k != 'average_across_concepts']
        for concept in concept_keys:
            metrics = det_met[concept]
            records.append({
                'Experiment': exp_name,
                'Concept': concept,
                'First Window Acc': metrics.get('first_window_accuracy', 0),
                'After First Window Acc': metrics.get('after_first_window_accuracy', 0),
                'Final Window Acc': metrics.get('final_window_accuracy', 0),
                'Total Acc': metrics.get('total_accuracy', 0)
            })
            
    if not records:
        return
        
    df = pd.DataFrame(records)
    csv_filename = os.path.join(output_dir, f"detailed_metrics_{config_key}.csv")
    df.round(2).to_csv(csv_filename, index=False)
    print(f"[{config_key}] Exported detailed per-concept metrics to {csv_filename}")


def export_global_metrics_summary(results, labels, output_dir="."):
    records = []
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'): continue
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'): continue
            det_met = data.get('history', {}).get('detailed_metrics', {})
            exp_name = labels.get(exp_key, exp_key)
            concept_keys = [k for k in det_met.keys() if k != 'average_across_concepts']
            for concept in concept_keys:
                metrics = det_met[concept]
                records.append({
                    'Config': config_key,
                    'Experiment': exp_name,
                    'Concept': concept,
                    'First Window Acc': metrics.get('first_window_accuracy', 0),
                    'After First Window Acc': metrics.get('after_first_window_accuracy', 0),
                    'Final Window Acc': metrics.get('final_window_accuracy', 0),
                    'Total Acc': metrics.get('total_accuracy', 0)
                })
    if records:
        df = pd.DataFrame(records)
        metric_cols = ['First Window Acc', 'After First Window Acc', 'Final Window Acc', 'Total Acc']
        df_summary = df.groupby(['Experiment', 'Concept'])[metric_cols].agg(['mean', 'std']).round(2)
        summary_csv = os.path.join(output_dir, "detailed_metrics_global_summary.csv")
        df_summary.to_csv(summary_csv)
        print(f"[Global] Exported global aggregated metrics to {summary_csv}")


# =========================================================================
# GLOBAL AGGREGATED FIGURES
# =========================================================================

def plot_adaptation_speed(results, labels, save_path=None):
    exp_metrics = {}
    
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
            
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
                
            det_met = data.get('history', {}).get('detailed_metrics', {})
            concept_keys = [k for k in det_met.keys() if k != 'average_across_concepts']
            
            if not concept_keys:
                continue
                
            if exp_key not in exp_metrics:
                exp_metrics[exp_key] = {
                    'first_window': [],
                    'after_first_window': [],
                    'final_window': [],
                    'total_acc': []
                }
                
            for concept in concept_keys:
                metrics = det_met[concept]
                exp_metrics[exp_key]['first_window'].append(metrics.get('first_window_accuracy', 0.0))
                exp_metrics[exp_key]['after_first_window'].append(metrics.get('after_first_window_accuracy', 0.0))
                exp_metrics[exp_key]['final_window'].append(metrics.get('final_window_accuracy', 0.0))
                exp_metrics[exp_key]['total_acc'].append(metrics.get('total_accuracy', 0.0))

    if not exp_metrics:
        print("No detailed metrics found for adaptation speed plot.")
        return
        
    exp_keys_sorted = sorted(list(exp_metrics.keys()), key=get_exp_num)
    
    experiment_names = []
    first_window_means, first_window_stds = [], []
    after_first_window_means, after_first_window_stds = [], []
    final_window_means, final_window_stds = [], []
    total_acc_means, total_acc_stds = [], []
    
    for exp_key in exp_keys_sorted:
        metrics = exp_metrics[exp_key]
        if len(metrics['first_window']) > 0:
            experiment_names.append(labels.get(exp_key, exp_key))
            first_window_means.append(float(np.nanmean(metrics['first_window'])))
            first_window_stds.append(float(np.nanstd(metrics['first_window'])))
            after_first_window_means.append(float(np.nanmean(metrics['after_first_window'])))
            after_first_window_stds.append(float(np.nanstd(metrics['after_first_window'])))
            final_window_means.append(float(np.nanmean(metrics['final_window'])))
            final_window_stds.append(float(np.nanstd(metrics['final_window'])))
            total_acc_means.append(float(np.nanmean(metrics['total_acc'])))
            total_acc_stds.append(float(np.nanstd(metrics['total_acc'])))

    x = np.arange(len(experiment_names))
    width = 0.19
    
    fig, ax = plt.subplots(figsize=(17, 8))
    err_kwargs = {'elinewidth': 1.0, 'ecolor': '#333333', 'capthick': 1.0}

    rects1 = ax.bar(x - 1.5 * width, first_window_means, width, yerr=first_window_stds, capsize=3,
                    label='Start (First 500 Samples)', color='#2B5C8F', edgecolor='black', linewidth=0.5, error_kw=err_kwargs)
    rects2 = ax.bar(x - 0.5 * width, after_first_window_means, width, yerr=after_first_window_stds, capsize=3,
                    label='Adaptation (500–1000 Samples)', color='#E67E22', edgecolor='black', linewidth=0.5, error_kw=err_kwargs)
    rects3 = ax.bar(x + 0.5 * width, final_window_means, width, yerr=final_window_stds, capsize=3,
                    label='End of Concept (Last 500 Samples)', color='#27AE60', edgecolor='black', linewidth=0.5, error_kw=err_kwargs)
    rects4 = ax.bar(x + 1.5 * width, total_acc_means, width, yerr=total_acc_stds, capsize=3,
                    label='Total Concept Accuracy', color='#8E44AD', edgecolor='black', linewidth=0.5, error_kw=err_kwargs)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Adaptation Speed & Concept Progression\n(Mean ± Std across all Concepts & Configurations)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=20, ha='right', fontsize=9.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CCCCCC')
    
    # Calculate top y-limit with safety headroom for error bars and labels
    max_val_with_err = max(
        max(m + s for m, s in zip(first_window_means, first_window_stds)) if first_window_means else 0,
        max(m + s for m, s in zip(after_first_window_means, after_first_window_stds)) if after_first_window_means else 0,
        max(m + s for m, s in zip(final_window_means, final_window_stds)) if final_window_means else 0,
        max(m + s for m, s in zip(total_acc_means, total_acc_stds)) if total_acc_means else 0,
    )
    ax.set_ylim(0, max(115, min(130, max_val_with_err + 14)))
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=7.5, fontweight='bold', rotation=45)
    ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=7.5, fontweight='bold', rotation=45)
    ax.bar_label(rects3, padding=3, fmt='%.1f', fontsize=7.5, fontweight='bold', rotation=45)
    ax.bar_label(rects4, padding=3, fmt='%.1f', fontsize=7.5, fontweight='bold', rotation=45)
    
    fig.subplots_adjust(left=0.06, right=0.72, top=0.90, bottom=0.20)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")


def compute_smart_ylim(results, mean_curves, drift_points=None, ymin_override=None, ymax_override=None):
    """
    Dynamically computes a tight Y-axis zoom range based on the stabilized
    concept trajectory (ignoring the first ~100 initial prequential samples of each concept).
    """
    if ymin_override is not None and ymax_override is not None:
        return ymin_override, ymax_override

    global_max = float('-inf')
    segment_mins = []
    
    for exp_key, curve in mean_curves.items():
        if len(curve) == 0:
            continue
        c_arr = np.array(curve)
        global_max = max(global_max, float(np.nanmax(c_arr)))
        
        sample_len = len(c_arr)
        boundaries = [0] + [dp for dp in (drift_points or []) if dp < sample_len] + [sample_len]
        boundaries = sorted(list(set(boundaries)))
        
        for seg_idx in range(len(boundaries) - 1):
            start_i = boundaries[seg_idx]
            end_i = boundaries[seg_idx + 1]
            seg = c_arr[start_i:end_i]
            
            # Look at stabilized segment after initial ~100 samples
            warmup = min(100, max(1, len(seg) // 20))
            if len(seg) > warmup:
                valid_seg = seg[warmup:]
                valid_seg = valid_seg[(~np.isnan(valid_seg)) & (valid_seg > 15.0)]
                if len(valid_seg) > 0:
                    segment_mins.append(float(np.min(valid_seg)))
            elif len(seg) > 0:
                valid_seg = seg[(~np.isnan(seg)) & (seg > 15.0)]
                if len(valid_seg) > 0:
                    segment_mins.append(float(np.min(valid_seg)))

    if segment_mins:
        stable_min = min(segment_mins)
        # Tight lower bound: just ~0.8% below the lowest stabilized point, rounded cleanly
        calc_ymin = max(0.0, np.floor((stable_min - 0.8) * 2) / 2.0)
    else:
        calc_ymin = 0.0

    calc_ymax = min(100.0, np.ceil((global_max + 1.2) * 2) / 2.0) if global_max > float('-inf') else 100.0

    if calc_ymin >= calc_ymax:
        calc_ymin = max(0.0, calc_ymax - 15.0)

    final_ymin = ymin_override if ymin_override is not None else calc_ymin
    final_ymax = ymax_override if ymax_override is not None else calc_ymax
    
    return final_ymin, final_ymax


def get_experiment_overall_mean(results, exp_key):
    """
    Computes the average overall accuracy across all concepts and configurations
    for a given experiment key (matching the 4th bar of adaptation speed).
    """
    accs = []
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
        if exp_key in config_data:
            data = config_data[exp_key]
            if 'final_accuracy' in data and data['final_accuracy'] > 0:
                accs.append(data['final_accuracy'])
            elif 'average_across_concepts' in data.get('history', {}).get('detailed_metrics', {}):
                accs.append(data['history']['detailed_metrics']['average_across_concepts'])
    if accs:
        return float(np.mean(accs))
    return None


def plot_global_rolling_accuracy(results, labels, colors, ymin=None, ymax=None, save_path=None):
    exp_rolling = {}
    drift_points = []
    x_axis = None
    
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
            hist = data.get("history", {})
            if 'rolling_accuracy' not in hist or not hist['rolling_accuracy']:
                continue
            if exp_key not in exp_rolling:
                exp_rolling[exp_key] = []
            exp_rolling[exp_key].append(hist['rolling_accuracy'])
            if x_axis is None and 'total_samples_seen' in hist:
                x_axis = hist['total_samples_seen']
            if not drift_points and 'drift_points' in hist:
                drift_points = hist['drift_points']
                
    if not exp_rolling:
        return
        
    exp_keys_sorted = sorted(list(exp_rolling.keys()), key=get_exp_num)
    
    fig, ax = plt.subplots(figsize=(17, 8))
    mean_curves = {}
    sample_len = 0
    
    for exp_key in exp_keys_sorted:
        curves = exp_rolling[exp_key]
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean_curve = np.mean(arr, axis=0)
        mean_curves[exp_key] = mean_curve
        sample_len = max(sample_len, min_len)
        
    x = np.array(x_axis[:sample_len] if x_axis is not None else np.arange(1, sample_len + 1))
    boundaries = [0] + [dp for dp in drift_points if dp < sample_len] + [sample_len]
    boundaries = sorted(list(set(boundaries)))

    for exp_key in exp_keys_sorted:
        mean_curve = mean_curves[exp_key]
        overall_mean = get_experiment_overall_mean(results, exp_key)
        if overall_mean is None:
            overall_mean = mean_curve[-1]
            
        color = colors.get(exp_key, "black")
        label = labels.get(exp_key, exp_key)
        
        for seg_idx in range(len(boundaries) - 1):
            start_i = boundaries[seg_idx]
            end_i = boundaries[seg_idx + 1]
            seg_x = x[start_i:end_i]
            seg_y = mean_curve[start_i:end_i]
            seg_label = f"{label} [{overall_mean:.1f}%]" if seg_idx == 0 else None
            ax.plot(seg_x, seg_y, label=seg_label, color=color, linewidth=2.4)
        
    y_min_zoomed, y_max_zoomed = compute_smart_ylim(results, mean_curves, drift_points, ymin_override=ymin, ymax_override=ymax)

    if drift_points:
        y_range = max(1.0, y_max_zoomed - y_min_zoomed)
        x_offset = max(250, int(sample_len * 0.006))
        for idx, dp in enumerate(drift_points):
            if dp < sample_len:
                ax.axvline(x=dp, color='#666666', linestyle='--', alpha=0.75, linewidth=1.2)
                ax.text(dp + x_offset, y_min_zoomed + y_range * 0.03, f'Drift {idx+1}', rotation=90, color='#555555', fontsize=9, fontweight='bold', va='bottom', ha='left')
            
    ax.set_title("Global Rolling Accuracy (Window = 1000 Images)\n(Averaged across all Configurations)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Total Samples Seen in Stream", fontsize=12, fontweight='bold')
    ax.set_ylabel("Rolling Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_ylim(y_min_zoomed, y_max_zoomed)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9.5, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CCCCCC')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.subplots_adjust(left=0.06, right=0.64, top=0.90, bottom=0.10)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")


def plot_global_cumulative_accuracy(results, labels, colors, ymin=None, ymax=None, save_path=None):
    exp_cumulative = {}
    drift_points = []
    x_axis = None
    
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
            hist = data.get("history", {})
            if 'cumulative_accuracy' not in hist or not hist['cumulative_accuracy']:
                continue
            if exp_key not in exp_cumulative:
                exp_cumulative[exp_key] = []
            exp_cumulative[exp_key].append(hist['cumulative_accuracy'])
            if x_axis is None and 'total_samples_seen' in hist:
                x_axis = hist['total_samples_seen']
            if not drift_points and 'drift_points' in hist:
                drift_points = hist['drift_points']
                
    if not exp_cumulative:
        return
        
    exp_keys_sorted = sorted(list(exp_cumulative.keys()), key=get_exp_num)
    
    fig, ax = plt.subplots(figsize=(17, 8))
    mean_curves = {}
    sample_len = 0
    
    for exp_key in exp_keys_sorted:
        curves = exp_cumulative[exp_key]
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean_curve = np.mean(arr, axis=0)
        mean_curves[exp_key] = mean_curve
        sample_len = max(sample_len, min_len)
        
    x = np.array(x_axis[:sample_len] if x_axis is not None else np.arange(1, sample_len + 1))
    boundaries = [0] + [dp for dp in drift_points if dp < sample_len] + [sample_len]
    boundaries = sorted(list(set(boundaries)))

    for exp_key in exp_keys_sorted:
        mean_curve = mean_curves[exp_key]
        overall_mean = get_experiment_overall_mean(results, exp_key)
        if overall_mean is None:
            overall_mean = mean_curve[-1]
            
        color = colors.get(exp_key, "black")
        label = labels.get(exp_key, exp_key)
        
        # Plot concept segment by segment to eliminate artificial drop lines at drift transitions
        for seg_idx in range(len(boundaries) - 1):
            start_i = boundaries[seg_idx]
            end_i = boundaries[seg_idx + 1]
            seg_x = x[start_i:end_i]
            seg_y = mean_curve[start_i:end_i]
            seg_label = f"{label} [{overall_mean:.1f}%]" if seg_idx == 0 else None
            ax.plot(seg_x, seg_y, label=seg_label, color=color, linewidth=2.4)
        
    y_min_zoomed, y_max_zoomed = compute_smart_ylim(results, mean_curves, drift_points, ymin_override=ymin, ymax_override=ymax)

    if drift_points:
        y_range = max(1.0, y_max_zoomed - y_min_zoomed)
        x_offset = max(250, int(sample_len * 0.006))
        for idx, dp in enumerate(drift_points):
            if dp < sample_len:
                ax.axvline(x=dp, color='#666666', linestyle='--', alpha=0.75, linewidth=1.2)
                ax.text(dp + x_offset, y_min_zoomed + y_range * 0.03, f'Drift {idx+1}', rotation=90, color='#555555', fontsize=9, fontweight='bold', va='bottom', ha='left')
            
    ax.set_title("Global Cumulative Accuracy\n(Averaged across all Configurations)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Total Samples Seen in Stream", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_ylim(y_min_zoomed, y_max_zoomed)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9.5, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CCCCCC')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.subplots_adjust(left=0.06, right=0.64, top=0.90, bottom=0.10)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")


def plot_global_cl_matrices(results, labels, save_path=None):
    exp_matrices = {}
    
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
            cl = data.get("cl_matrix", [])
            if not cl:
                continue
                
            train_concepts = [entry['train_concept'] for entry in cl]
            eval_concepts = []
            for entry in cl:
                for c in entry['evaluations'].keys():
                    if c not in eval_concepts:
                        eval_concepts.append(c)
                        
            matrix = np.zeros((len(train_concepts), len(eval_concepts)))
            for i, entry in enumerate(cl):
                for j, c in enumerate(eval_concepts):
                    matrix[i, j] = entry['evaluations'].get(c, np.nan)
                    
            if exp_key not in exp_matrices:
                exp_matrices[exp_key] = []
            exp_matrices[exp_key].append(matrix)
            
    if not exp_matrices:
        return
        
    exp_keys_sorted = sorted(list(exp_matrices.keys()), key=get_exp_num)
    n_exps = len(exp_keys_sorted)
    
    ncols = 4 if n_exps > 3 else n_exps
    nrows = int(np.ceil(n_exps / ncols))
    
    fig_height = 4.5 * nrows + 1.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, fig_height), squeeze=False)
    
    last_heatmap = None
    for idx, exp_key in enumerate(exp_keys_sorted):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        mats = exp_matrices[exp_key]
        stacked = np.stack(mats, axis=0)
        mean_mat = np.nanmean(stacked, axis=0)
        std_mat = np.nanstd(stacked, axis=0)
        
        T_train, T_eval = mean_mat.shape
        t_labels = [f"C{i}" for i in range(T_train)]
        e_labels = [f"C{j}" for j in range(T_eval)]
        
        annot_array = np.empty((T_train, T_eval), dtype=object)
        for i in range(T_train):
            for j in range(T_eval):
                if np.isnan(mean_mat[i, j]):
                    annot_array[i, j] = ""
                else:
                    annot_array[i, j] = f"{mean_mat[i, j]:.1f}%\n±{std_mat[i, j]:.1f}"
                    
        hm = sns.heatmap(mean_mat, annot=annot_array, fmt="", cmap="Blues",
                         xticklabels=e_labels, yticklabels=t_labels, ax=ax,
                         cbar=False, vmin=0, vmax=100, 
                         annot_kws={'fontsize': 7.0, 'fontweight': 'bold'})
        last_heatmap = hm
        
        ax.set_title(format_exp_title(labels.get(exp_key, exp_key)), fontsize=9.5, fontweight='bold', pad=6)
        
        is_bottom = (row == nrows - 1) or (idx + ncols >= n_exps)
        if is_bottom:
            ax.set_xlabel("Evaluated Concept", fontsize=9, fontweight='bold', labelpad=4)
        else:
            ax.set_xlabel("")
            
        if col == 0:
            ax.set_ylabel("Trained Concept", fontsize=9, fontweight='bold', labelpad=4)
        else:
            ax.set_ylabel("")
            
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
        
    for idx in range(n_exps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
        
    if last_heatmap is not None:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.65])
        cbar = fig.colorbar(last_heatmap.get_children()[0], cax=cbar_ax)
        cbar.set_label('Accuracy (%)', fontsize=11, fontweight='bold', labelpad=8)
        cbar.ax.tick_params(labelsize=9)
        
    fig.suptitle("Global Continual Learning Matrices (Mean ± Std across all Configurations)", fontsize=14.5, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.85, bottom=0.08, hspace=0.48, wspace=0.32)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")


def plot_global_forgetting(results, labels, save_path=None):
    exp_forgetting = {}
    
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
            cl = data.get("cl_matrix", [])
            if not cl or len(cl) < 2:
                continue
                
            train_concepts = [entry['train_concept'] for entry in cl]
            imm = []
            fin = []
            for i, concept in enumerate(train_concepts):
                eval_key = concept.replace('_recurrent', '') if '_recurrent' in concept else concept
                imm.append(cl[i]['evaluations'].get(eval_key, np.nan))
                fin.append(cl[-1]['evaluations'].get(eval_key, np.nan))
                
            if exp_key not in exp_forgetting:
                exp_forgetting[exp_key] = {'imm': [], 'fin': [], 'concepts': train_concepts}
            exp_forgetting[exp_key]['imm'].append(imm)
            exp_forgetting[exp_key]['fin'].append(fin)
            
    if not exp_forgetting:
        return
        
    exp_keys_sorted = sorted(list(exp_forgetting.keys()), key=get_exp_num)
    n_exps = len(exp_keys_sorted)
    
    ncols = 4 if n_exps > 3 else n_exps
    nrows = int(np.ceil(n_exps / ncols))
    
    fig_height = 4.5 * nrows + 1.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, fig_height), squeeze=False)
    
    rect1_sample = None
    rect2_sample = None
    
    for idx, exp_key in enumerate(exp_keys_sorted):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        info = exp_forgetting[exp_key]
        imm_arr = np.array(info['imm'])
        fin_arr = np.array(info['fin'])
        
        imm_mean = np.nanmean(imm_arr, axis=0)
        imm_std = np.nanstd(imm_arr, axis=0)
        fin_mean = np.nanmean(fin_arr, axis=0)
        fin_std = np.nanstd(fin_arr, axis=0)
        
        x = np.arange(len(info['concepts']))
        width = 0.35
        
        r1 = ax.bar(x - width/2, imm_mean, width, yerr=imm_std, capsize=3, label='Right after learning', color='#5DADE2', edgecolor='black', linewidth=0.5)
        r2 = ax.bar(x + width/2, fin_mean, width, yerr=fin_std, capsize=3, label='At end of stream', color='#EC7063', edgecolor='black', linewidth=0.5)
        if rect1_sample is None:
            rect1_sample, rect2_sample = r1, r2
            
        if col == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=9.5, fontweight='bold')
        else:
            ax.set_ylabel('')
            
        ax.set_title(format_exp_title(labels.get(exp_key, exp_key)), fontsize=9.5, fontweight='bold', pad=6)
        ax.set_xticks(x)
        compact_labels = [f"C{i}" for i in range(len(info['concepts']))]
        ax.set_xticklabels(compact_labels, rotation=0, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 118)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        for i, (m, s) in enumerate(zip(imm_mean, imm_std)):
            if not np.isnan(m):
                err = s if not np.isnan(s) else 0
                ax.text(i - width/2, min(106, m + err + 1.5), f'{m:.1f}', ha='center', va='bottom', fontsize=7.0, fontweight='bold', rotation=45)
        for i, (m, s) in enumerate(zip(fin_mean, fin_std)):
            if not np.isnan(m):
                err = s if not np.isnan(s) else 0
                ax.text(i + width/2, min(106, m + err + 1.5), f'{m:.1f}', ha='center', va='bottom', fontsize=7.0, fontweight='bold', rotation=45)
            
    for idx in range(n_exps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
        
    fig.suptitle("Global Forgetting: Immediate vs Final Accuracy (Mean ± Std across all Configurations)", fontsize=14.5, fontweight='bold', y=0.98)
    
    if rect1_sample is not None and rect2_sample is not None:
        fig.legend(handles=[rect1_sample, rect2_sample], labels=['Right after learning', 'At end of stream'],
                   loc='upper center', bbox_to_anchor=(0.5, 0.925), ncol=2, fontsize=10.5, 
                   frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CCCCCC')
        
    fig.subplots_adjust(left=0.06, right=0.96, top=0.76, bottom=0.08, hspace=0.48, wspace=0.28)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plotting suite for FMOW Continual Learning Experiments")
    parser.add_argument('--results', type=str, default="experiment_results.json", help="Path to experiment results JSON file")
    parser.add_argument('--output_dir', type=str, default="./plots", help="Directory to save generated PNG charts")
    parser.add_argument('--show_individual', action='store_true', help="Open interactive GUI windows for each individual config and experiment (warning: generates 100+ windows)")
    parser.add_argument('--save_individual', action='store_true', default=True, help="Save individual config plots to output_dir")
    parser.add_argument('--no_show', action='store_true', help="Skip interactive plt.show() display")
    parser.add_argument('--ymin', type=float, default=None, help="Custom minimum Y-axis limit for accuracy plots (default: auto-zoomed)")
    parser.add_argument('--ymax', type=float, default=None, help="Custom maximum Y-axis limit for accuracy plots (default: auto-zoomed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "global"), exist_ok=True)

    try:
        with open(args.results, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{args.results}'. Make sure the experiment results file exists.")
        return

    colors, labels = get_palette()
    
    is_nested = any(k.startswith('config_') for k in results.keys())
    if not is_nested:
        results = {"config_default": results}

    # 1. PER-CONFIG PROCESSING & OPTIONAL INDIVIDUAL SAVING/SHOWING
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
            
        config_out_dir = os.path.join(args.output_dir, config_key)
        if args.save_individual:
            os.makedirs(config_out_dir, exist_ok=True)
            
        export_detailed_metrics_table(config_key, config_data, labels, output_dir=config_out_dir if args.save_individual else ".")
        
        # Individual figures for this config
        if args.show_individual or args.save_individual:
            # Per-experiment 3-subplot figures
            for exp_key, data in config_data.items():
                if not exp_key.startswith('exp_'): continue
                hist = data.get("history", {})
                acc = data.get("final_accuracy", 0.0)
                cl = data.get("cl_matrix", [])
                color = colors.get(exp_key, "black")
                label = labels.get(exp_key, exp_key)
                
                fig = plt.figure(figsize=(22, 6))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(hist.get('total_samples_seen', []), hist.get('cumulative_accuracy', []),
                         label=f"Cumulative Acc. [{acc:.1f}%]", color=color, linestyle='-', linewidth=2)
                if 'rolling_accuracy' in hist:
                    ax1.plot(hist.get('total_samples_seen', []), hist['rolling_accuracy'],
                             label="Rolling Acc.", color=color, linestyle=':', alpha=0.6, linewidth=1.5)
                if 'drift_points' in hist:
                    for dp in hist['drift_points']:
                        ax1.axvline(x=dp, color='gray', linestyle='--', alpha=0.7)
                ax1.set_title("Streaming Accuracy", fontsize=13, fontweight='bold')
                ax1.set_xlabel("Samples Seen", fontsize=10)
                ax1.set_ylabel("Accuracy (%)", fontsize=10)
                ax1.legend(loc="lower right")
                ax1.grid(True, linestyle='--', alpha=0.6)
                
                ax2 = fig.add_subplot(1, 3, 2)
                if cl:
                    plot_cl_matrix(cl, "Continual Learning Matrix", ax2)
                else:
                    ax2.text(0.5, 0.5, 'No CL Matrix Data', ha='center', va='center')
                    ax2.axis('off')
                    
                ax3 = fig.add_subplot(1, 3, 3)
                if cl and len(cl) > 1:
                    plot_forgetting(cl, "Forgetting (Immediate vs Final)", ax3)
                else:
                    ax3.text(0.5, 0.5, 'No Forgetting Data', ha='center', va='center')
                    ax3.axis('off')
                    
                fig.suptitle(f"{label} ({config_key})", fontsize=14, fontweight='bold', y=0.98)
                fig.subplots_adjust(top=0.88, bottom=0.12, left=0.05, right=0.96, wspace=0.25)
                
                if args.save_individual:
                    fig.savefig(os.path.join(config_out_dir, f"{exp_key}_overview.png"), dpi=200, bbox_inches='tight')
                if not args.show_individual:
                    plt.close(fig)

    # 2. GLOBAL AGGREGATED FIGURES (Primary Focus)
    print("\n=======================================================")
    print(" GENERATING GLOBAL SUMMARY FIGURES ACROSS ALL CONFIGS")
    print("=======================================================")
    
    global_dir = os.path.join(args.output_dir, "global")
    
    # Task 0 & 4: Adaptation Speed (4 Bars)
    plot_adaptation_speed(results, labels, save_path=os.path.join(global_dir, "global_adaptation_speed.png"))
    
    # Task 2: Global Rolling Accuracy (Mean ± Std)
    plot_global_rolling_accuracy(results, labels, colors, ymin=args.ymin, ymax=args.ymax, save_path=os.path.join(global_dir, "global_rolling_accuracy.png"))
    
    # Task 3: Global Cumulative Accuracy (Mean ± Std, Zoomed Y-axis)
    plot_global_cumulative_accuracy(results, labels, colors, ymin=args.ymin, ymax=args.ymax, save_path=os.path.join(global_dir, "global_cumulative_accuracy.png"))
    
    # Task 5: Global Continual Learning Heatmaps (Mean ± Std in each cell)
    plot_global_cl_matrices(results, labels, save_path=os.path.join(global_dir, "global_cl_matrices.png"))
    
    # Global Forgetting Plot (Mean ± Std)
    plot_global_forgetting(results, labels, save_path=os.path.join(global_dir, "global_forgetting.png"))
    
    # Global CSV Summary Export
    export_global_metrics_summary(results, labels, output_dir=global_dir)

    print(f"\nAll global and summary plots successfully exported to '{global_dir}'.")

    if not args.no_show:
        print("Displaying global summary figures...")
        plt.show()


if __name__ == "__main__":
    main()