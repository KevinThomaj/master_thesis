import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_cl_matrix(cl_matrix, title, ax):
    if not cl_matrix:
        return
        
    train_concepts = [entry['train_concept'] for entry in cl_matrix]
    
    # Collect all unique eval concepts to form the columns
    eval_concepts = []
    for entry in cl_matrix:
        for c in entry['evaluations'].keys():
            if c not in eval_concepts:
                eval_concepts.append(c)
                
    # Create matrix
    matrix = np.zeros((len(train_concepts), len(eval_concepts)))
    for i, entry in enumerate(cl_matrix):
        for j, c in enumerate(eval_concepts):
            matrix[i, j] = entry['evaluations'].get(c, np.nan)
            
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=eval_concepts, yticklabels=train_concepts, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Evaluated on Concept")
    ax.set_ylabel("Trained on Concept")


def plot_forgetting(cl_matrix, title, ax):
    if not cl_matrix or len(cl_matrix) < 2:
        return
        
    train_concepts = [entry['train_concept'] for entry in cl_matrix]
    
    immediate_acc = []
    final_acc = []
    
    for i, concept in enumerate(train_concepts):
        # If it's a recurrent concept, we need to map it back to the original test concept key
        eval_key = concept.replace('_recurrent', '') if '_recurrent' in concept else concept

        # Accuracy immediately after learning the concept
        immediate = cl_matrix[i]['evaluations'].get(eval_key, 0)
        immediate_acc.append(immediate)
        
        # Accuracy at the end of the stream
        final = cl_matrix[-1]['evaluations'].get(eval_key, 0)
        final_acc.append(final)
        
    x = np.arange(len(train_concepts))
    width = 0.35
    
    ax.bar(x - width/2, immediate_acc, width, label='Right after learning', color='skyblue')
    ax.bar(x + width/2, final_acc, width, label='At the end of stream', color='lightcoral')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(train_concepts, rotation=45, ha='right')
    ax.legend()
    
    # Add values on top of bars
    for i, v in enumerate(immediate_acc):
        ax.text(i - width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(final_acc):
        ax.text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)


def export_detailed_metrics_table(config_key, config_data, labels):
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
    csv_filename = f"detailed_metrics_{config_key}.csv"
    df.round(2).to_csv(csv_filename, index=False)
    print(f"[{config_key}] Exported detailed per-concept metrics to {csv_filename}")


def export_global_metrics_summary(results, labels):
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
        df_summary = df.groupby(['Experiment', 'Concept']).agg(['mean', 'std']).round(2)
        df_summary.to_csv("detailed_metrics_global_summary.csv")
        print("[Global] Exported global aggregated metrics to detailed_metrics_global_summary.csv")


def plot_adaptation_speed(results, labels):
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
                    'first_window_acc_sum': 0.0,
                    'after_first_window_acc_sum': 0.0,
                    'final_window_acc_sum': 0.0,
                    'total_acc_sum': 0.0,
                    'count': 0
                }
                
            for concept in concept_keys:
                metrics = det_met[concept]
                exp_metrics[exp_key]['first_window_acc_sum'] += metrics.get('first_window_accuracy', 0)
                exp_metrics[exp_key]['after_first_window_acc_sum'] += metrics.get('after_first_window_accuracy', 0)
                exp_metrics[exp_key]['final_window_acc_sum'] += metrics.get('final_window_accuracy', 0)
                exp_metrics[exp_key]['total_acc_sum'] += metrics.get('total_accuracy', 0)
                exp_metrics[exp_key]['count'] += 1

    if not exp_metrics:
        print("No detailed metrics found for adaptation speed plot.")
        return
        
    def get_exp_num(k):
        try:
            return int(k.split('_')[1])
        except:
            return 999
            
    exp_keys_sorted = sorted(list(exp_metrics.keys()), key=get_exp_num)
    
    experiment_names = []
    first_window_means = []
    after_first_window_means = []
    final_window_means = []
    total_acc_means = []
    
    for exp_key in exp_keys_sorted:
        metrics = exp_metrics[exp_key]
        count = metrics['count']
        if count > 0:
            experiment_names.append(labels.get(exp_key, exp_key))
            first_window_means.append(metrics['first_window_acc_sum'] / count)
            after_first_window_means.append(metrics['after_first_window_acc_sum'] / count)
            final_window_means.append(metrics['final_window_acc_sum'] / count)
            total_acc_means.append(metrics['total_acc_sum'] / count)
            
    x = np.arange(len(experiment_names))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 = ax.bar(x - 1.5 * width, first_window_means, width, label='First 500 Images (Start)', color='#4C72B0')
    rects2 = ax.bar(x - 0.5 * width, after_first_window_means, width, label='After First 500 Images (Adaptation)', color='#DD8452')
    rects3 = ax.bar(x + 0.5 * width, final_window_means, width, label='Last 500 Images (End of Concept)', color='#55A868')
    rects4 = ax.bar(x + 1.5 * width, total_acc_means, width, label='Total Concept Accuracy', color='#8172B2')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Adaptation Speed & Concept Progression: Start, Adaptation, End, and Total\n(Mean across all Concepts & Configs)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=25, ha='right', fontsize=8)
    ax.legend(loc='lower right', fontsize=10)
    
    ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=7)
    ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=7)
    ax.bar_label(rects3, padding=3, fmt='%.1f', fontsize=7)
    ax.bar_label(rects4, padding=3, fmt='%.1f', fontsize=7)
    
    fig.tight_layout()


def plot_global_rolling_accuracy(results, labels, colors):
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
        
    def get_exp_num(k):
        try:
            return int(k.split('_')[1])
        except:
            return 999
            
    exp_keys_sorted = sorted(list(exp_rolling.keys()), key=get_exp_num)
    
    plt.figure(figsize=(14, 8))
    y_min_all = float('inf')
    y_max_all = float('-inf')
    
    for exp_key in exp_keys_sorted:
        curves = exp_rolling[exp_key]
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        x = x_axis[:min_len] if x_axis is not None else np.arange(1, min_len + 1)
        
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        color = colors.get(exp_key, "black")
        label = labels.get(exp_key, exp_key)
        
        plt.plot(x, mean_curve, label=f"{label}", color=color, linewidth=2)
        plt.fill_between(x, np.maximum(0, mean_curve - std_curve), np.minimum(100, mean_curve + std_curve),
                         color=color, alpha=0.15)
                         
        y_min_all = min(y_min_all, np.min(mean_curve - std_curve))
        y_max_all = max(y_max_all, np.max(mean_curve + std_curve))
        
    if drift_points:
        for dp in drift_points:
            plt.axvline(x=dp, color='gray', linestyle='--', alpha=0.7)
            
    plt.title("Global Rolling Accuracy (Window = 1000) — Mean ± Std across Configs", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Total Samples Seen", fontsize=12)
    plt.ylabel("Rolling Accuracy (%)", fontsize=12)
    if y_min_all < y_max_all:
        plt.ylim(max(0, y_min_all - 3), min(100, y_max_all + 3))
    plt.legend(loc="lower right", fontsize=9, frameon=True, edgecolor='lightgray')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_global_cumulative_accuracy(results, labels, colors):
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
        
    def get_exp_num(k):
        try:
            return int(k.split('_')[1])
        except:
            return 999
            
    exp_keys_sorted = sorted(list(exp_cumulative.keys()), key=get_exp_num)
    
    plt.figure(figsize=(14, 8))
    y_min_all = float('inf')
    y_max_all = float('-inf')
    
    for exp_key in exp_keys_sorted:
        curves = exp_cumulative[exp_key]
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        x = x_axis[:min_len] if x_axis is not None else np.arange(1, min_len + 1)
        
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        final_mean = mean_curve[-1]
        final_std = std_curve[-1]
        
        color = colors.get(exp_key, "black")
        label = labels.get(exp_key, exp_key)
        
        plt.plot(x, mean_curve, label=f"{label} [{final_mean:.1f} ± {final_std:.1f}%]", color=color, linewidth=2)
        plt.fill_between(x, np.maximum(0, mean_curve - std_curve), np.minimum(100, mean_curve + std_curve),
                         color=color, alpha=0.15)
                         
        y_min_all = min(y_min_all, np.min(mean_curve - std_curve))
        y_max_all = max(y_max_all, np.max(mean_curve + std_curve))
        
    if drift_points:
        for dp in drift_points:
            plt.axvline(x=dp, color='gray', linestyle='--', alpha=0.7)
            
    plt.title("Global Cumulative Accuracy — Mean ± Std across Configs (Zoomed)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Total Samples Seen", fontsize=12)
    plt.ylabel("Cumulative Accuracy (%)", fontsize=12)
    # Dynamic zoom per requirement 3
    if y_min_all < y_max_all:
        plt.ylim(max(0, y_min_all - 2), min(100, y_max_all + 2))
    plt.legend(loc="lower right", fontsize=9, frameon=True, edgecolor='lightgray')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_global_cl_matrices(results, labels):
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
        
    def get_exp_num(k):
        try:
            return int(k.split('_')[1])
        except:
            return 999
            
    exp_keys_sorted = sorted(list(exp_matrices.keys()), key=get_exp_num)
    n_exps = len(exp_keys_sorted)
    
    ncols = min(3, n_exps)
    nrows = int(np.ceil(n_exps / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    
    for idx, exp_key in enumerate(exp_keys_sorted):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        mats = exp_matrices[exp_key]
        stacked = np.stack(mats, axis=0)
        mean_mat = np.nanmean(stacked, axis=0)
        std_mat = np.nanstd(stacked, axis=0)
        
        T_train, T_eval = mean_mat.shape
        t_labels = [f"Concept_{i}" for i in range(T_train)]
        e_labels = [f"Concept_{j}" for j in range(T_eval)]
        
        annot_array = np.empty((T_train, T_eval), dtype=object)
        for i in range(T_train):
            for j in range(T_eval):
                if np.isnan(mean_mat[i, j]):
                    annot_array[i, j] = ""
                else:
                    annot_array[i, j] = f"{mean_mat[i, j]:.1f}\n±{std_mat[i, j]:.1f}"
                    
        sns.heatmap(mean_mat, annot=annot_array, fmt="", cmap="YlGnBu",
                    xticklabels=e_labels, yticklabels=t_labels, ax=ax,
                    cbar=True, vmin=0, vmax=100)
        ax.set_title(labels.get(exp_key, exp_key), fontsize=10, fontweight='bold')
        ax.set_xlabel("Evaluated on Concept", fontsize=9)
        ax.set_ylabel("Trained on Concept", fontsize=9)
        
    for idx in range(n_exps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
        
    fig.suptitle("Global Continual Learning Matrices (Mean ± Std across all Configs)", fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()


def plot_global_forgetting(results, labels):
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
        
    def get_exp_num(k):
        try:
            return int(k.split('_')[1])
        except:
            return 999
            
    exp_keys_sorted = sorted(list(exp_forgetting.keys()), key=get_exp_num)
    n_exps = len(exp_keys_sorted)
    
    ncols = min(3, n_exps)
    nrows = int(np.ceil(n_exps / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    
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
        
        rects1 = ax.bar(x - width/2, imm_mean, width, yerr=imm_std, capsize=4, label='Right after learning', color='skyblue')
        rects2 = ax.bar(x + width/2, fin_mean, width, yerr=fin_std, capsize=4, label='At end of stream', color='lightcoral')
        
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_title(labels.get(exp_key, exp_key), fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(info['concepts'], rotation=30, ha='right', fontsize=8)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 105)
        
        for i, (m, s) in enumerate(zip(imm_mean, imm_std)):
            ax.text(i - width/2, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=7)
        for i, (m, s) in enumerate(zip(fin_mean, fin_std)):
            ax.text(i + width/2, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=7)
            
    for idx in range(n_exps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
        
    fig.suptitle("Global Forgetting: Immediate vs Final Accuracy (Mean ± Std across Configs)", fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()


def main():
    # 1. Load the exported JSON data
    file_path = "experiment_results.json"

    try:
        with open(file_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it is in the same directory as this script.")
        return

    colors = {
        "exp_1": "#E41A1C",
        "exp_2": "#8B0000",
        "exp_3": "#377EB8",
        "exp_4": "#00CED1",
        "exp_5": "#4DAF4A",
        "exp_6": "#32CD32",
        "exp_7": "#FF7F00",
        "exp_8": "#984EA3",
        "exp_9": "#00BFFF",
        "exp_10": "#00008B",
        "exp_11": "#A65628"
    }
    
    labels = {
        "exp_1":"Exp 1: S(hist) + S(inf)",
        "exp_2":"Exp 2: S+P(hist) + S(inf)",
        "exp_3":"Exp 3: S(hist) + S(stream)",
        "exp_4":"Exp 4: S(hist) + S+P(stream)",
        "exp_5":"Exp 5: S+P(hist) + S+P(stream)",
        "exp_6":"Exp 6: S+P(hist) + S(stream, P frozen)",
        "exp_7":"Exp 7: S+P(hist) + S(stream, no P)",
        "exp_8":"Exp 8: S(hist) + EMA Teacher",
        "exp_9":"Exp 9: S+P(hist) + S+P(stream) [Stop at 2k]",
        "exp_10":"Exp 10: S+P(hist) + S+P(stream) [Stop at 4k]",
        "exp_11":"Exp 11: S+P(hist) + S+P(stream) [Stop at 8k]"
    }
    
    is_nested = any(k.startswith('config_') for k in results.keys())
    if not is_nested:
        results = {"config_default": results}

    # Export per-config detailed metrics and individual plots
    for config_key, config_data in results.items():
        if not config_key.startswith('config_'):
            continue
            
        print(f"Plotting results for {config_key}...")
        
        # -------------------------------------------------------------
        # 1) DETAILED FIGURE PER EXPERIMENT
        # -------------------------------------------------------------
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
                
            hist = data.get("history", {})
            acc = data.get("final_accuracy", 0.0)
            cl = data.get("cl_matrix", [])
            color = colors.get(exp_key, "black")
            label = labels.get(exp_key, exp_key)
    
            fig = plt.figure(figsize=(24, 6))
            
            # Left subplot: Accuracies
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(hist.get('total_samples_seen', []), hist.get('cumulative_accuracy', []),
                     label=f"Cumulative Acc. [{acc:.1f}%]", color=color, linestyle='-', linewidth=2)
                     
            if 'rolling_accuracy' in hist:
                ax1.plot(hist.get('total_samples_seen', []), hist['rolling_accuracy'],
                         label="Rolling Acc.", color=color, linestyle=':', alpha=0.6, linewidth=1.5)
    
            if 'drift_points' in hist:
                for drift_pt in hist['drift_points']:
                    ax1.axvline(x=drift_pt, color='gray', linestyle='--', alpha=0.7)
                    bottom_y = ax1.get_ylim()[0]
                    ax1.text(drift_pt + 10, bottom_y + 5, 'Concept Drift', rotation=90, color='gray', fontsize=9)
                    
            ax1.set_title("Streaming Accuracy", fontsize=14, pad=10)
            ax1.set_xlabel("Total Samples Seen", fontsize=11)
            ax1.set_ylabel("Accuracy (%)", fontsize=11)
            ax1.legend(loc="lower right", frameon=True)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Middle subplot: CL Matrix Heatmap
            ax2 = fig.add_subplot(1, 3, 2)
            if cl:
                plot_cl_matrix(cl, "Continual Learning Matrix", ax2)
            else:
                ax2.text(0.5, 0.5, 'No CL Matrix Data', horizontalalignment='center', verticalalignment='center')
                ax2.axis('off')
                
            # Right subplot: Forgetting comparison
            ax3 = fig.add_subplot(1, 3, 3)
            if cl and len(cl) > 1:
                plot_forgetting(cl, "Forgetting (Immediate vs Final)", ax3)
            else:
                ax3.text(0.5, 0.5, 'No Forgetting Data', horizontalalignment='center', verticalalignment='center')
                ax3.axis('off')
                
            fig.suptitle(f"{label} ({config_key})", fontsize=16, fontweight='bold')
            fig.tight_layout()
    
        # -------------------------------------------------------------
        # 2) SUMMARY FIGURE (Cumulative Accuracy per config)
        # -------------------------------------------------------------
        plt.figure(figsize=(12, 7))
        drift_points_plotted = False
    
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
                
            hist = data.get("history", {})
            acc = data.get("final_accuracy", 0.0)
            color = colors.get(exp_key, "black")
            label = labels.get(exp_key, exp_key)
    
            plt.plot(hist.get('total_samples_seen', []), hist.get('cumulative_accuracy', []),
                     label=f"{label} [{acc:.1f}%]", color=color, linestyle='-', linewidth=2)
    
            if not drift_points_plotted and 'drift_points' in hist:
                for drift_pt in hist['drift_points']:
                    plt.axvline(x=drift_pt, color='gray', linestyle='--', alpha=0.7)
                drift_points_plotted = True
    
        plt.title(f"Summary Comparison: Cumulative Accuracy - {config_key}", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Total Samples Seen", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='lightgray')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # -------------------------------------------------------------
        # 3) SUMMARY FIGURE (Rolling Accuracy per config)
        # -------------------------------------------------------------
        has_rolling = any('rolling_accuracy' in data.get('history', {}) for exp_key, data in config_data.items() if exp_key.startswith('exp_'))
        
        if has_rolling:
            plt.figure(figsize=(12, 7))
            drift_points_plotted = False
        
            for exp_key, data in config_data.items():
                if not exp_key.startswith('exp_'):
                    continue
                    
                hist = data.get("history", {})
                if 'rolling_accuracy' not in hist:
                    continue
                    
                color = colors.get(exp_key, "black")
                label = labels.get(exp_key, exp_key)
        
                plt.plot(hist.get('total_samples_seen', []), hist['rolling_accuracy'],
                         label=f"{label}", color=color, linestyle='-', linewidth=2)
        
                if not drift_points_plotted and 'drift_points' in hist:
                    for drift_pt in hist['drift_points']:
                        plt.axvline(x=drift_pt, color='gray', linestyle='--', alpha=0.7)
                    drift_points_plotted = True
        
            plt.title(f"Summary Comparison: Rolling Accuracy - {config_key}", fontsize=16, fontweight='bold', pad=15)
            plt.xlabel("Total Samples Seen", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='lightgray')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

        # Export detailed metrics table for this config
        export_detailed_metrics_table(config_key, config_data, labels)

    # -------------------------------------------------------------
    # 4) GLOBAL AGGREGATED FIGURES ACROSS ALL CONFIGS
    # -------------------------------------------------------------
    print("\nGenerating Global Aggregated Figures across all Configurations...")
    
    # Task 0 & 4: Adaptation speed with 4 bars
    plot_adaptation_speed(results, labels)
    
    # Task 2: Global Rolling Accuracy (Mean ± Std)
    plot_global_rolling_accuracy(results, labels, colors)
    
    # Task 3: Global Cumulative Accuracy (Mean ± Std, Zoomed Y-axis)
    plot_global_cumulative_accuracy(results, labels, colors)
    
    # Task 5: Global CL Matrices Heatmaps (Mean ± Std in each cell)
    plot_global_cl_matrices(results, labels)
    
    # Global Forgetting Plot (Mean ± Std)
    plot_global_forgetting(results, labels)
    
    # Global CSV Summary Export
    export_global_metrics_summary(results, labels)

    # Display all figures
    plt.show()


if __name__ == "__main__":
    main()