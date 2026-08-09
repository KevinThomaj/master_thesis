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
        det_met = data['history'].get('detailed_metrics', {})
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
    print(f"\n[{config_key}] Exported detailed per-concept metrics to {csv_filename}")


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
                    'count': 0
                }
                
            for concept in concept_keys:
                metrics = det_met[concept]
                exp_metrics[exp_key]['first_window_acc_sum'] += metrics.get('first_window_accuracy', 0)
                exp_metrics[exp_key]['after_first_window_acc_sum'] += metrics.get('after_first_window_accuracy', 0)
                exp_metrics[exp_key]['final_window_acc_sum'] += metrics.get('final_window_accuracy', 0)
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
    
    for exp_key in exp_keys_sorted:
        metrics = exp_metrics[exp_key]
        count = metrics['count']
        if count > 0:
            experiment_names.append(labels.get(exp_key, exp_key))
            first_window_means.append(metrics['first_window_acc_sum'] / count)
            after_first_window_means.append(metrics['after_first_window_acc_sum'] / count)
            final_window_means.append(metrics['final_window_acc_sum'] / count)
            
    x = np.arange(len(experiment_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, first_window_means, width, label='First 500 Images', color='skyblue')
    rects2 = ax.bar(x, after_first_window_means, width, label='After First 500 Images', color='lightcoral')
    rects3 = ax.bar(x + width, final_window_means, width, label='Last 500 Images', color='lightgreen')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Adaptation Speed: First, After First, and Last 500 Images\n(Mean across all Concepts & Configs)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right')
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    ax.bar_label(rects3, padding=3, fmt='%.1f')
    
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
        "exp_1": "red",
        "exp_2": "darkred",
        "exp_3": "blue",
        "exp_4": "cyan",
        "exp_5": "green",
        "exp_6": "lime",
        "exp_7": "orange",
        "exp_8": "purple",
        "exp_9": "cyan",
        "exp_10": "blue",
        "exp_11": "brown"
    }
    
    labels = {
        "exp_1": "Exp 1: S(hist) + S(inf)",
        "exp_2": "Exp 2: S+P(hist) + S(inf)",
        "exp_3": "Exp 3: S(hist) + S(stream)",
        "exp_4": "Exp 4: S(hist) + S+P(stream)",
        "exp_5": "Exp 5: S+P(hist) + S+P(stream)",
        "exp_6": "Exp 6: S+P(hist) + S(stream, P frozen)",
        "exp_7": "Exp 7: S+P(hist) + S(stream, no P)",
        "exp_8": "Exp 8: S(hist) + EMA Teacher",
        "exp_9": "Exp 9: S+P(hist) + S+P(stream) [Stop at 1k]",
        "exp_10": "Exp 10: S+P(hist) + S+P(stream) [Stop at 2k]",
        "exp_11": "Exp 11: S+P(hist) + S+P(stream) [Stop at 4k]"
    }
    
    # Check if results has the new nested config structure or the old flat structure
    is_nested = any(k.startswith('config_') for k in results.keys())
    if not is_nested:
        # Wrap it to make it compatible
        results = {"config_default": results}

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
                
            hist = data["history"]
            acc = data["final_accuracy"]
            cl = data.get("cl_matrix", [])
            color = colors.get(exp_key, "black")
            label = labels.get(exp_key, exp_key)
    
            fig = plt.figure(figsize=(24, 6))
            
            # Left subplot: Accuracies
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(hist['total_samples_seen'], hist['cumulative_accuracy'],
                     label=f"Cumulative Acc. [{acc:.1f}%]", color=color, linestyle='-', linewidth=2)
                     
            if 'rolling_accuracy' in hist:
                ax1.plot(hist['total_samples_seen'], hist['rolling_accuracy'],
                         label="Rolling Acc.", color=color, linestyle=':', alpha=0.6, linewidth=1.5)
    
            if 'drift_points' in hist:
                for drift_pt in hist['drift_points']:
                    ax1.axvline(x=drift_pt, color='gray', linestyle='--', alpha=0.7)
                    bottom_y = ax1.get_ylim()[0]
                    ax1.text(drift_pt + 10, bottom_y + 5, 'Concept Drift', rotation=90, color='gray', fontsize=9)
                    
            ax1.set_title(f"Streaming Accuracy", fontsize=14, pad=10)
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
        # 2) SUMMARY FIGURE (Cumulative Accuracy for comparison)
        # -------------------------------------------------------------
        plt.figure(figsize=(12, 7))
        drift_points_plotted = False
    
        for exp_key, data in config_data.items():
            if not exp_key.startswith('exp_'):
                continue
                
            hist = data["history"]
            acc = data["final_accuracy"]
            color = colors.get(exp_key, "black")
            label = labels.get(exp_key, exp_key)
    
            plt.plot(hist['total_samples_seen'], hist['cumulative_accuracy'],
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
        # 3) SUMMARY FIGURE (Rolling Accuracy for comparison)
        # -------------------------------------------------------------
        has_rolling = any('rolling_accuracy' in data.get('history', {}) for exp_key, data in config_data.items() if exp_key.startswith('exp_'))
        
        if has_rolling:
            plt.figure(figsize=(12, 7))
            drift_points_plotted = False
        
            for exp_key, data in config_data.items():
                if not exp_key.startswith('exp_'):
                    continue
                    
                hist = data["history"]
                if 'rolling_accuracy' not in hist:
                    continue
                    
                color = colors.get(exp_key, "black")
                label = labels.get(exp_key, exp_key)
        
                plt.plot(hist['total_samples_seen'], hist['rolling_accuracy'],
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

        # -------------------------------------------------------------
        # 4) EXPORT DETAILED METRICS TABLE
        # -------------------------------------------------------------
        export_detailed_metrics_table(config_key, config_data, labels)

    # -------------------------------------------------------------
    # 5) ADAPTATION SPEED FIGURE (Mean across all configs)
    # -------------------------------------------------------------
    plot_adaptation_speed(results, labels)

    # Display all figures
    plt.show()


if __name__ == "__main__":
    main()