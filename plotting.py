import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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

def plot_detailed_metrics_bar(config_key, config_data, labels, colors):
    exp_metrics = {}
    for exp_key, data in config_data.items():
        if not exp_key.startswith('exp_'): continue
        det_met = data['history'].get('detailed_metrics', {})
        
        concept_keys = [k for k in det_met.keys() if k != 'average_across_concepts']
        if not concept_keys: continue
        
        avg_first = np.mean([det_met[c].get('first_window_accuracy', 0) for c in concept_keys])
        avg_final = np.mean([det_met[c].get('final_window_accuracy', 0) for c in concept_keys])
        avg_after = np.mean([det_met[c].get('after_first_window_accuracy', 0) for c in concept_keys])
        avg_total = det_met.get('average_across_concepts', 0)
        
        exp_metrics[exp_key] = {
            'First Window': avg_first,
            'Final Window': avg_final,
            'After First Window': avg_after,
            'Total': avg_total
        }
        
    if not exp_metrics:
        return

    metrics_names = ['First Window', 'Final Window', 'After First Window', 'Total']
    exp_keys = list(exp_metrics.keys())
    
    x = np.arange(len(metrics_names))
    width = 0.8 / len(exp_keys)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, exp_key in enumerate(exp_keys):
        vals = [exp_metrics[exp_key][m] for m in metrics_names]
        color = colors.get(exp_key, 'black')
        label = labels.get(exp_key, exp_key)
        
        # Offset bars
        offset = (i - len(exp_keys) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.8)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title(f'Detailed Metrics Comparison - {config_key}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()


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
        "exp_9": "brown",
        "exp_10": "magenta"
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
        "exp_9": "Exp 9: S+P(hist) + Distillation Stream",
        "exp_10": "Exp 10: S(hist) + S(stream)"
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
    
            fig = plt.figure(figsize=(16, 6))
            
            # Left subplot: Accuracies
            ax1 = fig.add_subplot(1, 2, 1)
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
            
            # Right subplot: CL Matrix Heatmap
            ax2 = fig.add_subplot(1, 2, 2)
            if cl:
                plot_cl_matrix(cl, "Continual Learning Matrix", ax2)
            else:
                ax2.text(0.5, 0.5, 'No CL Matrix Data', horizontalalignment='center', verticalalignment='center')
                ax2.axis('off')
                
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
        # 3) DETAILED METRICS BAR CHART
        # -------------------------------------------------------------
        plot_detailed_metrics_bar(config_key, config_data, labels, colors)

    # Display all figures
    plt.show()


if __name__ == "__main__":
    main()