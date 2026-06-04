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


def main():
    # 1. Load the exported JSON data
    file_path = "experiment_results.json"

    try:
        with open(file_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it is in the same directory as this script.")
        return

    # 2. Extract histories and final accuracies
    hist_inf = results["inference"]["history"]
    acc_inf = results["inference"]["final_accuracy"]
    cl_inf = results["inference"].get("cl_matrix", [])

    hist_ft = results["fine_tuning"]["history"]
    acc_ft = results["fine_tuning"]["final_accuracy"]
    cl_ft = results["fine_tuning"].get("cl_matrix", [])

    hist_dist = results["distillation"]["history"]
    acc_dist = results["distillation"]["final_accuracy"]
    cl_dist = results["distillation"].get("cl_matrix", [])

    # 3. Setup the plot for Streaming Accuracies
    plt.figure(figsize=(12, 7))

    # Plot Pure Inference
    plt.plot(hist_inf['total_samples_seen'], hist_inf['cumulative_accuracy'],
             label=f"Pure Inference (Cum.) [{acc_inf:.1f}%]", color='red', linestyle='-', linewidth=2)
    if 'rolling_accuracy' in hist_inf:
        plt.plot(hist_inf['total_samples_seen'], hist_inf['rolling_accuracy'],
                 label=f"Pure Inference (Roll.)", color='red', linestyle=':', alpha=0.6, linewidth=1.5)

    # Plot Online Fine-Tuning
    plt.plot(hist_ft['total_samples_seen'], hist_ft['cumulative_accuracy'],
             label=f"Online Fine-Tuning (Cum.) [{acc_ft:.1f}%]", color='blue', linewidth=2.5)
    if 'rolling_accuracy' in hist_ft:
        plt.plot(hist_ft['total_samples_seen'], hist_ft['rolling_accuracy'],
                 label=f"Online Fine-Tuning (Roll.)", color='blue', linestyle=':', alpha=0.6, linewidth=1.5)

    # Plot Distillation
    plt.plot(hist_dist['total_samples_seen'], hist_dist['cumulative_accuracy'],
             label=f"Fine-Tuning + Distillation (Cum.) [{acc_dist:.1f}%]", color='green', linewidth=2.5)
    if 'rolling_accuracy' in hist_dist:
        plt.plot(hist_dist['total_samples_seen'], hist_dist['rolling_accuracy'],
                 label=f"Fine-Tuning + Distillation (Roll.)", color='green', linestyle=':', alpha=0.6, linewidth=1.5)

    # Add Concept Drift markers (Using drift points from the inference history)
    for drift_pt in hist_inf['drift_points']:
        plt.axvline(x=drift_pt, color='gray', linestyle='--', alpha=0.7)
        # Position the text slightly to the right of the line, near the bottom
        bottom_y = plt.ylim()[0]
        plt.text(drift_pt + 10, bottom_y + 5, 'Concept Drift', rotation=90, color='gray', fontsize=9)

    # 4. Formatting
    plt.title("Prequential Evaluation on Data Stream (2016-2017)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Total Samples Seen", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True, edgecolor='lightgray')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. Continual Learning Matrix Heatmaps
    if cl_inf or cl_ft or cl_dist:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_cl_matrix(cl_inf, "CL Matrix: Pure Inference", axes[0])
        plot_cl_matrix(cl_ft, "CL Matrix: Online Fine-Tuning", axes[1])
        plot_cl_matrix(cl_dist, "CL Matrix: Fine-Tuning + Distillation", axes[2])
        plt.tight_layout()

    # 6. Display the plot
    plt.show()


if __name__ == "__main__":
    main()