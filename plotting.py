import json
import matplotlib.pyplot as plt


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

    hist_ft = results["fine_tuning"]["history"]
    acc_ft = results["fine_tuning"]["final_accuracy"]

    # 3. Setup the plot
    plt.figure(figsize=(12, 7))

    # Plot Pure Inference
    plt.plot(hist_inf['total_samples_seen'], hist_inf['cumulative_accuracy'],
             label=f"Pure Inference [{acc_inf:.1f}%]", color='red', linestyle='--', linewidth=2)

    # Plot Online Fine-Tuning
    plt.plot(hist_ft['total_samples_seen'], hist_ft['cumulative_accuracy'],
             label=f"Online Fine-Tuning [{acc_ft:.1f}%]", color='blue', linewidth=2.5)

    # Add Concept Drift markers (Using drift points from the inference history)
    for drift_pt in hist_inf['drift_points']:
        plt.axvline(x=drift_pt, color='gray', linestyle=':', alpha=0.7)
        # Position the text slightly to the right of the line, near the bottom
        bottom_y = plt.ylim()[0]
        plt.text(drift_pt + 10, bottom_y + 5, 'Concept Drift', rotation=90, color='gray', fontsize=9)

    # 4. Formatting
    plt.title("Prequential Evaluation on Data Stream (2016-2017)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Total Samples Seen", fontsize=12)
    plt.ylabel("Cumulative Accuracy (%)", fontsize=12)

    # Optional: lock the Y-axis to 0-100 if you want a fixed scale
    # plt.ylim(0, 100)

    plt.legend(loc="lower right", fontsize=11, frameon=True, edgecolor='lightgray')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. Display the plot
    plt.show()


if __name__ == "__main__":
    main()