import gc
import math

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from FeatureDistillation import LinearReluDistiller, MLPDistiller, LinearBNLogSumDistiller
from LinearProbe import LinearProbe
from Student import Student
import traceback
from sklearn.model_selection import ParameterGrid


class GridSearch:

    def __init__(self):
        # Shared base hyperparameters
        self.base_params = {
            'lr': [1e-3],
            'batch_size': [25],
            'epochs_per_batch': [1]
        }

        # 1. Online Without Distillation
        self.grid_no_distill = list(ParameterGrid({**self.base_params}))

        # 2. Online with Linear Distillation
        self.grid_linear_distill = list(ParameterGrid({
            **self.base_params,
            'distill_weight': [1]
        }))

        # 2. Online with LogSum Distillation
        self.grid_logsum_distill = list(ParameterGrid({
            **self.base_params,
            'distill_weight': [0.1]
        }))

        # 3. Online with MLP Distillation
        self.grid_mlp_distill = list(ParameterGrid({
            **self.base_params,
            'distill_weight': [1],
            'hidden_dim': [256]
        }))

        # 4. Online with Linear Distillation + EMA Teacher
        self.grid_linear_ema = list(ParameterGrid({
            **self.base_params,
            'distill_weight': [0.5, 1.0],
            'ema_alpha': [0.99, 0.999]  # Common EMA momentum values
        }))

        # 5. Online with MLP Distillation + EMA Teacher
        self.grid_mlp_ema = list(ParameterGrid({
            **self.base_params,
            'distill_weight': [0.5, 1.0],
            'hidden_dim': [256, 512],
            'ema_alpha': [0.99, 0.999]
        }))

        # 6. Linear Probing (Upper Bound)
        self.grid_linear_probing = list(ParameterGrid({**self.base_params}))

    # ==========================================
    # GRID SEARCH EXECUTION LOOP
    # ==========================================
    def run_grid_search(self,df_stream, embedding_dim_base, pipeline, transform_imagenet, transform_fmow, get_input):
        total_classes = len(df_stream['category'].unique())

        # We define new names so the logic easily knows if it's pretrained or not, and what distillation to use.
        experiments = [
            ("0b_NoPretrain_LogSumDistillation", self.grid_logsum_distill),
            ("0a_NoPretrain_NoDistillation", self.grid_no_distill),
            ("0c_NoPretrain_LinearDistillation", self.grid_linear_distill),
            ("0d_NoPretrain_MLPDistillation", self.grid_mlp_distill),

            ("2_Pretrained_LogSumDistillation", self.grid_logsum_distill),
            ("1_Pretrained_NoDistillation", self.grid_no_distill),
            ("3_Pretrained_LinearDistillation", self.grid_linear_distill),
            ("4_Pretrained_MLPDistillation", self.grid_mlp_distill),

            ("6_Linear_Probing_UB", self.grid_linear_probing)
        ]

        master_results_list = []

        for exp_name, param_grid in experiments:
            print(f"\n{'=' * 60}\nStarting Experiment Suite: {exp_name}\n{'=' * 60}")

            suite_results = []

            for idx, params in enumerate(param_grid):
                print(f"\n--- Running Config {idx + 1}/{len(param_grid)}: {params} ---")

                # 1. Initialize Models & Assign Transforms Dynamically
                if "Linear_Probing" in exp_name:
                    torch.manual_seed(42)
                    model = LinearProbe(num_classes=total_classes, input_dim=embedding_dim_base)
                    current_transform = transform_imagenet
                elif "NoPretrain" in exp_name:
                    torch.manual_seed(42)
                    model = Student(numberOfClasses=total_classes, pretrained=False)
                    current_transform = transform_fmow
                else:
                    torch.manual_seed(42)
                    model = Student(numberOfClasses=total_classes, pretrained=True)
                    current_transform = transform_imagenet

                criterion = nn.CrossEntropyLoss()
                distillator = None

                # 2. Setup Distillators conditionally based on exp_name
                if "LinearDistillation" in exp_name:
                    torch.manual_seed(42)
                    distillator = LinearReluDistiller(dimFeatureStudent=512, dimFeatureTeacher=embedding_dim_base)
                elif "MLPDistillation" in exp_name:
                    torch.manual_seed(42)
                    distillator = MLPDistiller(
                        dimFeatureStudent=512,
                        dimFeatureTeacher=embedding_dim_base,
                        hiddenLayerSize=params.get('hidden_dim')
                    )
                elif "LogSumDistillation" in exp_name:
                    torch.manual_seed(42)
                    distillator = LinearBNLogSumDistiller(dimFeatureStudent=512, dimFeatureTeacher=embedding_dim_base)


                if distillator is not None:
                    optimizer = torch.optim.Adam([
                        {'params': model.parameters(), 'lr': params['lr']},
                        {'params': distillator.parameters(), 'lr': 1e-2}
                    ])
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])


                # 4. Execute Streaming Pipeline
                try:
                    if "Linear_Probing" in exp_name:
                        final_acc, history = pipeline.linear_probing_online(
                            classifier=model,
                            df=df_stream.copy(),
                            criterion=criterion,
                            optimizer=optimizer,
                            batch_size=params['batch_size'],
                            num_epochs_per_batch=params['epochs_per_batch']
                        )
                    else:
                        final_acc, history = pipeline.train_online(
                            df=df_stream.copy(),
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            get_input_fn=get_input,
                            transform_fn=current_transform,
                            target_col='category',
                            batch_size=params['batch_size'],
                            num_epochs_per_batch=params['epochs_per_batch'],
                            distillator=distillator,
                            distill_weight=params.get('distill_weight')
                        )

                    # Append to the suite-specific results
                    suite_results.append({
                        'experiment_group': exp_name,
                        'config_id': f"{exp_name}_cfg{idx}",
                        'params': params,
                        'final_accuracy': final_acc,
                        'history': history
                    })

                except Exception as e:
                    print(f"[!] Failed on config {params}")
                    traceback.print_exc()

                #5. Strict Memory Cleanup between runs
                del model, optimizer, criterion, distillator
                gc.collect()

            # ==========================================
            # PLOT AFTER EACH RESULT
            # ==========================================
            print(f"\nPlotting results for {exp_name}...")
            self.plot_performance_over_time(suite_results)
            master_results_list.extend(suite_results)
            # Purge the history data from RAM before starting the next experiment suite
            del suite_results
            gc.collect()

        print("\nAll experiments completed successfully!")
        print("\nGenerating Distillation Comparisons...")
        self.plot_distillation_comparisons(master_results_list)

    def plot_performance_over_time(self, all_results):
        """
        Plots strictly the cumulative accuracy over time, grouped by experiment.
        Includes smart legends, distinct colors, and end-point highlights.
        """
        if not all_results:
            print("No results to plot.")
            return

        # Extract unique experiment groups
        experiment_groups = list(set(r['experiment_group'] for r in all_results))
        experiment_groups.sort()

        # Calculate grid dimensions for subplots
        cols = min(2, len(experiment_groups))
        rows = math.ceil(len(experiment_groups) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows), squeeze=False)
        fig.suptitle("Cumulative Accuracy Over Time by Experiment", fontsize=20, fontweight='bold', y=1.02)
        axes = axes.flatten()

        # Define colors and line styles for better distinction
        colors = plt.cm.tab10.colors
        line_styles = ['-', '--', '-.', ':']

        for i, group_name in enumerate(experiment_groups):
            ax = axes[i]
            group_results = [r for r in all_results if r['experiment_group'] == group_name]

            # --- SMART LEGEND LOGIC ---
            # Isolate parameters that vary vs. remain constant in this specific group
            all_params = [r['params'] for r in group_results]
            param_keys = all_params[0].keys() if all_params else []

            varied_params = []
            constant_params = {}

            for k in param_keys:
                # Check if the parameter has more than one unique value in this group
                unique_values = set(str(p[k]) for p in all_params)
                if len(unique_values) > 1:
                    varied_params.append(k)
                else:
                    constant_params[k] = all_params[0][k]

            # Display constant parameters as a subtitle
            const_str = " | ".join([f"{k}: {v}" for k, v in constant_params.items()])
            ax.set_title(group_name.replace("_", " "), fontsize=14, fontweight='bold', pad=20)
            if const_str:
                ax.text(0.5, 1.02, f"Fixed: {const_str}", transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=10, color='dimgray')

            # Plot each configuration
            for j, res in enumerate(group_results):
                total_samples = res['history']['total_samples_seen']
                accuracy = res['history']['cumulative_accuracy']
                final_acc = res.get('final_accuracy', accuracy[-1] if accuracy else 0)

                # Create a concise label using ONLY the varied parameters
                if varied_params:
                    label_parts = [f"{k}={res['params'][k]}" for k in varied_params]
                    label = ", ".join(label_parts)
                else:
                    label = "Base Config"

                # Append the final accuracy to the legend for quick comparison
                label += f" → {final_acc:.1f}%"

                # Cycle through colors and styles
                color = colors[j % len(colors)]
                linestyle = line_styles[(j // len(colors)) % len(line_styles)]

                # Plot the main curve
                ax.plot(total_samples, accuracy, label=label, color=color,
                        linestyle=linestyle, alpha=0.85, linewidth=2.5)

                # Plot a marker at the final point to anchor the eye
                if len(total_samples) > 0:
                    ax.plot(total_samples[-1], accuracy[-1], marker='o', color=color, markersize=6)

            # Format the subplot axes
            ax.set_xlabel("Total Samples Seen", fontsize=12, fontweight='medium')
            ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12, fontweight='medium')

            #Optional: Lock Y-axis if you want consistent scaling across all plots
            ax.set_ylim(0, 60)

            ax.grid(True, linestyle='--', alpha=0.6)

            # Place legend outside the plot, styled nicely
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                      frameon=True, edgecolor='lightgray')

        # Hide unused subplots if the grid isn't perfectly filled
        for j in range(len(experiment_groups), len(axes)):
            axes[j].axis('off')

        # tight_layout handles spacing, but sometimes needs adjusting with suptitle
        plt.tight_layout()
        plt.show()

    def plot_distillation_comparisons(self, all_results):
        """
        Plots comparisons of Distillation methods (None vs Linear vs MLP)
        for the SAME base hyperparameters (lr, batch_size, epochs_per_batch).
        """
        if not all_results:
            print("No results to plot.")
            return

        # 1. Define the parameters that make up the "Base Config"
        base_keys = ['lr', 'batch_size', 'epochs_per_batch']

        # 2. Group results by their base configurations
        from collections import defaultdict
        grouped_results = defaultdict(list)

        for res in all_results:
            base_vals = tuple(res['params'].get(k) for k in base_keys)
            grouped_results[base_vals].append(res)

        # 3. Create a comparison figure for each base configuration
        for base_vals, results_for_base in grouped_results.items():
            base_config_str = ", ".join([f"{k}={v}" for k, v in zip(base_keys, base_vals)])

            # Create a 1x3 grid for No Pretrain, Pretrained, and Pretrained+EMA
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            fig.suptitle(f"Distillation Comparison | Base Config: {base_config_str}",
                         fontsize=16, fontweight='bold', y=1.05)

            # Define which prefixes go to which subplot
            categories = {
                "1. Without Pretraining": ["0a", "0b", "0c"],
                "2. With Pretraining": ["1", "2", "3"]
                #"3. Pretrained + EMA Teacher": ["4", "5"]
            }


            for ax, (cat_name, prefixes) in zip(axes, categories.items()):
                ax.set_title(cat_name, fontsize=14, pad=10)

                # Filter results belonging to this specific subplot
                cat_results = [r for r in results_for_base if
                               any(r['experiment_group'].startswith(p) for p in prefixes)]

                if not cat_results:
                    ax.text(0.5, 0.5, "No data for this config", ha='center', va='center',
                            transform=ax.transAxes, color='gray')
                    continue

                for res in cat_results:
                    total_samples = res['history']['total_samples_seen']
                    accuracy = res['history']['cumulative_accuracy']

                    # Clean up the label name (e.g., extracting "NoDistillation")
                    exp_name = res['experiment_group']
                    distill_type = exp_name.split('_', 2)[-1].replace("Distillation", " Distill")

                    # Find distillation-specific hyperparams (like distill_weight, hidden_dim)
                    extra_params = {k: v for k, v in res['params'].items() if k not in base_keys}

                    # Build the legend label
                    if extra_params:
                        extra_str = ", ".join([f"{k}={v}" for k, v in extra_params.items()])
                        label = f"{distill_type} ({extra_str})"
                    else:
                        label = f"{distill_type}"

                    # Add final accuracy
                    final_acc = res.get('final_accuracy', accuracy[-1] if accuracy else 0)
                    label += f" [{final_acc:.1f}%]"

                    ax.plot(total_samples, accuracy, label=label, linewidth=2, alpha=0.85)

                # Formatting
                ax.set_xlabel("Total Samples Seen", fontsize=12)
                if ax == axes[0]:
                    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)

                # Move legend inside the plot but keep it neat
                ax.legend(fontsize=9, loc='upper right', frameon=True, edgecolor='lightgray')

            plt.tight_layout()
            plt.show()