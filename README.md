# Continual Learning & Knowledge Distillation for Earth Observation

This repository contains the experimental framework for studying whether **feature distillation from a ground-based Foundation Model can accelerate on-orbit concept adaptation and mitigate catastrophic forgetting** in lightweight satellite models.

The benchmark is built on the **WILDS-FMoW (Functional Map of the World)** dataset, capturing real-world geographic, seasonal, and semantic distribution shifts over time.

---

## 📌 Project Overview & Requirements

* **Dataset:** [WILDS-FMoW](https://wilds.stanford.edu/datasets/#fmow) (~500K RGB satellite images, 224×224 resolution, 62 original categories).
* **Temporal Drift & Data Split:**
  * **Historical Phase (Pre-training):** 2002 – 2013.
  * **Temporal Gap:** 2014 – 2015. Excluded from training to enforce natural geographic and semantic drift (e.g., massive expansion of agricultural land, changes in satellite sensor resolutions, and regional demographic shifts in Asia and Africa).
  * **Online Streaming Phase:** 2016 – 2017.
* **Class Selection & Balancing:**
  * **25 stable classes** identified from the most populated categories in the historical phase and maintained throughout streaming.
  * **Historical data:** 2,200 images per class (55,000 total historical samples).
  * **Streaming data:** 2,300 images per class (57,500 total streaming samples).
* **Streaming Protocol (Class-Incremental Settings):**
  * The 25 classes are partitioned into **5 sequential concepts** (5 classes per concept).
  * Evaluated across **30 independent deterministic concept configurations** (permutations) for robust statistical confidence intervals.
  * For each concept, **100 images** are held out as a test set to evaluate backward and forward transfer in the **Continual Learning Matrix**.

---

## 🛰️ Architecture & Distillation

### 1. Ground Foundation Model (Extended Pre-training)
* **Backbone:** Meta's **DINOv2** Vision Transformer (`vit_base_patch14_dinov2.lvd142m`, 768-dim embeddings).
* **Training Data:** All images from 2002–2013 (~140K images across all 62 classes, completely unsupervised).
* **Adaptation:** Self-Supervised Learning using the original **DINO loss** with multi-crop augmentations (2 global crops + 6 local views). The transformer backbone is frozen, and **Low-Rank Adaptation (LoRA)** is applied to the Query, Key, Value (`qkv`), and Output Projection (`proj`) attention matrices.

### 2. Satellite Student Model (On-Orbit Model)
* **Backbone:** Lightweight **ResNet-18** (~11M parameters) or **ViT-Tiny** (`vit_tiny_patch16_224`, ~5.7M parameters).
* **Historical Warm-up:** Pre-trained on the historical 25 classes (2002–2013) either with supervised cross-entropy or jointly with feature distillation.

### 3. Feature Distillation Projectors
Aligns the student's penultimate feature representation with the Foundation Model's embedding space:
* **Linear Projector:** Single linear layer mapping student dimensions to 768 with Mean Squared Error (MSE) loss.
* **MLP Projector:** 2-layer MLP (Linear → ReLU → Linear, hidden dimension 1024) with MSE loss.
* **Relational Knowledge Distillation (RKD):** Parameter-free distillation using Huber loss on normalized pairwise Euclidean distance matrices.

---

## 🧪 Experiments Matrix

| Exp ID | Pre-training Setup | Streaming Setup | Objective |
| :---: | :--- | :--- | :--- |
| **Exp 1** | Student (supervised) | Pure inference (no adaptation) | Baseline zero-shot performance and drift decay |
| **Exp 2** | Student + Projector (distilled) | Pure inference (no adaptation) | Evaluates if distillation warm-up enhances robustness |
| **Exp 3** | Student (supervised) | Student fine-tuning | Standard online supervised fine-tuning |
| **Exp 4** | Student (supervised) | (Student + Projector) fine-tuning | Online streaming distillation with randomly initialized projector |
| **Exp 5** | Student + Projector (distilled) | (Student + Projector) fine-tuning | Full continuous feature distillation on stream |
| **Exp 6** | Student + Projector (distilled) | Student fine-tuning, Projector frozen | Streaming distillation with fixed representation mapping |
| **Exp 7** | Student + Projector (distilled) | Student fine-tuning, No projector | Supervised online fine-tuning from distilled initialization |
| **Exp 8** | Student + Projector (distilled) | Distillation stops after 1,000 images/concept | Resource budget analysis |
| **Exp 9** | Student + Projector (distilled) | Distillation stops after 2,000 images/concept | Resource budget analysis |
| **Exp 10** | Student + Projector (distilled) | Distillation stops after 4,000 images/concept | Resource budget analysis |
| **Exp 11** | Student + Projector (distilled) | Distillation stops after 8,000 images/concept | Resource budget analysis |
| **Exp 12** | Frozen Foundation Model | Online Linear Probing on stream | Upper-bound feature representation benchmark |

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/master_thesis.git
   cd master_thesis
   ```

2. **Set up a Python virtual environment (Python 3.10+ recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run Experiments

The main pipeline downloads WILDS-FMoW data automatically into `./data`, manages checkpointing, runs offline feature extraction, and executes streaming experiments.

### Basic Run (All Experiments on 30 Configurations)
```bash
python main.py
```

### Targeted Execution
Run specific experiments (e.g., Exps 1, 3, 5, 7) on 3 concept configurations using ResNet-18 and Linear distillation:
```bash
python main.py --experiments 1 3 5 7 --num_configurations 3 --student_type resnet --projector_type linear
```

### Using ViT-Tiny Student with MLP Distillation:
```bash
python main.py --experiments 1 3 5 --student_type vit --projector_type mlp --mlp_hidden_size 1024
```

### CLI Arguments Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--experiments` | `1 2 3 ... 12` | List of experiment IDs to execute |
| `--num_configurations` | `30` | Number of concept configurations to run (1..N) |
| `--concept_configurations` | `None` | Explicit list of configuration IDs (e.g., `--concept_configurations 1 2 5`) |
| `--student_type` | `resnet` | Student architecture: `resnet` (ResNet-18) or `vit` (ViT-Tiny) |
| `--projector_type` | `linear` | Distillation projector: `linear`, `mlp`, or `rkd` |
| `--mlp_hidden_size` | `1024` | Hidden dimension for MLP projector |
| `--stream_batch_size` | `10` | Mini-batch size during the streaming phase |
| `--stream_epochs` | `1` | Training epochs per incoming streaming batch |
| `--lr_ft` | `1e-4` | Learning rate for student fine-tuning |
| `--lr_dist_proj` | `1e-3` | Learning rate for feature distillation projector |
| `--distill_weight` | `1.0` | Scalar weight $\lambda$ for feature distillation loss |
| `--rolling_window_size` | `1000` | Window size for rolling accuracy computation |
| `--adaptation_window_size` | `500` | Window size for concept adaptation speed analysis |

*Results are automatically saved and checkpointed after each configuration in `experiment_results.json`.*

---

## 📊 Generating Visualizations and Metrics

Once `experiment_results.json` is generated, run the plotting suite:

```bash
python plotting.py --results experiment_results.json --output_dir ./plots --no_show
```

### Generated Figures in `./plots/global/`:
1. **`global_adaptation_speed.png`**: 4-bar breakdown quantifying adaptation speed across all concepts:
   * **Start:** Accuracy in the first 500 samples of a concept.
   * **Adaptation:** Accuracy between samples 500 and 1,000.
   * **End of Concept:** Accuracy in the last 500 samples.
   * **Total:** Overall accuracy throughout the concept.
2. **`global_rolling_accuracy.png`**: Mean rolling accuracy trajectory (window = 1,000 samples) with drift boundary markers.
3. **`global_cumulative_accuracy.png`**: Cumulative accuracy curves with per-concept segmentation.
4. **`global_cl_matrices.png`**: Continual Learning heatmaps showing forward transfer, learning, and backward transfer (Mean ± Std in each cell).
5. **`global_forgetting.png`**: Immediate accuracy after learning vs final accuracy at the end of stream.
6. **`detailed_metrics_global_summary.csv`**: Aggregated metrics table with mean and standard deviation.

---

## 📂 Repository Structure

```
master_thesis/
├── Config.py               # Hyperparameter and configuration management
├── Student.py              # ResNet-18 student wrapper with intermediate feature extraction
├── StudentViT.py           # ViT-Tiny student alternative
├── FoundationModel.py      # DINOv2 ViT-Base with LoRA adapters and SSL pre-training wrapper
├── FeatureDistillation.py  # Linear, MLP, and Relational Knowledge Distillation modules
├── FmowManager.py          # Dataset download, temporal splits, and concept scheduling
├── FmowTorchDataset.py     # PyTorch Dataset supporting on-the-fly caching & teacher embeddings
├── TrainingManager.py      # DINO pre-training, offline student warm-up, and online streaming loops
├── ExperimentRunner.py     # Registry and execution engine for Experiments 1..12
├── LinearProbe.py          # Linear classification head for representation evaluation
├── main.py                 # Primary entry point orchestrating end-to-end execution
├── plotting.py             # Visualization and statistical analysis suite
└── requirements.txt        # Python package dependencies
```
