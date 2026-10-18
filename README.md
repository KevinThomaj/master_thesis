# Streaming Feature Distillation from a Foundation Model to an Edge Learner

Code for the MSc thesis on **knowledge transfer from a centralised Slow Learner (a foundation model) to a resource-constrained Fast Learner that learns online from a non-stationary stream**, with a limited communication budget between the two.

The Fast Learner (a small student network) learns from a stream of satellite images with concept drift. While it learns, it aligns its embedding with the embedding of a frozen foundation model (the teacher) through a small trainable **projector**. The teacher is queried only on a limited number of samples per concept. The projector is used only during training, so the deployed student is exactly as large and as fast as one trained without distillation.

The application is onboard Earth Observation, simulated with [WILDS-fMoW](https://wilds.stanford.edu/datasets/#fmow):

| Component | Implementation |
|---|---|
| Teacher (Slow Learner) | DINOv2 ViT-B/14, adapted to satellite imagery with LoRA + DINO continued pre-training on 2002–2013 images |
| Students (Fast Learner) | ResNet-18 or ViT-tiny, both starting from ImageNet weights |
| Projectors | Linear or MLP (one hidden layer) |
| Historical data | 2002–2013, 25 classes × 2,200 images |
| Stream | 2016–2017, 25 classes × 2,300 images, split into 5 concepts of 5 new classes each |

Years 2014–2015 are discarded, so the student meets an abrupt shift when the stream starts.

---

## Repository structure

All code is in `src/`: `main.py` runs the pipeline and `plotting.py` makes the figures. The other modules hold the configuration, the data, the models, the projectors (`FeatureDistillation.py`), training and the experiment registry (`ExperimentRunner.py`).

Data, weights, embeddings and results are generated in the repository root and are not tracked by git. **Always run commands from the repository root**, since all paths are relative to it.

---

## Installation

Python 3.10+ and a CUDA GPU are recommended. The thesis results were obtained on a single NVIDIA RTX 4000 SFF Ada (20 GB).

```bash
pip install -r requirements.txt
```

For GPU support, install the CUDA build of PyTorch first, following [pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

WILDS-fMoW is downloaded automatically on the first run into `./data/fmow_v1.1/`. The download is large (tens of GB).

---

## Usage

```bash
python src/main.py [options]
```

The pipeline runs these stages:

1. Load WILDS-fMoW, split it into historical (2002–2013) and streaming (2016–2017) periods, select and balance the 25 classes.
2. Adapt the teacher with LoRA on all historical images (**25 epochs, the longest step**), or load the saved weights.
3. Pre-train the students needed by the selected experiments on historical data, with or without the projector.
4. Compute the teacher embeddings of the stream and compare the off-the-shelf and adapted teacher by linear probing.
5. For each class partition, build the stream and run every selected experiment.
6. Save the results to `experiment_results.json`.

Stages 2–4 are cached on disk (see [Cached files](#cached-files)), so later runs go straight to the streaming experiments.

### Examples

Unlimited-access study, ResNet-18 with the linear projector, 12 class partitions:

```bash
python src/main.py --student_type resnet --projector_type linear \
                   --experiments 1 2 3 4 5 6 7 --num_configurations 12
```

Same study for ViT-tiny with the MLP projector (512 hidden units, as in the thesis):

```bash
python src/main.py --student_type vit --projector_type mlp --mlp_hidden_size 512 \
                   --experiments 1 2 3 4 5 6 7 --num_configurations 12
```

Communication-budget study (ViT-tiny, linear projector):

```bash
python src/main.py --student_type vit --projector_type linear \
                   --experiments 5 7 9 10 11 --num_configurations 12
```

Only specific class partitions:

```bash
python src/main.py --experiments 3 5 --concept_configurations 1 4 7
```

### Main options

| Option | Default | Description |
|---|---|---|
| `--student_type` | `resnet` | `resnet` (ResNet-18) or `vit` (ViT-tiny) |
| `--projector_type` | `linear` | `linear`, `mlp`, or `rkd` (experimental, not used in the thesis) |
| `--mlp_hidden_size` | `1024` | Hidden width of the MLP projector (the thesis uses `512`) |
| `--experiments` | `1 … 12` | Experiment IDs to run (see below) |
| `--num_configurations` | `30` | Run class partitions `1 … N` (the thesis uses `12`) |
| `--concept_configurations` | – | Explicit list of partition IDs; overrides `--num_configurations` |
| `--stream_batch_size` | `10` | Buffer size on the stream |
| `--stream_epochs` | `1` | Passes per buffer (1 = single pass) |
| `--lr_ft` | `1e-4` | Student learning rate on the stream |
| `--lr_dist_proj` | `1e-3` | Projector learning rate on the stream |
| `--distill_weight` | `1.0` | Weight λ of the distillation loss |
| `--rolling_window_size` | `1000` | Window for the rolling accuracy |
| `--adaptation_window_size` | `500` | Window after each drift for the adaptation accuracy |
| `--recurrent_concept` | – | Repeat a concept at the end of the stream, e.g. `Concept_1` |

### Plotting

The experiments write no figures. Once a run has finished, generate the plots from `experiment_results.json`:

```bash
python src/plotting.py
```

---

## Experiments

Experiments are defined in the registry in `src/ExperimentRunner.py`. `S(Hist)` means the student was pre-trained on historical data with cross-entropy only; `S+P(Hist)` means it was pre-trained together with the projector (cross-entropy + distillation).

| ID | Before deployment | On the stream | Teacher budget per concept |
|---|---|---|---|
| 1 | `S(Hist)` | inference only | – |
| 2 | `S+P(Hist)` | inference only | – |
| 3 | `S(Hist)` | fine-tuning, labels only (**main baseline**) | 0 |
| 4 | `S(Hist)` | fine-tuning + distillation, new projector | all samples |
| 5 | `S+P(Hist)` | fine-tuning + distillation, pre-trained projector (**full method**) | all samples |
| 6 | `S+P(Hist)` | as 5, but projector frozen | all samples |
| 7 | `S+P(Hist)` | fine-tuning, labels only | 0 |
| 8 | `S+P(Hist)` | as 5, distillation stops after | 1,000 |
| 9 | `S+P(Hist)` | as 5, distillation stops after | 2,000 |
| 10 | `S+P(Hist)` | as 5, distillation stops after | 4,000 |
| 11 | `S+P(Hist)` | as 5, distillation stops after | 8,000 |
| 12 | – | frozen adapted teacher + linear probe trained on the stream (reference) | – |


To add an experiment, add a new `ExperimentSetup` entry to `get_experiment_registry()`. Nothing else needs to change.

Each projector takes the student and teacher embeddings and **returns the distillation loss directly** (MSE averaged over samples and dimensions):

```python
distiller = LinearDistiller(dimFeatureStudent=512, dimFeatureTeacher=768)
loss = ce_loss + distill_weight * distiller(student_features, teacher_features)
```

---

## Outputs

`experiment_results.json` has one entry per class partition and per experiment:

```json
{
  "config_1": {
    "exp_5": {
      "final_accuracy": ...,
      "history": { ... },
      "cl_matrix": [[ ... ]]
    }
  }
}
```

- `final_accuracy`: prequential accuracy over the whole stream.
- `history`: cumulative, rolling and per-concept adaptation accuracies.
- `cl_matrix`: accuracy matrix, where entry `[i][j]` is the accuracy on concept `j` after learning concept `i`.

The file is rewritten after every partition, so an interrupted run keeps all completed partitions. **It is overwritten by each new run**, so rename it before launching another study.

---

## Cached files

Expensive stages are saved in the repository root and reused on later runs:

| File | Content |
|---|---|
| `pretrained_dinov2_lora.pth` | Adapted teacher |
| `pretrained_student_weights_<student>.pth` | Student pre-trained with cross-entropy only |
| `pretrained_student_proj_weights_<student>_<projector>.pth` | Student pre-trained with the projector |
| `pretrained_projector_weights_<student>_<projector>.pth` | Projector pre-trained with the student |
| `dino_preDF` | Teacher embeddings of the historical images (Parquet) |
| `dino` | Adapted-teacher embeddings of the stream (Parquet) |
| `dino_noExt` | Off-the-shelf DINOv2 embeddings of the stream (Parquet) |

Because the teacher is frozen during the stream, its embeddings are computed once and stored. A "query" to the teacher in the streaming loop is a lookup of the stored embedding, and the budget decides whether that embedding is used.

If you retrain the teacher, **delete the three embedding files** as well, since they are not invalidated automatically. Delete a `.pth` file to force the corresponding model to be retrained.

---

## Reproducibility

- Class sampling, splits, stream order and class partitions use fixed seeds, so the same command produces the same data.
- Partition 1 groups classes by index modulo 5, partition 2 groups consecutive classes, and partitions 3+ are seeded random permutations, guaranteed to be distinct from each other.
- Thesis results are averages over partitions 1–12.
