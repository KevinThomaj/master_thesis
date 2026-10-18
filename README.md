# Teaching Small Models with Big Ones, on the Fly

**Streaming feature distillation from a foundation model to an edge learner.** Code for my MSc thesis at Politecnico di Milano.

## The idea in short

Many devices have to make decisions where the data is produced: satellites, drones, robots, remote sensors. They can only run **small models**, and the world they observe **keeps changing**: new places, new seasons, new kinds of objects. When that happens, a small model becomes wrong and has to relearn quickly from very few examples.

Meanwhile, very powerful **foundation models** exist, but they are far too large to run on these devices. They live on servers or ground stations, and the link to reach them is slow and expensive.

This work connects the two. The small model keeps running and learning **on its own**, but for a limited number of images it asks the large model how it "sees" them, and learns to organise its own internal representation the same way. The large model is never modified, and it is never needed to make predictions: once the small model has learned, it works fully on its own.

Think of an apprentice who works alone but can occasionally ask an expert. They don't ask the expert to do the job; they ask how the expert looks at the problem, and get better at it themselves.

## Why it is useful

Tested on real satellite images (WILDS-fMoW), with a small onboard model guided by a ground-based foundation model (DINOv2):

- **Faster recovery after a change.** On the first 500 images after the world changes, accuracy improves by up to **11.6 percentage points**.
- **Better overall.** Accuracy over the whole stream improves by about **3 points**, both for a convolutional network (ResNet-18) and a small transformer (ViT-tiny).
- **Cheap in communication.** Asking the large model about only **17% of the images** keeps about **60% of the benefit**. The help matters most right after a change, so a small budget goes a long way.
- **No cost at deployment.** The extra component used for learning (a small projector) is removed afterwards: the deployed model is exactly as small and as fast as before.

The same situation (a small learner on the edge, a large model far away, a limited connection) appears in drones, robotics, autonomous vehicles, sensor networks and medical devices, so the approach is not limited to satellites.

## What you can do with this code

- **Reproduce the thesis results** on WILDS-fMoW with a single command.
- **Try your own models.** The large model only needs to turn an image into a vector, so you can swap DINOv2 for another foundation model, or use a different small model, without changing the rest of the pipeline.
- **Test different communication budgets**, from never asking the large model to asking it about every image.
- **Add new experiments** by writing one entry in a registry, as a starting point for your own research on learning from changing data streams.

---

## Quick start

**1. Install** (Python 3.10+; a GPU is strongly recommended). For GPU support, first install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

**2. Run** from the repository root:

```bash
python src/main.py --student_type resnet --projector_type linear --num_configurations 12
```

The first run downloads the dataset (tens of GB) and prepares the models, which takes a while. Everything expensive is saved to disk, so later runs start straight from the experiments.

**3. Plot the results:**

```bash
python src/plotting.py
```

## Repository structure

All code is in `src/`: `main.py` runs everything and `plotting.py` makes the figures. Data, model weights and results are created in the repository root and are not tracked by git. **Always run commands from the repository root.**

---

## How it works

1. **Two time periods.** Images from 2002–2013 are the "past", used to prepare the models before deployment. Images from 2016–2017 are the "future" the small model meets once deployed. The years in between are skipped, so the change is sudden.
2. **Preparing the large model.** DINOv2 was trained on everyday photos, not satellite images. It is adapted to satellite imagery using only past images and no labels, with a lightweight technique (LoRA) that fits on a single GPU.
3. **Preparing the small model.** The small model is trained on past images, optionally already learning to match the large model.
4. **The stream.** The future images arrive in small groups, organised into 5 phases, each introducing 5 new kinds of places (airports, ports, crop fields, …). For each group, the small model first predicts, then learns from the true labels and, while the budget allows, from the large model's view of the images.

---

## Reference

### Experiments

Choose which experiments to run with `--experiments`. "Pre-aligned" means the small model already learned to match the large model on past data.

| ID | Small model before deployment | During the stream | Help from the large model |
|---|---|---|---|
| 1 | Normal | Only predicts, never learns | – |
| 2 | Pre-aligned | Only predicts, never learns | – |
| 3 | Normal | Learns from labels only (**baseline**) | none |
| 4 | Normal | Learns from labels + large model | every image |
| 5 | Pre-aligned | Learns from labels + large model (**full method**) | every image |
| 6 | Pre-aligned | As 5, with the projector frozen | every image |
| 7 | Pre-aligned | Learns from labels only | none |
| 8–11 | Pre-aligned | As 5, then labels only | first 1,000 / 2,000 / 4,000 / 8,000 images of each phase |
| 12 | – | Large model + simple classifier (reference) | – |


To add an experiment, add an entry to `get_experiment_registry()` in `src/ExperimentRunner.py`.

### Main options

| Option | Default  | Description |
|---|----------|---|
| `--student_type` | `resnet` | Small model: `resnet` (ResNet-18) or `vit` (ViT-tiny) |
| `--projector_type` | `linear` | `linear` or `mlp` |
| `--mlp_hidden_size` | `512`    | Hidden size of the MLP projector (the thesis uses `512`) |
| `--experiments` | `1 … 12` | Which experiments to run |
| `--num_configurations` | `30`     | Number of different groupings of classes into phases (the thesis uses `12`) |
| `--concept_configurations` | –        | Run only specific groupings, e.g. `1 4 7` |
| `--stream_batch_size` | `10`     | Images per group on the stream |
| `--distill_weight` | `1.0`    | How much the large model's guidance counts during learning |

Run `python src/main.py --help` for the full list.

### Outputs and saved files

- `experiment_results.json` holds all results. It is saved after every grouping, so an interrupted run is not lost, but **it is overwritten by each new run**: rename it before starting another one.
- Model weights (`*.pth`) and the large model's outputs on the images (`dino`, `dino_noExt`, `dino_preDF`) are saved and reused. If you retrain the large model, delete these three `dino*` files too. Delete a `.pth` file to force that model to be retrained.

All random choices use fixed seeds, so the same command always produces the same data and results. Thesis numbers are averages over 12 groupings.

---
