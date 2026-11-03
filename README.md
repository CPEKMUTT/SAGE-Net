# SAGE-Net

**Semantic and Geometry-Aware Foundation Adaptation for Few-Shot Remote Sensing Scene Classification**

![Task](https://img.shields.io/badge/Task-Few--shot%20Remote%20Sensing-3766A2)
![Backbone](https://img.shields.io/badge/Backbone-Frozen%20RemoteCLIP-7949AD)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)

> [!IMPORTANT]
> **Release policy:** The source code of SAGE-Net will be made publicly available in this repository after final acceptance of the manuscript. The training scripts, evaluation scripts, configuration files, and instructions are not yet released. Please check back for updates.

## What is SAGE-Net?

SAGE-Net adapts frozen RemoteCLIP representations to remote sensing scene classification through semantic and geometry-aware modules:

1. **Structured Semantic Prompt Bank (SSPB)**: class identity, object composition, and geometry/layout prompts are encoded by the frozen RemoteCLIP text tower and combined using a learnable class-specific gate.
2. **Geometry-Aware Scene Token Constructor (GSTC)**: visual ViT patches are softly assigned to six scene groups; each group receives a 6-value geometry vector representing centroid, area, aspect ratio, and orientation.
3. **Semantic-Geometry Reasoning Module (SGRM)**: geometry-biased self-attention refines scene tokens, then class-specific semantic matching aggregates evidence for the candidate categories.

The RemoteCLIP visual and text encoders **remain frozen** during adaptation. 

## Project structure

```text
SAGE-Net/
├── sage_net/
│   ├── backbone.py      # Frozen RemoteCLIP with ViT patch extraction
│   ├── model.py         # SSPB, GSTC, SGRM, fusion, and losses
│   ├── prompts.py       # Class-specific semantic prompt definitions
│   ├── data.py          # Split manifests, datasets, and episodic sampler
│   └── engine.py        # Training, baselines, metrics, and checkpointing
├── configs/
│   ├── sage_default.json
│   ├── dataset_classes.json
│   ├── transfer_mappings.json
│   └── prompt_overrides.example.json
├── scripts/          
├── tests/              
├── prepare_data.py
├── run_experiment.py
└── requirements.txt
```

## 1. Set up the environment

Requires Python 3.10+ and a CUDA-capable GPU for full-scale experiments.

```bash
python -m venv sagenet
conda activate sagenet
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download a pretrained checkpoint from the **[official RemoteCLIP project](https://github.com/ChenDelong1999/RemoteCLIP)** or its official [Hugging Face repository](https://huggingface.co/chendelong/RemoteCLIP). Place it at a local path such as:

```text
checkpoints/RemoteCLIP-ViT-B-32.pt
```

This implementation supports `ViT-B-32` and `ViT-L-14`. Use the exact architecture that corresponds to the checkpoint. **No pretrained weights or remote sensing datasets are redistributed here.**

## 2. Prepare datasets and fixed splits

Arrange each dataset in class folders. The script also accepts a dataset containing an `images/` subdirectory:

```text
data/NWPU-RESISC45/
  airport/*.jpg
  harbor/*.jpg
  forest/*.jpg
  ...
data/AID/
  Airport/*.jpg
  Port/*.jpg
  ...
data/UCM/
  agricultural/*.tif
  harbor/*.tif
  ...
```

Create deterministic stratified split manifests:

```bash
python prepare_data.py --dataset-root data/NWPU-RESISC45 --dataset nwpu --output manifests/nwpu42.json --seed 42
python prepare_data.py --dataset-root data/AID --dataset aid --output manifests/aid42.json --seed 42
python prepare_data.py --dataset-root data/UCM --dataset ucm --output manifests/ucm42.json --seed 42
```

**Important:** The default `60% / 20% / 20%` image-level splits are a convenience for new experiments, **not** the official manuscript splits. To claim reproduction, replace these manifests with the exact train/test protocol, class selection, seeds, and sample identities used for the reported results. See [reproducibility notes](docs/REPRODUCIBILITY.md).

The optional `--dataset` argument verifies the expected 45 NWPU, 30 AID, or
21 UC-Merced categories after name normalization. Omit it for an intentional
subset of classes.

## 3. Run few-shot adaptation

Each independent episode samples **five classes**, `K` training-support examples per class, and **15 held-out test queries per class**. A **fresh adaptation model** is trained from scratch for each task, preventing cross-episode query contamination. The paper specifies 50 adaptation epochs and AdamW with learning rate `1e-4`.

```bash
# NWPU-RESISC45, 5-way 1-shot
python run_experiment.py \
  --mode fewshot --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --architecture ViT-B-32 --shots 1 --ways 5 --queries 15 \
  --episodes 100 --seed 42 --output runs/nwpu_1shot

# NWPU-RESISC45, 5-way 5-shot
python run_experiment.py \
  --mode fewshot --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --architecture ViT-B-32 --shots 5 --ways 5 --queries 15 \
  --episodes 100 --seed 42 --output runs/nwpu_5shot

# Repeat with manifests/aid42.json for AID
```

The output `results.json` contains every episode's accuracy, classes, sample counts, mean, standard deviation, and an approximate 95% confidence-interval half-width. Increase `--episodes` according to the published evaluation protocol; 100 is an illustrative setting, not a manuscript-verified count.

## 4. Evaluate the backbone contribution

These two baselines are **not the same experiment**:

```bash
# True zero-shot: independent 5-way episodes, no support-label training
python run_experiment.py \
  --mode zero_shot --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --shots 1 --ways 5 --queries 15 --episodes 100 \
  --output runs/nwpu_zeroshot

# K-shot linear probe: frozen RemoteCLIP global image feature + trainable classifier
python run_experiment.py \
  --mode probe --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --shots 1 --episodes 100 --output runs/nwpu_probe_1shot
```

For the advisor-requested comparison with AGPLwS, run an independent AGPLwS implementation using the **identical checkpoint, preprocessing, split/episodes, and evaluation metric**. AGPLwS is not silently relabeled or embedded inside this SAGE-Net implementation.

## 5. Train and evaluate cross-dataset transfer

This experiment trains only on source **NWPU-RESISC45** classes that have explicit semantic matches to the target dataset. The mappings are editable in `configs/transfer_mappings.json`; check them against the manuscript before publication.

```bash
# Source-only training for NWPU -> AID
python run_experiment.py \
  --mode transfer_train --manifest manifests/nwpu42.json \
  --target-dataset aid --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --output runs/nwpu_to_aid_train

# Evaluate on AID, with no target-domain training
python run_experiment.py \
  --mode transfer_eval --manifest manifests/nwpu42.json \
  --target-manifest manifests/aid42.json --target-dataset aid \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --adapter-checkpoint runs/nwpu_to_aid_train/sage_adapter.pt \
  --output runs/nwpu_to_aid_eval
```

Replace `aid` with `ucm`, and use `manifests/ucm42.json`, for the UC-Merced experiment. The evaluation report states how many target examples were included and excluded by the class mapping.

## 6. Ablations

The command-line `--ablation` switch runs controlled alternatives using the
**same dataset manifest and episode-sampling seed**. Available variants are:

| Manuscript experiment | Configuration change |
| --- | --- |
| Full model | Default configuration |
| No structured prompt composition | `--ablation identity` |
| Identity and object prompts only | `--ablation identity_object` |
| Identity and layout prompts only | `--ablation identity_geometry` |
| Three prompts with equal weights | `--ablation equal_prompts` |
| No geometry descriptors | `--ablation no_geometry` |
| Centroid only | `--ablation centroid` |
| Centroid + area | `--ablation centroid_area` |
| Centroid + area + aspect ratio | `--ablation centroid_area_aspect` |
| No geometry bias in SGRM | `--ablation no_geometry_bias` |
| No semantic alignment loss | `--ablation no_alignment` |

For example:

```bash
python run_experiment.py \
  --mode fewshot --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --shots 5 --episodes 100 --seed 42 \
  --ablation centroid_area \
  --output runs/nwpu_centroid_area
```

Run all 11 presets with `bash scripts/ablation_nwpu_5shot.sh` (the arguments
are the manifest path, checkpoint path, and number of episodes). For **Table 11**,
set `scene_tokens` to `2`, `4`, `6`, `8`, or `10` in separate configuration JSON
files; `--config` accepts any such file. For the plain global-RemoteCLIP
baseline, use `--mode probe`. Use `--mode zero_shot` for the genuinely untrained
RemoteCLIP comparison. These experiments are not interchangeable: identity-only
SAGE-Net still trains GSTC and SGRM, whereas a global linear probe does not.

The exact formulas and implementation choices are documented in [METHOD_ALIGNMENT.md](docs/METHOD_ALIGNMENT.md).

## 7. Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q sage_net run_experiment.py prepare_data.py
```

The unit tests use synthetic features and small temporary images. They verify tensor dimensions, assignment normalization, class-specific weights, differentiability, support/query separation, and component switches. **Full numerical reproduction requires the real datasets, an official checkpoint, and a suitable GPU.**

## Citation and acknowledgment

Manuscript: **N. Sultan, M. Hayat, and S. Prom-on, "SAGE-Net: Semantic and Geometry-Aware Foundation Adaptation for Few-Shot Remote Sensing Scene Classification."** Formal citation details will be added upon publication.

We thank the authors of **[APPLeNet](https://github.com/mainaksingha01/APPLeNet)** — Mainak Singha, Ankit Jha, Bhupendra Solanki, Shirsha Bose, and Biplab Banerjee — for generously releasing their code. Its availability served as a valuable reference for few-shot remote sensing experimentation and helped accelerate this research. We also acknowledge the **[RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)** team for releasing pretrained vision-language models. The original APPLeNet MIT copyright notice is preserved in `third_party_licenses/APPLeNet-LICENSE.txt`.
