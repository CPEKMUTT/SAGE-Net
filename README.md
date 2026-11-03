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

## Set up the environment

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

This implementation supports `ViT-B-32` and `ViT-L-14`. Use the exact architecture that corresponds to the checkpoint.

## Prepare datasets

Arrange each dataset in class folders.

```text
data/NWPU-RESISC45/
  airport.jpg
  harbor.jpg
  forest.jpg
  ...
data/AID/
  Airport.jpg
  Port.jpg
  ...
data/UCM/
  agricultural.tif
  harbor.tif
  ...
```

Split:

```bash
python prepare_data.py --dataset-root data/NWPU-RESISC45 --dataset nwpu --output manifests/nwpu42.json --seed 42
python prepare_data.py --dataset-root data/AID --dataset aid --output manifests/aid42.json --seed 42
python prepare_data.py --dataset-root data/UCM --dataset ucm --output manifests/ucm42.json --seed 42
```

## Run few-shot adaptation

Each independent episode samples **five classes**, `K` training-support examples per class, and **15 held-out test queries per class**.

```bash
# NWPU-RESISC45, 5-way 1-shot

python run_experiment.py \
  --mode fewshot --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --architecture ViT-B-32 --shots 1 --ways 5 --queries 15 \
  --episodes --seed 42 --output runs/nwpu_1shot

# NWPU-RESISC45, 5-way 5-shot

python run_experiment.py \
  --mode fewshot --manifest manifests/nwpu42.json \
  --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
  --architecture ViT-B-32 --shots 5 --ways 5 --queries 15 \
  --episodes --seed 42 --output runs/nwpu_5shot

# Repeat with manifests/aid42.json for AID
```

## Train and evaluate cross-dataset transfer

This experiment trains only on source **NWPU-RESISC45** classes that have explicit semantic matches to the target dataset. The mappings are editable in `configs/transfer_mappings.json`

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

Replace `aid` with `ucm`, and use `manifests/ucm42.json`, for the UC-Merced experiment.


## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q sage_net run_experiment.py prepare_data.py
```


## Acknowledgment

We thank the authors of **[APPLeNet](https://github.com/mainaksingha01/APPLeNet)** for generously releasing their code. Its availability served as a valuable reference for few-shot remote sensing experimentation and helped accelerate this research. We also acknowledge the **[RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)** team for releasing pretrained vision-language models.
