# SAGE-Net

**Semantic and Geometry-Aware Foundation Adaptation for Few-Shot Remote Sensing Scene Classification**

![Task](https://img.shields.io/badge/Task-Few--shot%20Remote%20Sensing-3766A2)
![Backbone](https://img.shields.io/badge/Backbone-Frozen%20RemoteCLIP-7949AD)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)

SAGE-Net adapts frozen RemoteCLIP encoders to remote sensing scene classification with three modules:

1. **Structured Semantic Prompt Bank (SSPB).** Each class is described by an identity prompt, an object-composition prompt and a geometry-layout prompt. The prompts are encoded once by the frozen RemoteCLIP text encoder and fused into a class prototype by a learnable, class-specific gate.
2. **Geometry-Aware Scene Token Constructor (GSTC).** ViT patch tokens are softly assigned to `S` scene groups. Each group gets an appearance token and a 6-value geometry descriptor (centroid `u, v`, area, aspect ratio, `sin θ`, `cos θ`) computed from the soft assignment; the descriptor is projected by an MLP and added to the token.
3. **Semantic-Geometry Reasoning Module (SGRM).** Self-attention over the scene tokens with a pairwise geometry bias, followed by class-guided evidence aggregation against the SSPB prototypes.

## Repository layout

```text
SAGE-Net/
├── sage_net/
│   ├── backbone.py      # frozen RemoteCLIP, projected ViT patch tokens
│   ├── prompts.py       # identity / object / geometry prompts for NWPU, AID, UCM
│   ├── model.py         # SSPB, GSTC, SGRM, SAGENet, loss
│   ├── data.py          # class-folder datasets, transforms, episode sampler
│   └── engine.py        # config, training, evaluation, baselines, checkpoints
├── configs/
│   ├── default.json                     # implementation settings
│   ├── transfer_mappings.json           # NWPU -> AID / UCM class matching
│   └── prompt_overrides.example.json
├── scripts/download_remoteclip.py
├── tests/test_model.py
├── run_experiment.py
└── requirements.txt
```

## Setup

Python 3.10+.

```bash
conda create --name sage_net
conda activate sage_net
pip install -r requirements.txt
python scripts/download_remoteclip.py --arch ViT-B-32 
```

The checkpoint can also be downloaded manually from the [RemoteCLIP Hugging Face page](https://huggingface.co/chendelong/RemoteCLIP). `ViT-B-32` (7×7 patch grid at 224 px).

## Data

Each dataset is a folder with one sub-folder per class:

```text
data/NWPU-RESISC45/airplane/airplane_001.jpg
data/AID/Airport/airport_1.jpg
data/UCMerced_LandUse/Images/agricultural/agricultural00.tif
```

Class folder names are normalized automatically (`DenseResidential`, `denseresidential,` and `dense_residential` all become `dense_residential`). A single wrapper directory such as `Images/` is skipped.

## Implementation settings

| Setting | Value |
|---|---|
| Input size | 224 × 224 |
| Frozen | RemoteCLIP visual and text encoders |
| Trainable | prompt-fusion gate, GSTC, SGRM, classifier |
| Scene tokens `S` | 6 |
| Optimiser | AdamW, lr 1e-4, weight decay 5e-4 |
| Schedule | cosine annealing |
| Batch size | 32 |
| Epochs | 50 |
| `λ` (alignment) | 0.5 |
| `τ` (temperature) | 0.07 |
| Episodes | 5-way K-shot, 15 queries per class |



## Few-shot adaptation

Every episode samples 5 classes, `K` support and 15 query images per class (disjoint), trains a new adapter on the support set for 50 epochs and evaluates it on the queries. Results are averaged over episodes and seeds.

```bash
# NWPU-RESISC45, 5-way 1-shot and 5-shot
python run_experiment.py fewshot --data-root data/NWPU-RESISC45 \
    --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
    --shots 1 --episodes 100 --seeds 1 2 3 --output runs/nwpu_5w1s

python run_experiment.py fewshot --data-root data/NWPU-RESISC45 \
    --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt \
    --shots 5 --episodes 100 --seeds 1 2 3 --output runs/nwpu_5w5s

# AID: same commands with --data-root data/AID
```

`results.json` reports the mean and standard deviation over seeds, the episode-level 95% confidence interval and every episode's accuracy. `predictions.npz` stores the class names, predictions and labels of each episode (useful for confusion matrices) and, with `--save-features`, the pooled query features (for UMAP / t-SNE) and the learned prompt-fusion weights.

## Cross-dataset transfer

The adapter is trained on the NWPU-RESISC45 classes that have a counterpart in the target dataset.

```bash
python run_experiment.py transfer --source-root data/NWPU-RESISC45 \
    --target-root data/AID --target aid \
    --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt --seeds 1 2 3 --output runs/nwpu_to_aid

python run_experiment.py transfer --source-root data/NWPU-RESISC45 \
    --target-root data/UCMerced_LandUse --target ucm \
    --checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt --seeds 1 2 3 --output runs/nwpu_to_ucm
```


## Prompts

The descriptions for all 45 NWPU-RESISC45, 30 AID and 21 UC-Merced classes are in `sage_net/prompts.py`. Prompt sample:

| Type | Prompt |
|---|---|
| Identity | A remote sensing image of a harbor. |
| Object composition | A harbor scene usually contains water, ships, docks, piers, and port infrastructure. |
| Geometry layout | A harbor often contains ships and docks arranged along water boundaries, with linear piers connected to coastal areas. |


## Acknowledgment

We thank the authors of [APPLeNet](https://github.com/mainaksingha01/APPLeNet) for releasing their code, which served as a reference for few-shot remote sensing experiments, and the [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) team for releasing pretrained vision-language models.
