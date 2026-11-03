from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ImageClassificationDataset, SceneSample, image_transforms
from .model import SAGENet, sage_loss, trainable_parameter_count
from .prompts import encode_prompt_bank


@dataclass
class ExperimentConfig:
    scene_tokens: int = 6
    assignment_dim: int = 128    
    prompt_hidden: int = 256     
    geometry_hidden: int = 128    
    relation_hidden: int = 128    
    temperature: float = 0.07   
    alignment_weight: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 0
    image_size: int = 224
    identity_only: bool = False
    prompt_fusion: str = "gated"  # gated, equal, identity, identity_object, identity_geometry
    geometry_mode: str = "full"  # none, centroid, centroid_area, centroid_area_aspect, full
    use_geometry: bool = True
    use_geometry_bias: bool = True
    prompt_overrides: str | None = None
    allow_generic_prompts: bool = False

    @classmethod
    def from_json(cls, filename: str | Path) -> "ExperimentConfig":
        values = json.loads(Path(filename).read_text(encoding="utf-8"))
        allowed = set(cls.__dataclass_fields__)
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"Unknown configuration parameters: {sorted(extra)}")
        return cls(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_choice(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def build_sage(backbone: nn.Module, classes: Sequence[str], cfg: ExperimentConfig,
               device: torch.device, *, prompt_embeddings: Tensor | None = None
               ) -> SAGENet:
    if not classes:
        raise ValueError("At least one class is required")
    if prompt_embeddings is None:
        with torch.no_grad():
            prompt_embeddings = encode_prompt_bank(
                backbone, classes, override=cfg.prompt_overrides,
                strict=not cfg.allow_generic_prompts,
            )
    model = SAGENet(
        backbone, prompt_embeddings, scene_tokens=cfg.scene_tokens,
        assignment_dim=cfg.assignment_dim,
        prompt_hidden=cfg.prompt_hidden, geometry_hidden=cfg.geometry_hidden,
        relation_hidden=cfg.relation_hidden, temperature=cfg.temperature,
        identity_only=cfg.identity_only, use_geometry=cfg.use_geometry,
        use_geometry_bias=cfg.use_geometry_bias,
        prompt_fusion=cfg.prompt_fusion, geometry_mode=cfg.geometry_mode,
    ).to(device)
    return model


def make_loader(samples: Sequence[SceneSample], classes: Sequence[str],
                cfg: ExperimentConfig, *, train: bool,
                seed: int | None = None) -> DataLoader:
    dataset = ImageClassificationDataset(
        samples, classes, image_transforms(train=train, size=cfg.image_size)
    )
    if len(dataset) == 0:
        raise ValueError("Cannot create an empty dataloader")
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=train,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_sage(model: SAGENet, samples: Sequence[SceneSample],
               classes: Sequence[str], cfg: ExperimentConfig, *,
               device: torch.device, seed: int = 42) -> list[dict[str, float]]:
    loader = make_loader(samples, classes, cfg, train=True, seed=seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        accum = {"total": 0.0, "cross_entropy": 0.0, "alignment": 0.0}
        count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            losses = sage_loss(out, labels, alignment_weight=cfg.alignment_weight,
                               temperature=cfg.temperature)
            losses["total"].backward()
            optimizer.step()
            n = labels.shape[0]
            for key in accum:
                accum[key] += float(losses[key].detach()) * n
            count += n
        scheduler.step()
        history.append({"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0],
                        **{key: value / count for key, value in accum.items()}})
    return history


@torch.no_grad()
def evaluate_sage(model: SAGENet, samples: Sequence[SceneSample],
                  classes: Sequence[str], cfg: ExperimentConfig, *,
                  device: torch.device) -> dict:
    loader = make_loader(samples, classes, cfg, train=False)
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    for images, labels in loader:
        out = model(images.to(device))
        truth.extend(labels.tolist())
        predicted.extend(out["logits"].argmax(dim=-1).cpu().tolist())
    accuracy = sum(a == b for a, b in zip(predicted, truth)) / len(truth)
    by_class = {}
    for idx, cls in enumerate(classes):
        n = sum(y == idx for y in truth)
        if n:
            by_class[cls] = sum(y == idx and pred == idx for y, pred in zip(truth, predicted)) / n
    return {"accuracy": accuracy, "examples": len(truth),
            "per_class_accuracy": by_class, "targets": truth, "predictions": predicted}


@torch.no_grad()
def evaluate_zero_shot(backbone: nn.Module, samples: Sequence[SceneSample],
                       classes: Sequence[str], cfg: ExperimentConfig, *,
                       device: torch.device) -> dict:
    loader = make_loader(samples, classes, cfg, train=False)
    texts = [
        f"A remote sensing image of {'an' if name[0] in 'aeiou' else 'a'} "
        f"{name.replace('_', ' ')}." for name in classes
    ]
    prototypes = F.normalize(backbone.encode_texts(texts).to(device), dim=-1)
    correct = total = 0
    for images, labels in loader:
        _patches, global_feature = backbone.encode_visual(images.to(device))
        scores = global_feature @ prototypes.t()
        correct += int((scores.argmax(dim=-1).cpu() == labels).sum())
        total += labels.numel()
    return {"accuracy": correct / total, "examples": total}


class FrozenGlobalLinearProbe(nn.Module):

    def __init__(self, backbone: nn.Module, classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, classes)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        _patches, global_feature = self.backbone.encode_visual(images)
        return self.head(global_feature)


def train_probe(model: FrozenGlobalLinearProbe, samples: Sequence[SceneSample],
                classes: Sequence[str], cfg: ExperimentConfig, *,
                device: torch.device, seed: int = 42) -> None:
    loader = make_loader(samples, classes, cfg, train=True, seed=seed)
    optim = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    for _epoch in range(cfg.epochs):
        model.train()
        for images, labels in loader:
            optim.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            optim.step()
        scheduler.step()


@torch.no_grad()
def evaluate_probe(model: FrozenGlobalLinearProbe,
                   samples: Sequence[SceneSample], classes: Sequence[str],
                   cfg: ExperimentConfig, *, device: torch.device) -> dict:
    loader = make_loader(samples, classes, cfg, train=False)
    model.eval()
    correct = total = 0
    for images, labels in loader:
        scores = model(images.to(device))
        correct += int((scores.argmax(-1).cpu() == labels).sum())
        total += labels.numel()
    return {"accuracy": correct / total, "examples": total}


def confidence_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize zero episodes")
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return {"mean_accuracy": avg, "std_accuracy": sd,
            "ci95_half_width": 1.96 * sd / math.sqrt(len(values)),
            "episodes": len(values)}


def save_sage_checkpoint(path: str | Path, model: SAGENet,
                         classes: Sequence[str], cfg: ExperimentConfig,
                         *, backbone_architecture: str, history: list | None = None,
                         metadata: dict | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never duplicate the full pretrained backbone in a lightweight adapter ckpt.
    adapter_state = {
        key: tensor.detach().cpu()
        for key, tensor in model.state_dict().items()
        if not key.startswith("backbone.") and key != "prompt_bank.embeddings"
    }
    torch.save({
        "format": "SAGE-Net-adapter-v1", "classes": list(classes),
        "architecture": backbone_architecture,
        "config": vars(cfg), "model": adapter_state,
        "prompt_embeddings": model.prompt_bank.embeddings.detach().cpu(),
        "history": history or [], "metadata": metadata or {},
        "method": "SSPB+GSTC+SGRM+CE+alignment",
    }, target)
    return target


def load_sage_checkpoint(path: str | Path, backbone: nn.Module,
                         device: torch.device) -> tuple[SAGENet, list[str], ExperimentConfig, dict]:
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    if saved.get("format") != "SAGE-Net-adapter-v1":
        raise ValueError("Not a SAGE-Net v1 adapter checkpoint")
    if saved["architecture"] != backbone.architecture:
        raise ValueError("RemoteCLIP architecture does not match saved adapter")
    cfg = ExperimentConfig(**saved["config"])
    classes = saved["classes"]
    # Preserve the exact frozen text embeddings used in training, even when
    # the external prompt override JSON is unavailable or later edited.
    model = build_sage(
        backbone, classes, cfg, device,
        prompt_embeddings=saved["prompt_embeddings"],
    )
    current = model.state_dict()
    current.update(saved["model"])
    model.load_state_dict(current, strict=True)
    model.eval()
    return model, classes, cfg, saved.get("metadata", {})
