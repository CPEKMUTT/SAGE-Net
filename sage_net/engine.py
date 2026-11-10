from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ImageList, build_transform
from .model import SAGENet, count_trainable, sage_loss
from .prompts import PROMPT_SETS, build_prompts


@dataclass
class Config:
    # architecture
    num_tokens: int = 6
    assign_dim: int = 128
    gate_hidden: int = 256
    geo_hidden: int = 128
    rel_hidden: int = 64
    temperature: float = 0.07
    dropout: float = 0.1
    # ablation switches
    prompt_set: str = "full"            # classname | identity | identity_object | identity_geometry | full
    prompt_fusion: str = "gated"        # gated | equal
    geometry: str = "none"              # none | centroid | centroid_area | centroid_area_aspect | full
    inject_geometry: bool = True        # add phi_g(g) to the scene tokens
    geometry_bias: bool = True          # geometry-biased attention 
    lambda_align: float = 0.5
    # optimisation
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 50
    batch_size: int = 32
    image_size: int = 224
    num_workers: int = 4
    amp: bool = True
    prompt_overrides: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "Config":
        values = json.loads(Path(path).read_text()) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.prompt_set not in PROMPT_SETS:
            raise ValueError(f"prompt_set must be one of {list(PROMPT_SETS)}")
        return cfg


VARIANTS = {
    "full": {},
    "classname_prompt": {"prompt_set": "classname"},
    "no_sspb": {"prompt_set": "identity"},
    "no_geometry_descriptors": {"inject_geometry": False},
    "no_geometry_bias": {"geometry_bias": False},
    "no_alignment": {"lambda_align": 0.0},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(items, cfg: Config, train: bool, seed: int = 0, cache: bool = False) -> DataLoader:
    dataset = ImageList(items, build_transform(train, cfg.image_size), cache=cache)
    gen = torch.Generator().manual_seed(seed)
    # Cached datasets live in the main process; workers would each hold a copy.
    workers = 0 if cache else cfg.num_workers
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=train, generator=gen,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False)


def build_sage(encoder, prompt_embeddings: Tensor, cfg: Config) -> SAGENet:
    model = SAGENet(
        encoder, prompt_embeddings,
        num_tokens=cfg.num_tokens, assign_dim=cfg.assign_dim, gate_hidden=cfg.gate_hidden,
        geo_hidden=cfg.geo_hidden, rel_hidden=cfg.rel_hidden, temperature=cfg.temperature,
        prompt_fusion=cfg.prompt_fusion, geometry=cfg.geometry,
        inject_geometry=cfg.inject_geometry, geometry_bias=cfg.geometry_bias, dropout=cfg.dropout,
    )
    return model.to(encoder.device)


def _encode(encoder, images: Tensor, amp: bool) -> tuple[Tensor, Tensor]:
    use_amp = amp and images.is_cuda
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        patches, global_feat = encoder.encode_image(images)
    return patches.float(), global_feat.float()


def train_sage(model: SAGENet, loader: DataLoader, cfg: Config, log_every: int = 0) -> list[dict]:
    device = model.encoder.device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "align": 0.0, "acc": 0.0}
        seen = 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            patches, _ = _encode(model.encoder, images, cfg.amp)
            out = model.forward_features(patches)
            losses = sage_loss(out, labels, cfg.lambda_align, cfg.temperature)

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            scheduler.step()

            n = labels.numel()
            seen += n
            for k in ("loss", "ce", "align"):
                sums[k] += losses[k].item() * n
            sums["acc"] += (out["logits"].argmax(-1) == labels).sum().item()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                  **{k: v / seen for k, v in sums.items()}}
        history.append(record)
        if log_every and (epoch % log_every == 0 or epoch == cfg.epochs):
            print(f"  epoch {epoch:3d}  loss {record['loss']:.4f}  ce {record['ce']:.4f}  "
                  f"align {record['align']:.4f}  train acc {record['acc']:.4f}", flush=True)
    return history


@torch.no_grad()
def predict_sage(model: SAGENet, loader: DataLoader, cfg: Config, keep_features: bool = False) -> dict:
    device = model.encoder.device
    model.eval()
    preds, targets, feats, alphas = [], [], [], None
    for images, labels in loader:
        patches, _ = _encode(model.encoder, images.to(device), cfg.amp)
        out = model.forward_features(patches)
        preds.append(out["logits"].argmax(-1).cpu())
        targets.append(labels)
        if keep_features:
            feats.append(out["pooled"].cpu())
        alphas = out["prompt_weights"].cpu()
    result = {"pred": torch.cat(preds).numpy(), "target": torch.cat(targets).numpy(),
              "prompt_weights": alphas.numpy()}
    if keep_features:
        result["features"] = torch.cat(feats).numpy()
    result["accuracy"] = float((result["pred"] == result["target"]).mean())
    return result


class LinearProbe(nn.Module):
    """Baseline: frozen RemoteCLIP global embedding + linear classifier."""

    def __init__(self, encoder, num_classes: int):
        super().__init__()
        self.__dict__["encoder"] = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward_features(self, global_feat: Tensor) -> Tensor:
        return self.head(global_feat)


def train_probe(model: LinearProbe, loader: DataLoader, cfg: Config):
    device = model.encoder.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader))
    for _ in range(cfg.epochs):
        model.train()
        for images, labels in loader:
            _, g = _encode(model.encoder, images.to(device), cfg.amp)
            loss = F.cross_entropy(model.forward_features(g), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()


@torch.no_grad()
def predict_probe(model: LinearProbe, loader: DataLoader, cfg: Config) -> dict:
    device = model.encoder.device
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        _, g = _encode(model.encoder, images.to(device), cfg.amp)
        preds.append(model.forward_features(g).argmax(-1).cpu())
        targets.append(labels)
    pred, target = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {"pred": pred, "target": target, "accuracy": float((pred == target).mean())}


@torch.no_grad()
def predict_zero_shot(encoder, class_names: Sequence[str], loader: DataLoader, cfg: Config) -> dict:
    """Zero-shot RemoteCLIP with the identity prompt."""
    texts = [p[0] for p in build_prompts(class_names, "identity", cfg.prompt_overrides)]
    text = encoder.encode_text(texts)
    preds, targets = [], []
    for images, labels in loader:
        _, g = _encode(encoder, images.to(encoder.device), cfg.amp)
        preds.append((g @ text.t()).argmax(-1).cpu())
        targets.append(labels)
    pred, target = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {"pred": pred, "target": target, "accuracy": float((pred == target).mean())}


def summarize(accuracies: Sequence[float]) -> dict:
    acc = np.asarray(accuracies, dtype=float) * 100
    std = float(acc.std(ddof=1)) if len(acc) > 1 else 0.0
    return {"mean": float(acc.mean()), "std": std,
            "ci95": 1.96 * std / math.sqrt(len(acc)), "n": int(len(acc))}


def save_adapter(path: str | Path, model: SAGENet, class_names: Sequence[str], cfg: Config,
                 arch: str, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "classes": list(class_names),
        "config": asdict(cfg),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "trainable_params": count_trainable(model),
        **(extra or {}),
    }, path)


def load_adapter(path: str | Path, encoder) -> tuple[SAGENet, list[str], Config]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["arch"] != encoder.arch:
        raise ValueError(f"adapter was trained with {ckpt['arch']}, encoder is {encoder.arch}")
    cfg = Config(**ckpt["config"])
    prompts = ckpt["state_dict"]["sspb.embeddings"]
    model = build_sage(encoder, prompts, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], cfg

