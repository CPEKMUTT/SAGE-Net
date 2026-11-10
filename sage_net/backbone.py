from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SUPPORTED_ARCHS = ("ViT-B-32", "ViT-L-14")


def _load_state_dict(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        state = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state dict")
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


class RemoteCLIP(nn.Module):
    """RemoteCLIP image and text encoders with all weights frozen."""

    def __init__(self, checkpoint: str | Path, arch: str = "ViT-B-32"):
        super().__init__()
        if arch not in SUPPORTED_ARCHS:
            raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}, got {arch!r}")
        import open_clip

        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"RemoteCLIP checkpoint not found at {path}. "
                "Run scripts/download_remoteclip.py or download it from "
                "https://huggingface.co/chendelong/RemoteCLIP"
            )

        clip = open_clip.create_model(arch, pretrained=None)
        clip.load_state_dict(_load_state_dict(path), strict=True)
        clip.float().eval()
        for p in clip.parameters():
            p.requires_grad_(False)

        self.clip = clip
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.arch = arch
        self.embed_dim = int(clip.visual.output_dim)
        self.image_size = 224

    def train(self, mode: bool = True):
        # The encoders are never put in training mode.
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return self.clip.visual.conv1.weight.device

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], batch_size: int = 128) -> Tensor:
        """L2-normalised text embeddings, shape [len(texts), d]."""
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(list(texts[i:i + batch_size])).to(self.device)
            out.append(F.normalize(self.clip.encode_text(tokens).float(), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def encode_image(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return projected patch tokens [B, P, d] and the global embedding [B, d]."""
        visual = self.clip.visual
        captured = {}

        def hook(_module, _inputs, output):
            captured["x"] = output[0] if isinstance(output, tuple) else output

        handle = visual.transformer.register_forward_hook(hook)
        try:
            global_feat = self.clip.encode_image(images)
        finally:
            handle.remove()

        x = captured["x"]
        batch_first = getattr(visual.transformer, "batch_first", None)
        if batch_first is None:
            batch_first = x.shape[0] == images.shape[0]
        if not batch_first:
            x = x.permute(1, 0, 2)
        x = x[:, 1:, :]

        side = math.isqrt(x.shape[1])
        if side * side != x.shape[1]:
            raise RuntimeError(f"expected a square patch grid, got {x.shape[1]} tokens")

        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x.float(), F.normalize(global_feat.float(), dim=-1)
