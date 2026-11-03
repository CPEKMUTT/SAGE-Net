from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RemoteCLIPBackbone(nn.Module):

    def __init__(self, checkpoint: str | Path, architecture: str = "ViT-B-32"):
        super().__init__()
        if architecture not in ("ViT-B-32", "ViT-L-14"):
            raise ValueError(
                f"{architecture!r} is unsupported for spatial ViT patch extraction; "
                "choose ViT-B-32 or ViT-L-14."
            )
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("Install open_clip_torch from requirements.txt") from exc

        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"RemoteCLIP checkpoint not found: {path}. Download the official "
                "checkpoint separately"
            )

        model = open_clip.create_model(architecture, pretrained=None)
        try:
            weights = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older PyTorch versions
            weights = torch.load(path, map_location="cpu")
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        elif isinstance(weights, dict) and "model" in weights:
            weights = weights["model"]
        if not isinstance(weights, dict):
            raise TypeError("The checkpoint must contain a PyTorch state dict")
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)
        model.float().eval()
        for param in model.parameters():
            param.requires_grad_(False)

        if not hasattr(model.visual, "transformer"):
            raise RuntimeError("Only the ViT visual tower with a transformer is supported")
        self.clip_model = model
        self.tokenizer = open_clip.get_tokenizer(architecture)
        self.architecture = architecture
        self.embed_dim = int(model.visual.output_dim)
        self.image_size = 224

    def train(self, mode: bool = True):
        super().train(False)  # both pretrained towers stay in evaluation mode
        return self

    @torch.no_grad()
    def encode_texts(self, prompts: Sequence[str], batch_size: int = 64) -> Tensor:
        if not prompts:
            raise ValueError("At least one text prompt is required")
        device = next(self.clip_model.parameters()).device
        chunks: list[Tensor] = []
        for start in range(0, len(prompts), batch_size):
            tokens = self.tokenizer(list(prompts[start:start + batch_size])).to(device)
            encoded = self.clip_model.encode_text(tokens).float()
            chunks.append(F.normalize(encoded, dim=-1))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_visual(self, images: Tensor) -> tuple[Tensor, Tensor]:
        cache: dict[str, Tensor] = {}

        def capture(_module: nn.Module, _args: tuple, output: Tensor) -> None:
            cache["transformer"] = output[0] if isinstance(output, tuple) else output

        handle = self.clip_model.visual.transformer.register_forward_hook(capture)
        try:
            dtype = self.clip_model.visual.conv1.weight.dtype
            global_feature = self.clip_model.encode_image(images.to(dtype=dtype))
        finally:
            handle.remove()

        if "transformer" not in cache:
            raise RuntimeError("Could not capture ViT patch tokens from RemoteCLIP")
        x = cache["transformer"]
        batch_size = images.shape[0]
        if x.ndim != 3:
            raise RuntimeError(f"Expected 3-D transformer tokens; received {tuple(x.shape)}")
        if x.shape[0] == batch_size:
            x = x[:, 1:, :]  # batch-first ViT
        elif x.shape[1] == batch_size:
            x = x[1:, :, :].transpose(0, 1)  # sequence-first ViT
        else:
            raise RuntimeError(f"Unexpected transformer layout {tuple(x.shape)}")

        side = int(x.shape[1] ** 0.5)
        if side * side != x.shape[1]:
            raise RuntimeError("GSTC requires a square patch grid; check input size")

        visual = self.clip_model.visual
        x = visual.ln_post(x)
        projection = getattr(visual, "proj", None)
        if isinstance(projection, nn.Linear):
            x = projection(x)
        elif projection is not None:
            x = x @ projection
        if x.shape[-1] != self.embed_dim:
            raise RuntimeError("Visual patch and text dimensions do not match")
        return x.float(), F.normalize(global_feature.float(), dim=-1)
