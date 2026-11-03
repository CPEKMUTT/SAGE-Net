from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StructuredSemanticPromptBank(nn.Module):

    def __init__(self, prompt_embeddings: Tensor, hidden_dim: int = 256):
        super().__init__()
        if prompt_embeddings.ndim != 3 or prompt_embeddings.shape[1] != 3:
            raise ValueError("Prompt embeddings must have shape [classes,3,dim]")
        classes, _types, dim = prompt_embeddings.shape
        if classes < 2:
            raise ValueError("At least two scene classes are required")
        self.register_buffer("embeddings", F.normalize(prompt_embeddings.float(), dim=-1))
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, identity_only: bool = False,
                fusion: str = "gated") -> tuple[Tensor, Tensor]:
        # Table 10: allow controlled tests of prompt subsets and equal fusion.
        if identity_only:
            fusion = "identity"
        masks = {
            "identity": (True, False, False),
            "identity_object": (True, True, False),
            "identity_geometry": (True, False, True),
            "equal": (True, True, True),
            "gated": (True, True, True),
        }
        if fusion not in masks:
            raise ValueError(f"Unknown prompt fusion mode: {fusion!r}")
        mask = torch.tensor(masks[fusion], device=self.embeddings.device,
                            dtype=torch.bool)
        if fusion in ("identity", "equal"):
            weights = mask.to(self.embeddings.dtype).expand(
                self.embeddings.shape[0], -1
            )
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            logits = self.gate(self.embeddings.flatten(1))
            weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1)
        prototypes = (weights.unsqueeze(-1) * self.embeddings).sum(dim=1)
        return F.normalize(prototypes, dim=-1), weights


class GeometryAwareSceneTokenConstructor(nn.Module):
    """GSTC; differentiable soft grouping and geometry"""

    def __init__(self, dim: int, scene_tokens: int = 6, assign_dim: int = 128,
                 geometry_hidden: int = 128, eps: float = 1e-6):
        super().__init__()
        if scene_tokens < 1 or assign_dim < 1:
            raise ValueError("scene_tokens and assign_dim must be positive")
        self.scene_tokens = scene_tokens
        self.eps = eps
        self.queries = nn.Parameter(torch.randn(scene_tokens, dim) * 0.02)
        self.patch_query = nn.Linear(dim, assign_dim, bias=False)
        self.group_key = nn.Linear(dim, assign_dim, bias=False)
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, geometry_hidden), nn.GELU(),
            nn.Linear(geometry_hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def normalized_coordinates(patches: int, *, device: torch.device,
                               dtype: torch.dtype) -> Tensor:
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError(f"Patch count {patches} is not a square grid")
        axis = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx.flatten(), yy.flatten()), dim=-1)

    def forward(self, patch_tokens: Tensor, *, use_geometry: bool = True,
                geometry_mode: str = "full"
                ) -> tuple[Tensor, Tensor, Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("Expected patch tokens [batch,patches,dim]")
        batch, patches, _dim = patch_tokens.shape
        coords = self.normalized_coordinates(
            patches, device=patch_tokens.device, dtype=patch_tokens.dtype
        )

        # A[b,p,s], normalized across scene groups S (equation 12).
        q = self.patch_query(patch_tokens)
        k = self.group_key(self.queries)
        assignments = (q @ k.t() / math.sqrt(q.shape[-1])).softmax(dim=-1)
        weights = assignments.transpose(1, 2)  # [B,S,P]
        masses = weights.sum(dim=-1).clamp_min(self.eps)  # [B,S]

        appearance = weights @ patch_tokens / masses.unsqueeze(-1)
        centroid = (weights @ coords) / masses.unsqueeze(-1)  # [B,S,2]
        area = assignments.mean(dim=1)  # [B,S], equation (14)

        # Weighted 2-D covariance; its principal axis is defined modulo pi.
        centered = coords[None, None, :, :] - centroid[:, :, None, :]
        covariance = torch.einsum(
            "bsp,bspi,bspj->bsij", weights, centered, centered
        ) / masses[:, :, None, None]
        xx, xy, yy = covariance[..., 0, 0], covariance[..., 0, 1], covariance[..., 1, 1]
        trace = xx + yy
        gap = torch.sqrt((xx - yy).square() + 4.0 * xy.square() + self.eps**2)
        major = ((trace + gap) / 2.0).clamp_min(self.eps)
        minor = ((trace - gap) / 2.0).clamp_min(self.eps)
        aspect_ratio = torch.sqrt(major / minor).clamp(max=100.0)
        # Stable canonical orientation (-pi/2, pi/2], avoiding eigenvector sign flips.
        theta = 0.5 * torch.atan2(2.0 * xy, xx - yy)
        geometry = torch.cat(
            (centroid, area.unsqueeze(-1), aspect_ratio.unsqueeze(-1),
             torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)),
            dim=-1,
        )  # [B,S,6]: 2 centroid, 1 area, 1 aspect, 2 orientation

        # Table 9: incremental geometry descriptor ablations. Mask *both* the
        # token projection and pairwise relations to avoid leaked descriptors.
        masks = {
            "none": (0, 0, 0, 0, 0, 0),
            "centroid": (1, 1, 0, 0, 0, 0),
            "centroid_area": (1, 1, 1, 0, 0, 0),
            "centroid_area_aspect": (1, 1, 1, 1, 0, 0),
            "full": (1, 1, 1, 1, 1, 1),
        }
        if geometry_mode not in masks:
            raise ValueError(f"Unknown geometry mode: {geometry_mode!r}")
        geometry = geometry * geometry.new_tensor(masks[geometry_mode])

        if use_geometry and geometry_mode != "none":
            scene = self.norm(appearance + self.geometry_projection(geometry))
        else:
            scene = self.norm(appearance)
        assert scene.shape == (batch, self.scene_tokens, patch_tokens.shape[-1])
        return scene, geometry, assignments


class SemanticGeometryReasoningModule(nn.Module):
    
    def __init__(self, dim: int, relation_hidden: int = 128,
                 ffn_ratio: int = 4):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        # Δcentroid(2), distance(1), Δarea(1), Δaspect(1), sin/cosΔθ(2).
        self.geometry_bias = nn.Sequential(
            nn.Linear(7, relation_hidden), nn.GELU(),
            nn.Linear(relation_hidden, 1),
        )
        self.inner_norm = nn.LayerNorm(dim)
        self.outer_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio), nn.GELU(),
            nn.Linear(dim * ffn_ratio, dim),
        )
        self.dim = dim

    def relative_geometry(self, geometry: Tensor) -> Tensor:
        mu, area, aspect = geometry[..., :2], geometry[..., 2], geometry[..., 3]
        theta = torch.atan2(geometry[..., 4], geometry[..., 5])
        dmu = mu[:, :, None, :] - mu[:, None, :, :]
        dtheta = theta[:, :, None] - theta[:, None, :]
        return torch.cat((
            dmu,
            dmu.norm(dim=-1, keepdim=True),
            (area[:, :, None] - area[:, None, :]).unsqueeze(-1),
            (aspect[:, :, None] - aspect[:, None, :]).unsqueeze(-1),
            torch.sin(dtheta).unsqueeze(-1),
            torch.cos(dtheta).unsqueeze(-1),
        ), dim=-1)

    def forward(self, scene: Tensor, geometry: Tensor, prototypes: Tensor,
                *, temperature: float = 0.07, use_geometry_bias: bool = True
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        attention_logits = self.q(scene) @ self.k(scene).transpose(1, 2)
        attention_logits = attention_logits / math.sqrt(self.dim)
        if use_geometry_bias:
            bias = self.geometry_bias(self.relative_geometry(geometry)).squeeze(-1)
            attention_logits = attention_logits + bias
        attention = attention_logits.softmax(dim=-1) @ self.v(scene)

        # Equation (22): residual attention, FFN, then final LayerNorm.
        hidden = self.inner_norm(scene + attention)
        refined = self.outer_norm(scene + self.ffn(hidden))

        similarities = F.normalize(refined, dim=-1) @ F.normalize(prototypes, dim=-1).t()
        token_weights = (similarities / temperature).softmax(dim=1)  # [B,S,C]
        class_evidence = torch.einsum("bsc,bsd->bcd", token_weights, refined)
        semantic_scores = (
            F.normalize(class_evidence, dim=-1) *
            F.normalize(prototypes, dim=-1).unsqueeze(0)
        ).sum(dim=-1)  # [B,C], equation (26)
        return refined, semantic_scores, token_weights, class_evidence


class SAGENet(nn.Module):

    def __init__(self, backbone: nn.Module, prompt_embeddings: Tensor,
                 *, scene_tokens: int = 6, assignment_dim: int = 128,
                 prompt_hidden: int = 256, relation_hidden: int = 128,
                 geometry_hidden: int = 128, temperature: float = 0.07,
                 identity_only: bool = False, use_geometry: bool = True,
                 use_geometry_bias: bool = True,
                 prompt_fusion: str = "gated", geometry_mode: str = "full"):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        dim = int(backbone.embed_dim)
        if prompt_embeddings.shape[-1] != dim:
            raise ValueError("RemoteCLIP visual/text embedding dimensions must match")
        self.prompt_bank = StructuredSemanticPromptBank(prompt_embeddings, prompt_hidden)
        self.gstc = GeometryAwareSceneTokenConstructor(
            dim, scene_tokens, assignment_dim, geometry_hidden
        )
        self.sgrm = SemanticGeometryReasoningModule(dim, relation_hidden)
        self.classifier = nn.Linear(dim, prompt_embeddings.shape[0])
        self.temperature = temperature
        self.identity_only = identity_only
        self.use_geometry = use_geometry
        self.use_geometry_bias = use_geometry_bias
        self.prompt_fusion = prompt_fusion
        self.geometry_mode = geometry_mode

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # crucial: a frozen backbone must not enable dropout
        return self

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        patch_features, global_features = self.backbone.encode_visual(images)
        prototypes, prompt_gates = self.prompt_bank(
            identity_only=self.identity_only, fusion=self.prompt_fusion
        )
        scene, geometry, assignments = self.gstc(
            patch_features, use_geometry=self.use_geometry,
            geometry_mode=self.geometry_mode if self.use_geometry else "none",
        )
        refined, semantic, token_weights, class_evidence = self.sgrm(
            scene, geometry, prototypes, temperature=self.temperature,
            use_geometry_bias=self.use_geometry_bias,
        )
        visual_logits = self.classifier(refined.mean(dim=1))
        logits = visual_logits + semantic
        return {
            "logits": logits,
            "semantic_scores": semantic,
            "visual_logits": visual_logits,
            "prompt_gates": prompt_gates,
            "prototypes": prototypes,
            "scene_tokens": refined,
            "geometry": geometry,
            "assignments": assignments,
            "token_attention": token_weights,
            "class_evidence": class_evidence,
            "global_features": global_features,
        }


def sage_loss(outputs: dict[str, Tensor], labels: Tensor, *,
              alignment_weight: float = 0.5, temperature: float = 0.07
              ) -> dict[str, Tensor]:
    """Eq. (30)--(32): L_total = CE(logits,y) + lambda CE(s_sem/tau,y)."""
    if temperature <= 0 or alignment_weight < 0:
        raise ValueError("Invalid temperature or alignment weight")
    cross_entropy = F.cross_entropy(outputs["logits"], labels)
    alignment = F.cross_entropy(outputs["semantic_scores"] / temperature, labels)
    return {
        "cross_entropy": cross_entropy,
        "alignment": alignment,
        "total": cross_entropy + alignment_weight * alignment,
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
