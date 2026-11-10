from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Which entries of the 6-value descriptor [mu_u, mu_v, area, rho, sin, cos]
# are kept for each geometry setting.
GEOMETRY_MASKS = {
    "none": (0, 0, 0, 0, 0, 0),
    "centroid": (1, 1, 0, 0, 0, 0),
    "centroid_area": (1, 1, 1, 0, 0, 0),
    "centroid_area_aspect": (1, 1, 1, 1, 0, 0),
    "full": (1, 1, 1, 1, 1, 1),
}


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class SSPB(nn.Module):
    """Fuses the identity / object / geometry prompt embeddings of each class."""

    def __init__(self, prompt_embeddings: Tensor, hidden_dim: int = 256, fusion: str = "gated"):
        super().__init__()
        if prompt_embeddings.dim() != 3:
            raise ValueError("prompt_embeddings must be [classes, prompts, dim]")
        if fusion not in ("gated", "equal"):
            raise ValueError(f"unknown fusion {fusion!r}")
        num_prompts, dim = prompt_embeddings.shape[1:]
        self.register_buffer("embeddings", F.normalize(prompt_embeddings.float(), dim=-1))
        self.fusion = fusion if num_prompts > 1 else "equal"
        self.gate = mlp(num_prompts * dim, hidden_dim, num_prompts) if self.fusion == "gated" else None

    @property
    def num_classes(self) -> int:
        return self.embeddings.shape[0]

    def forward(self) -> tuple[Tensor, Tensor]:
        e = self.embeddings                                   # [C, M, d]
        if self.gate is None:
            alpha = e.new_full(e.shape[:2], 1.0 / e.shape[1])
        else:
            alpha = self.gate(e.flatten(1)).softmax(dim=-1)   # [C, M]
        prototypes = F.normalize((alpha.unsqueeze(-1) * e).sum(dim=1), dim=-1)
        return prototypes, alpha


class GSTC(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 6, assign_dim: int = 128,
                 geo_hidden: int = 128, geometry: str = "full",
                 inject_geometry: bool = True, max_aspect: float = 10.0, eps: float = 1e-6):
        super().__init__()
        if geometry not in GEOMETRY_MASKS:
            raise ValueError(f"unknown geometry setting {geometry!r}")
        self.num_tokens = num_tokens
        self.assign_dim = assign_dim
        self.inject_geometry = inject_geometry and geometry != "none"
        self.max_aspect = max_aspect
        self.eps = eps

        self.scene_queries = nn.Parameter(torch.randn(num_tokens, dim) * dim ** -0.5)
        self.w_q = nn.Linear(dim, assign_dim, bias=False)
        self.w_k = nn.Linear(dim, assign_dim, bias=False)
        self.geo_proj = mlp(6, geo_hidden, dim) if self.inject_geometry else None
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("geo_mask", torch.tensor(GEOMETRY_MASKS[geometry], dtype=torch.float))

    @staticmethod
    def patch_coords(num_patches: int, device, dtype) -> Tensor:
        """Normalised (u, v) centres of a square patch grid, shape [P, 2]."""
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(f"{num_patches} patches do not form a square grid")
        axis = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
        v, u = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((u.flatten(), v.flatten()), dim=-1)

    def geometry(self, assign: Tensor, coords: Tensor) -> Tensor:
        """Centroid, area, aspect ratio and orientation of each soft group.
        assign: [B, P, S], coords: [P, 2]. Returns [B, S, 6].
        """
        eps = self.eps
        mass = assign.sum(dim=1)                                        # [B, S]
        area = mass / assign.shape[1]
        w = assign / (mass.unsqueeze(1) + eps)                          # weights per group
        centroid = torch.einsum("bps,pi->bsi", w, coords)               # [B, S, 2]

        diff = coords[None, None] - centroid[:, :, None]                # [B, S, P, 2]
        cov = torch.einsum("bps,bspi,bspj->bsij", w, diff, diff)
        a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]

        # Closed-form eigen-decomposition of the 2x2 covariance.
        gap = torch.sqrt((a - c) ** 2 + 4 * b ** 2 + eps ** 2)
        lam_major = 0.5 * (a + c + gap)
        lam_minor = (0.5 * (a + c - gap)).clamp_min(eps)
        aspect = torch.sqrt(lam_major / lam_minor).clamp(max=self.max_aspect)

        # Orientation of the major axis, theta in (-pi/2, pi/2].
        cos2 = (a - c) / gap
        sin2 = 2 * b / gap
        cos_t = torch.sqrt(((1 + cos2) / 2).clamp_min(eps))
        sin_t = torch.sqrt(((1 - cos2) / 2).clamp_min(eps))
        sin_t = torch.where(sin2 < 0, -sin_t, sin_t)

        return torch.cat((centroid, area.unsqueeze(-1), aspect.unsqueeze(-1),
                          sin_t.unsqueeze(-1), cos_t.unsqueeze(-1)), dim=-1)

    def forward(self, patches: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """patches: [B, P, d] -> scene tokens [B, S, d], geometry [B, S, 6], assignment [B, P, S]."""
        q = self.w_q(patches)
        k = self.w_k(self.scene_queries)
        assign = (q @ k.t() / math.sqrt(self.assign_dim)).softmax(dim=-1)

        mass = assign.sum(dim=1)
        tokens = assign.transpose(1, 2) @ patches / (mass.unsqueeze(-1) + self.eps)

        coords = self.patch_coords(patches.shape[1], patches.device, patches.dtype)
        geo = self.geometry(assign, coords) * self.geo_mask

        if self.geo_proj is not None:
            tokens = tokens + self.geo_proj(geo)
        return self.norm(tokens), geo, assign


def relative_geometry(geo: Tensor) -> Tensor:
    """Pairwise descriptor [d_mu(2), dist, d_area, d_aspect, sin d_theta, cos d_theta], [B, S, S, 7]."""
    mu, area, aspect = geo[..., :2], geo[..., 2], geo[..., 3]
    sin_t, cos_t = geo[..., 4], geo[..., 5]
    d_mu = mu[:, :, None] - mu[:, None, :]
    dist = torch.sqrt((d_mu ** 2).sum(-1, keepdim=True) + 1e-8)
    d_area = area[:, :, None] - area[:, None, :]
    d_aspect = aspect[:, :, None] - aspect[:, None, :]
    # sin(a - b) and cos(a - b) from the per-token sin / cos.
    sin_d = sin_t[:, :, None] * cos_t[:, None, :] - cos_t[:, :, None] * sin_t[:, None, :]
    cos_d = cos_t[:, :, None] * cos_t[:, None, :] + sin_t[:, :, None] * sin_t[:, None, :]
    return torch.cat((d_mu, dist, d_area.unsqueeze(-1), d_aspect.unsqueeze(-1),
                      sin_d.unsqueeze(-1), cos_d.unsqueeze(-1)), dim=-1)


class SGRM(nn.Module):
    def __init__(self, dim: int, rel_hidden: int = 64, ffn_ratio: int = 4,
                 temperature: float = 0.07, geometry_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.bias_mlp = mlp(7, rel_hidden, 1) if geometry_bias else None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim), nn.Dropout(dropout),
        )

    def relational(self, z: Tensor, geo: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.q(z) @ self.k(z).transpose(1, 2) / math.sqrt(self.dim)
        if self.bias_mlp is not None:
            logits = logits + self.bias_mlp(relative_geometry(geo)).squeeze(-1)
        attn = logits.softmax(dim=-1)
        h = self.norm1(z + attn @ self.v(z))
        return self.norm2(z + self.ffn(h)), attn

    def semantic_matching(self, z: Tensor, prototypes: Tensor) -> tuple[Tensor, Tensor]:
        protos = F.normalize(prototypes, dim=-1)
        affinity = F.normalize(z, dim=-1) @ protos.t()                
        eta = (affinity / self.temperature).softmax(dim=1)              
        evidence = torch.einsum("bsc,bsd->bcd", eta, z)                
        scores = (F.normalize(evidence, dim=-1) * protos).sum(-1)       
        return scores, eta

    def forward(self, z: Tensor, geo: Tensor, prototypes: Tensor) -> dict[str, Tensor]:
        refined, attn = self.relational(z, geo)
        scores, eta = self.semantic_matching(refined, prototypes)
        return {"refined": refined, "semantic": scores, "token_weights": eta, "token_attention": attn}


class SAGENet(nn.Module):
    def __init__(self, encoder: nn.Module, prompt_embeddings: Tensor, *,
                 num_tokens: int = 6, assign_dim: int = 128, gate_hidden: int = 256,
                 geo_hidden: int = 128, rel_hidden: int = 64, temperature: float = 0.07,
                 prompt_fusion: str = "gated", geometry: str = "full",
                 inject_geometry: bool = True, geometry_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        dim = prompt_embeddings.shape[-1]
        num_classes = prompt_embeddings.shape[0]

        # Kept outside the module tree so it is not saved with the adapter
        # weights and is not touched by .train().
        self.__dict__["encoder"] = encoder

        self.sspb = SSPB(prompt_embeddings, gate_hidden, prompt_fusion)
        self.gstc = GSTC(
            dim, num_tokens, assign_dim, geo_hidden, geometry, inject_geometry)
        self.sgrm = SGRM(
            dim, rel_hidden, temperature=temperature, geometry_bias=geometry_bias, dropout=dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def forward_features(self, patches: Tensor) -> dict[str, Tensor]:
        prototypes, alpha = self.sspb()
        z, geo, assign = self.gstc(patches)
        out = self.sgrm(z, geo, prototypes)
        pooled = out["refined"].mean(dim=1)
        visual = self.classifier(pooled)
        out.update(
            logits=visual + out["semantic"],
            visual=visual,
            pooled=pooled,
            prototypes=prototypes,
            prompt_weights=alpha,
            scene_tokens=z,
            geometry=geo,
            assignment=assign,
        )
        return out

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        patches, _ = self.encoder.encode_image(images)
        return self.forward_features(patches)


def sage_loss(out: dict[str, Tensor], labels: Tensor, lambda_align: float = 0.5,
              temperature: float = 0.07) -> dict[str, Tensor]:
    ce = F.cross_entropy(out["logits"], labels)
    align = F.cross_entropy(out["semantic"] / temperature, labels)
    return {"loss": ce + lambda_align * align, "ce": ce, "align": align}


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

