"""
Minimalny ViT (1 blok) do proof-of-concept metod opartych o attention (rollout w stylu Chefer).
Wyłącznie do dem w repozytorium — nie zastępuje encoderów CLIP/SigLIP z produkcji.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyViTForXaiDemo(nn.Module):
    """ViT: patch embedding + 1 warstwa MHA + głowa binarna na tokenie CLS."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        dim: int = 96,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size musi być podzielne przez patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, dim))

        self.norm_pre = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_post = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.head = nn.Linear(dim, 1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca (logity B×1, wagi uwagi uśrednione po głowach B×(1+P)×(1+P)).
        """
        b = x.shape[0]
        t = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, P, dim
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls, t), dim=1) + self.pos_embed
        h = self.norm_pre(tokens)
        h2, attn_w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        h = tokens + h2
        h = h + self.mlp(self.norm_post(h))
        logits = self.head(h[:, 0, :])
        return logits, attn_w

    @staticmethod
    def attention_rollout_map(attn: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """
        attn: (B, 1+P, 1+P) — uśrednione po głowach (``average_attn_weights=True``).
        Po jednej warstwie przyjmujemy mapę Chefer-style jako uwagę CLS → patche.
        """
        b = attn.shape[0]
        rel = attn[:, 0, 1:]  # B, P
        gh, gw = grid_hw
        return rel.reshape(b, gh, gw)


def init_demo_weights(m: TinyViTForXaiDemo) -> None:
    """Deterministyczna inicjalizacja do powtarzalnych dem."""
    gen = torch.Generator(device=next(m.parameters()).device)
    gen.manual_seed(0)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)
        else:
            nn.init.zeros_(p)
