"""
RISE (Petsiuk et al.) — black-box saliencja przez losowe maski.
Uproszczenie: maski na przestrzeni pikseli (H×W), bez skalowania do wielu rozdzielczości.
"""

from __future__ import annotations

import torch


def rise_saliency_map(
    image_bchw: torch.Tensor,
    forward_logits: callable,
    n_masks: int = 64,
    p_keep: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """
    image_bchw: pojedynczy obraz (1, 3, H, W), wartości dowolne (model musi je akceptować).
    forward_logits: funkcja (x)-> logity (B, 1) wymagające gradientu tylko jeśli potrzeba.
    Zwraca mapę (H, W) — średnia ważona różnic względem średniej predykcji.
    """
    torch.manual_seed(seed)
    x = image_bchw.detach()
    device = x.device
    _, _, h, w = x.shape
    base = forward_logits(x).detach()

    acc = torch.zeros(h, w, device=device, dtype=torch.float32)
    for _ in range(n_masks):
        m = (torch.rand(1, 1, h, w, device=device) < p_keep).float()
        # unikaj całkowicie pustej maski
        if m.sum() < 1:
            m = torch.ones_like(m)
        x_m = x * m
        out = forward_logits(x_m).detach()
        diff = float((out - base).squeeze().abs().item())
        acc += diff * m.squeeze(0).squeeze(0)
    acc = acc / max(n_masks, 1)
    acc = acc - acc.min()
    acc = acc / (acc.max() + 1e-8)
    return acc
