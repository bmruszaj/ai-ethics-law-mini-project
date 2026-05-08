"""
Krzywe insertion / deletion względem dowolnej funkcji score(I) -> scalar.
Kolejność pikseli wg mapy saliencji (malejąco).
"""

from __future__ import annotations

import numpy as np

from xai_metrics import deletion_auc_from_scores, insertion_auc_from_scores


def _baseline_like(x: "torch.Tensor") -> "torch.Tensor":
    """Rozmycie boxowe jako tło pod insertion (start)."""
    import torch

    k = max(x.shape[-1] // 16, 3)
    if k % 2 == 0:
        k += 1
    # proste avg_pool jako „rozmyte” tło
    pad = k // 2
    y = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=k,
        stride=1,
        padding=0,
    )
    return y


def insertion_deletion_curves(
    image_1chw: "torch.Tensor",
    saliency_hw: np.ndarray | "torch.Tensor",
    forward_logits: callable,
    n_steps: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    image_1chw: (1,3,H,W), wymaga grad dla forwardu wewnętrznie jeśli model tego wymaga.
    saliency_hw: (H,W) — im wyższa wartość, tym ważniejszy piksel.
    Zwraca (ins_scores, del_scores) jako wektory długości n_steps.
    """
    import torch

    x0 = image_1chw.detach().clone()
    _, _, h, w = x0.shape
    s = np.asarray(saliency_hw, dtype=np.float64)
    s = s - s.min()
    s = s / (s.max() + 1e-8)
    order = np.argsort(-s.ravel())  # malejąco (ważne pierwsze)

    base = _baseline_like(x0)
    device = x0.device

    def score_tensor(xx: torch.Tensor) -> float:
        with torch.no_grad():
            return float(forward_logits(xx).squeeze().cpu())

    ins_scores = []
    del_scores = []
    total = h * w
    for t in range(n_steps):
        k = int((t + 1) / n_steps * total)
        k = max(1, min(k, total))
        mask_flat = np.zeros(total, dtype=np.float32)
        mask_flat[order[:k]] = 1.0
        mask = torch.from_numpy(mask_flat.reshape(h, w)).to(device).unsqueeze(0).unsqueeze(0)
        # insertion: od baseline do pełnego obrazu
        x_ins = base * (1.0 - mask) + x0 * mask
        ins_scores.append(score_tensor(x_ins))
        # deletion: od pełnego do usuwania najważniejszych
        mask_del = torch.from_numpy(mask_flat.reshape(h, w)).to(device).unsqueeze(0).unsqueeze(0)
        x_del = x0 * (1.0 - mask_del) + base * mask_del
        del_scores.append(score_tensor(x_del))

    return np.array(ins_scores, dtype=np.float64), np.array(del_scores, dtype=np.float64)


def insertion_deletion_aucs(
    image_1chw: "torch.Tensor",
    saliency_hw: np.ndarray | "torch.Tensor",
    forward_logits: callable,
    n_steps: int = 24,
) -> tuple[float, float]:
    ins, del_ = insertion_deletion_curves(image_1chw, saliency_hw, forward_logits, n_steps)
    return insertion_auc_from_scores(ins), deletion_auc_from_scores(del_)
