"""
Metryki ewaluacji map wyjaśnialności względem bboxów (numpy).
Opis metryk: docs/context/EWALUACJA.md

Konwencje:
- Mapa saliencji S: 2D ``numpy.ndarray`` typu float, kształt (H, W).
- Bbox w układzie pikseli: ``(x1, y1, x2, y2)`` włącznie z krańcami (clip do [0, W-1] / [0, H-1]).
- Normalizacja mapy przed sumowaniem „energii”: min-max na [0, 1], potem suma = 1 (rozkład dyskretny).
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def normalize_saliency_minmax(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max na [0, 1]; stała mapa → zera."""
    s = np.asarray(s, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < eps:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo + eps)


def normalize_saliency_to_distribution(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """S normalizowane do rozkładu dyskretnego (suma = 1)."""
    s = normalize_saliency_minmax(s)
    total = float(s.sum())
    if total < eps:
        return np.ones_like(s) / s.size
    return s / total


def _clip_box_xyxy(
    bbox_xyxy: tuple[float, float, float, float], height: int, width: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(round(x1), 0, width - 1))
    x2 = int(np.clip(round(x2), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    y2 = int(np.clip(round(y2), 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def pointing_game_hit(s: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> bool:
    """Czy piksel o maksymalnej saliencji (oryginalnej, przed normalizacją) leży w bbox?"""
    s = np.asarray(s, dtype=np.float64)
    h, w = s.shape
    y, x = divmod(int(np.argmax(s)), w)
    x1, y1, x2, y2 = _clip_box_xyxy(bbox_xyxy, h, w)
    return x1 <= x <= x2 and y1 <= y <= y2


def energy_fraction_inside_bbox(
    s: np.ndarray, bbox_xyxy: tuple[float, float, float, float], eps: float = 1e-12
) -> float:
    """
    Ułamek całkowitej energii po normalizacji min-max, który wpada do bbox
    (suma S_norm w prostokącie / suma S_norm globalnie).
    """
    s = np.asarray(s, dtype=np.float64)
    h, w = s.shape
    sn = normalize_saliency_minmax(s)
    total = float(sn.sum()) + eps
    x1, y1, x2, y2 = _clip_box_xyxy(bbox_xyxy, h, w)
    inside = float(sn[y1 : y2 + 1, x1 : x2 + 1].sum())
    return inside / total


def cam_quality_status(pointing_game: bool, energy_fraction: float) -> Literal["good", "weak", "failed"]:
    """
    Heurystyczna etykieta zgodności mapy saliencji z bbox (do raportów; nie jest ground truth).

    - ``good``: trafienie pointing game i ułamek energii w bbox ≥ 0.10
    - ``weak``: energia w bbox ≥ 0.03
    - ``failed``: poniżej progów weak
    """
    if pointing_game and energy_fraction >= 0.10:
        return "good"
    if energy_fraction >= 0.03:
        return "weak"
    return "failed"


def energy_enrichment_inside_bbox(
    s: np.ndarray, bbox_xyxy: tuple[float, float, float, float], eps: float = 1e-12
) -> float:
    """
    Wzbogacenie energii: energy_fraction / box_area_fraction.

    Interpretacja:
        ≈ 1.0  →  mapa jak losowa względem boxa (energia proporcjonalna do pola)
        > 1.0  →  mapa preferuje obszar boxa
        < 1.0  →  mapa omija obszar boxa

    Normalizacja przez pole boxa eliminuje faworyzowanie dużych boxa przez samo EF.
    """
    s = np.asarray(s, dtype=np.float64)
    h, w = s.shape
    x1, y1, x2, y2 = _clip_box_xyxy(bbox_xyxy, h, w)
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box_area_fraction = box_area / max(h * w, 1)
    if box_area_fraction < eps:
        return 0.0
    ef = energy_fraction_inside_bbox(s, bbox_xyxy, eps=eps)
    return ef / box_area_fraction


def energy_iou_distribution_vs_mask(
    s: np.ndarray, bbox_xyxy: tuple[float, float, float, float], eps: float = 1e-12
) -> float:
    """
    Wariant „IoU energii”: traktujemy znormalizowaną mapę jako masę P na siatce,
    maskę M = 1 w bbox. IoU = sum(min(P, M̂)) / (sum P + sum M̂ - sum(min(...))),
    gdzie M̂ = M / |M| (jednostkowa masa w bbox).
    """
    p = normalize_saliency_to_distribution(s)
    h, w = p.shape
    x1, y1, x2, y2 = _clip_box_xyxy(bbox_xyxy, h, w)
    m = np.zeros_like(p)
    m[y1 : y2 + 1, x1 : x2 + 1] = 1.0
    m_sum = m.sum()
    if m_sum < eps:
        return 0.0
    m_hat = m / m_sum
    inter = float(np.minimum(p, m_hat).sum())
    union = float(p.sum() + m_hat.sum() - inter) + eps
    return inter / union


def insertion_auc_from_scores(scores: np.ndarray) -> float:
    """
    AUC po krzywej insertion: ``scores[i]`` = pewność / logit po dodaniu i-tego
    frakcji pikseli (monotonicznie rosnąco po ważności). Oś x: równomiernie [0,1].
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size < 2:
        return float(scores[0]) if scores.size else 0.0
    x = np.linspace(0.0, 1.0, scores.size)
    return float(np.trapezoid(scores, x))


def deletion_auc_from_scores(scores: np.ndarray) -> float:
    """Analogicznie dla deletion (malejąco po usuwaniu ważnych pikseli); AUC na [0,1]."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size < 2:
        return float(scores[0]) if scores.size else 0.0
    x = np.linspace(0.0, 1.0, scores.size)
    return float(np.trapezoid(scores, x))


def proxy_insertion_deletion_scores_from_saliency(
    s: np.ndarray, n_steps: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """
    Krzywe insertion/deletion **bez forwardu sieci**: score = ułamek „energii” rozkładu P
    (po ``normalize_saliency_to_distribution``) pozostającej w widocznych pikselach.

    - **Insertion:** od maski pustej dodajemy kolejno piksele o malejącej saliencji (najważniejsze
      najpierw); score rośnie z masą P w odkrytym zbiorze.
    - **Deletion:** od pełnego obrazu usuwamy (zerujemy) najważniejsze piksele pierwsze; score maleje.

    Służy do dem i testów metryki AUC; pełna wierność wymaga pętli z ``forward`` modelu
    (patrz ``src/xai_ins_del.py``).
    """
    p = normalize_saliency_to_distribution(s)
    flat = p.ravel()
    order = np.argsort(-flat)
    n = flat.size
    ins: list[float] = []
    dels: list[float] = []
    total = float(flat.sum()) + 1e-12
    for t in range(n_steps):
        k = max(1, int((t + 1) / n_steps * n))
        idx_keep = order[:k]
        mask_i = np.zeros(n, dtype=np.float64)
        mask_i[idx_keep] = 1.0
        ins.append(float((flat * mask_i).sum() / total))
        mask_d = np.ones(n, dtype=np.float64)
        mask_d[order[:k]] = 0.0
        dels.append(float((flat * mask_d).sum() / total))
    return np.array(ins, dtype=np.float64), np.array(dels, dtype=np.float64)
