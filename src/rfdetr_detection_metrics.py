"""
Metryki map saliencji względem bboxów GT: EBPG, mIoU (Otsu), Δp / Δp_rand.

Definicje (README + ``design.md``):
- **EBPG**: po normalizacji min-max mapy ``S`` na [0,1]: ``sum(S w B_gt) / sum(S na całym obrazie)``.
- **mIoU**: ``S`` dopasowane do rozmiaru obrazu ewaluacji, progowanie Otsu (OpenCV), IoU maski binarnej z prostokątem GT.
- **Δp**: różnica maksymalnego ``score`` dla klasy ``c`` przed i po inpaintingu obszaru GT;
  **Δp_rand**: to samo po zamalowaniu losowego prostokąta o **tej samej powierzchni** co GT (losowy środek, przycięcie do obrazu).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from xai_metrics import energy_fraction_inside_bbox, normalize_saliency_minmax


def ebpg(s: np.ndarray, bbox_xyxy: Tuple[float, float, float, float], eps: float = 1e-12) -> float:
    """Energy Based Pointing Game — ułamek „masy” mapy po min-max wewnątrz prostokąta GT."""
    return energy_fraction_inside_bbox(s, bbox_xyxy, eps=eps)


def miou_otsu_vs_bbox(s: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """S dopasowane do ROI obrazu przez resize; binaryzacja Otsu; IoU z maską prostokąta GT."""
    s = np.asarray(s, dtype=np.float64)
    h, w = s.shape
    sn = normalize_saliency_minmax(s)
    u8 = np.clip(sn * 255.0, 0, 255).astype(np.uint8)
    _, bin_s = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_s = (bin_s > 0).astype(np.uint8)

    gt = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(round(x1), 0, w - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    gt[y1 : y2 + 1, x1 : x2 + 1] = 1

    inter = np.logical_and(bin_s, gt).sum()
    union = np.logical_or(bin_s, gt).sum()
    return float(inter) / float(union + 1e-12)


def resize_s_to_hw(s: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize mapy saliencji do ``(height, width)`` (float32)."""
    return cv2.resize(np.asarray(s, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def inpaint_bbox_xyxy(rgb: np.ndarray, bbox_xyxy: Tuple[float, float, float, float], radius: int = 5) -> np.ndarray:
    """Inpainting Telea w prostokącie ``bbox_xyxy`` (obraz RGB uint8)."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(round(x1), 0, w - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)


def random_bbox_same_area(
    bbox_xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Losowy prostokąt o powierzchni jak GT; w całości w granicach obrazu."""
    x1, y1, x2, y2 = bbox_xyxy
    area = max(1.0, abs(x2 - x1) * abs(y2 - y1))
    aspect = float(rng.uniform(0.5, 2.0))
    bh = int(max(1, round(np.sqrt(area / aspect))))
    bw = int(max(1, round(area / bh)))
    bw = min(bw, width)
    bh = min(bh, height)
    cx = int(rng.integers(bw // 2, max(bw // 2 + 1, width - bw // 2)))
    cy = int(rng.integers(bh // 2, max(bh // 2 + 1, height - bh // 2)))
    hx, hy = bw // 2, bh // 2
    rx1 = float(np.clip(cx - hx, 0, width - 1))
    ry1 = float(np.clip(cy - hy, 0, height - 1))
    rx2 = float(np.clip(rx1 + bw - 1, 0, width - 1))
    ry2 = float(np.clip(ry1 + bh - 1, 0, height - 1))
    return rx1, ry1, rx2, ry2


def max_detection_score_for_class(rfdetr_api, pil_image: Image.Image, class_id: int, threshold: float) -> float:
    """Maksymalny ``confidence`` spośród detekcji danej klasy (indeks modelu)."""
    det = rfdetr_api.predict(pil_image.convert("RGB"), threshold=threshold)
    if det is None or len(det) == 0:
        return 0.0
    mask = det.class_id == int(class_id)
    if not np.any(mask):
        return 0.0
    return float(np.max(det.confidence[mask]))


def delta_p_scores(
    rfdetr_api,
    pil_image: Image.Image,
    bbox_xyxy_orig: Tuple[float, float, float, float],
    class_id_model: int,
    threshold: float,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """
    Zwraca ``(p_orig, delta_p_gt_inpaint, delta_p_random)``.
    Δp = p_orig - p_po_zaburzeniu (inpainting GT lub losowy box).
    """
    rng = rng or np.random.default_rng()
    rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    p_orig = max_detection_score_for_class(rfdetr_api, pil_image, class_id_model, threshold)

    rgb_gt = inpaint_bbox_xyxy(rgb, bbox_xyxy_orig)
    p_gt = max_detection_score_for_class(rfdetr_api, Image.fromarray(rgb_gt), class_id_model, threshold)

    rb = random_bbox_same_area(bbox_xyxy_orig, w, h, rng)
    rgb_rand = inpaint_bbox_xyxy(rgb, rb)
    p_rand = max_detection_score_for_class(rfdetr_api, Image.fromarray(rgb_rand), class_id_model, threshold)

    return p_orig, float(p_orig - p_gt), float(p_orig - p_rand)
