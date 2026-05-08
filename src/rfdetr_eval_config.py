"""
Konfiguracja progów dopasowania pred–GT (jeden zestaw hiperparametrów na eksperyment).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DetectionEvalConfig:
    """Parametry ewaluacji detektora i metryk XAI opartych na parach pred–GT."""

    iou_match_threshold: float = 0.5
    confidence_threshold: float = 0.25
    # Nazwa pliku / uruchomienia (metadane w CSV i JSON).
    run_label: str = "default"

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> DetectionEvalConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def coco_bbox_xywh_to_xyxy(bbox_xywh: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return float(x), float(y), float(x + w), float(y + h)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match_single_class(
    pred_boxes_xyxy: list[tuple[float, float, float, float]],
    pred_scores: list[float],
    gt_boxes_xyxy: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """
    Zwraca listę dopasowań (pred_idx, gt_idx, iou) dla jednej klasy w jednym obrazie.
    Greedy: predykcje posortowane malejąco po ``score``.
    """
    order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
    gt_matched = [False] * len(gt_boxes_xyxy)
    matches: list[tuple[int, int, float]] = []
    for pi in order:
        best_j = -1
        best_iou = 0.0
        for gj in range(len(gt_boxes_xyxy)):
            if gt_matched[gj]:
                continue
            iou = iou_xyxy(pred_boxes_xyxy[pi], gt_boxes_xyxy[gj])
            if iou > best_iou:
                best_iou = iou
                best_j = gj
        if best_j >= 0 and best_iou >= iou_threshold:
            gt_matched[best_j] = True
            matches.append((pi, best_j, best_iou))
    return matches
