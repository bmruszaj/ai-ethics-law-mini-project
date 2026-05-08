"""
``DetectionScoreTarget`` dla biblioteki ``grad-cam`` (pakiet PyPI: ``grad-cam``):
wybór zapytania (query) o największym IoU z ``matched_box`` w przestrzeni wejścia modelu,
cel gradientu: ``sigmoid(logit)[class_id]``.
"""

from __future__ import annotations

from typing import Any

import torch

from rfdetr.utilities import box_ops


def _iou_xyxy_torch(pred_xyxy: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor:
    """pred_xyxy [Q,4], box_xyxy [4] — zwraca IoU dla każdego wiersza."""
    ax1, ay1, ax2, ay2 = pred_xyxy.unbind(dim=-1)
    bx1, by1, bx2, by2 = box_xyxy.unbind(dim=-1)
    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih
    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


class DetectionScoreTarget:
    """
    Parameters
    ----------
    matched_box_xyxy_orig:
        Bbox dopasowany do GT w układzie **oryginalnego** obrazu (piksele), format xyxy.
    class_id_model:
        Indeks klasy 0..C-1 zgodny z wyjściem modelu (jak ``supervision.Detections.class_id``).
    orig_hw:
        (wysokość, szerokość) obrazu źródłowego przed resize do kwadratu modelu.
    input_hw:
        (H, W) tensora wejściowego przekazanego do ``forward`` CAM (po ``F.resize``).
    """

    def __init__(
        self,
        matched_box_xyxy_orig: tuple[float, float, float, float],
        class_id_model: int,
        orig_hw: tuple[int, int],
        input_hw: tuple[int, int],
    ) -> None:
        self.class_id_model = int(class_id_model)
        oh, ow = orig_hw
        ih, iw = input_hw
        sx = iw / max(float(ow), 1e-6)
        sy = ih / max(float(oh), 1e-6)
        x1, y1, x2, y2 = matched_box_xyxy_orig
        self.box_xyxy_input = (
            float(x1 * sx),
            float(y1 * sy),
            float(x2 * sx),
            float(y2 * sy),
        )
        self.input_hw = (int(ih), int(iw))

    def __call__(self, model_output: dict[str, Any]) -> torch.Tensor:
        if not isinstance(model_output, dict):
            raise TypeError(f"Oczekiwano dict wyjścia LWDETR, otrzymano {type(model_output)}")
        logits = model_output["pred_logits"]
        boxes = model_output["pred_boxes"]
        if logits.ndim != 3 or boxes.ndim != 3:
            raise ValueError(
                f"Nieoczekiwany kształt wyjścia: logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}"
            )

        logits_b0 = logits[0]
        boxes_b0 = boxes[0]
        xyxy_norm = box_ops.box_cxcywh_to_xyxy(boxes_b0)
        ih, iw = self.input_hw
        scale = xyxy_norm.new_tensor([iw, ih, iw, ih])
        pred_xyxy = xyxy_norm * scale
        box_t = pred_xyxy.new_tensor(self.box_xyxy_input)
        ious = _iou_xyxy_torch(pred_xyxy, box_t)
        q = int(torch.argmax(ious).item())
        logit_q = logits_b0[q, self.class_id_model]
        return logit_q.sigmoid()
