"""
Wrapper RF-DETR pod interfejs ``GeneralObjectDetectionModelWrapper`` z pakietu ``vision-explanation-methods`` (D-RISE).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from vision_explanation_methods.explanations.common import DetectionRecord, expand_class_scores


class RFDETRDriseWrapper:
    """``predict(x) -> List[DetectionRecord]`` dla wsadu obrazów ``[B,3,H,W]`` w skali [0,1]."""

    def __init__(self, rfdetr_api: Any, num_classes: int, score_threshold: float = 0.25) -> None:
        self._api = rfdetr_api
        self._num_classes = int(num_classes)
        if self._num_classes < 2:
            raise ValueError("``expand_class_scores`` wymaga co najmniej 2 klas (D-RISE / RF-DETR).")
        self._score_threshold = float(score_threshold)
        self._device = rfdetr_api.model.device

    def predict(self, x: torch.Tensor) -> List[DetectionRecord]:
        """Akceptuje ``[B,3,H,W]`` lub ``[3,H,W]`` (pojedynczy obraz jak w D-RISE)."""
        if x.ndim == 3:
            if x.shape[0] != 3:
                raise ValueError(f"Tensor 3D musi mieć kształt [3,H,W], mam {tuple(x.shape)}")
            x = x.unsqueeze(0)
        elif x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Oczekiwano [B,3,H,W] lub [3,H,W], mam {tuple(x.shape)}")

        records: List[DetectionRecord] = []
        to_pil = T.ToPILImage()
        for i in range(x.shape[0]):
            tensor_cpu = x[i].detach().cpu().clamp(0, 1)
            pil = to_pil(tensor_cpu)
            det = self._api.predict(pil, threshold=self._score_threshold)

            if det is None or len(det) == 0:
                empty_boxes = torch.zeros((0, 4), device=self._device)
                empty_scores = torch.zeros((0,), device=self._device)
                empty_cls = torch.zeros((0, self._num_classes), device=self._device)
                records.append(
                    DetectionRecord(
                        bounding_boxes=empty_boxes,
                        objectness_scores=empty_scores,
                        class_scores=empty_cls,
                    )
                )
                continue

            xyxy = torch.as_tensor(det.xyxy, dtype=torch.float32, device=self._device)
            conf = torch.as_tensor(det.confidence, dtype=torch.float32, device=self._device)
            labels = torch.as_tensor(det.class_id, dtype=torch.long, device=self._device)
            expanded = expand_class_scores(conf.cpu(), labels.cpu(), self._num_classes).to(self._device)
            ones = torch.ones(conf.shape[0], device=self._device)
            records.append(
                DetectionRecord(
                    bounding_boxes=xyxy,
                    objectness_scores=ones,
                    class_scores=expanded,
                )
            )
        return records
