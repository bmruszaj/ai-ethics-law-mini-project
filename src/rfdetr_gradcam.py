"""
EigenCAM i GradCAM++ dla RF-DETR (wewnętrzny moduł ``LWDETR``).

Warstwa docelowa (pytorch-grad-cam): domyślnie kanoniczna przestrzenna warstwa projektora
``backbone.0.projector.stages.0.0.cv2`` (``DEFAULT_CAM_TARGET_LAYER_NAME`` w ``rfdetr_gradcampp_custom``),
jeśli istnieje w grafie; w przeciwnym razie ostatni blok ``C2f.cv2`` w ostatnim etapie ``MultiScaleProjector``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torchvision.transforms.functional as F_t
from PIL import Image

from rfdetr_detection_target import DetectionScoreTarget
from rfdetr_gradcampp_custom import (
    DEFAULT_CAM_TARGET_LAYER_NAME,
    default_target_layer_name,
    get_module_by_name,
    resolve_target_layer,
)

CamKind = Literal["eigen", "gradcampp"]


def resolve_cam_target_layers(inner: torch.nn.Module) -> list[torch.nn.Module]:
    """Zwraca listę warstw docelowych dla ``pytorch_grad_cam`` (jedna warstwa)."""
    try:
        return [get_module_by_name(inner, DEFAULT_CAM_TARGET_LAYER_NAME)]
    except KeyError:
        pass
    try:
        return [get_module_by_name(inner, default_target_layer_name(inner))]
    except KeyError:
        pass
    return [resolve_target_layer(inner)]


def preprocess_for_cam_batch(
    rfdetr_api: Any,
    pil_image: Image.Image,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Tensor [1,3,H,W] na urządzeniu modelu + kształty (orig_hw, input_hw)."""
    img = pil_image.convert("RGB")
    tensor = F_t.to_tensor(img)
    oh, ow = int(tensor.shape[1]), int(tensor.shape[2])
    res = int(rfdetr_api.model.resolution)
    resized = F_t.resize(tensor, [res, res])
    mean = torch.tensor(rfdetr_api.means, device=device).view(-1, 1, 1)
    std = torch.tensor(rfdetr_api.stds, device=device).view(-1, 1, 1)
    normalized = (resized.to(device) - mean) / std
    batch = normalized.unsqueeze(0).detach().requires_grad_(True)
    return batch, (oh, ow), (res, res)


def run_cam_and_save(
    rfdetr_api: Any,
    pil_image: Image.Image,
    matched_box_xyxy_orig: tuple[float, float, float, float],
    class_id_model: int,
    out_png: str | Path,
    method: CamKind = "eigen",
    metadata_extra: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Uruchamia EigenCAM lub GradCAM++, zapisuje PNG mapy (szarość nałożona na obraz) oraz JSON metadanych.
    Zwraca mapę saliencji 2D float ``numpy`` (ta sama przestrzeń co obraz wejściowy modelu — kwadrat ``resolution``).
    """
    from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image

    inner = rfdetr_api.model.model
    inner.eval()
    device = rfdetr_api.model.device
    target_layers = resolve_cam_target_layers(inner)

    batch, orig_hw, input_hw = preprocess_for_cam_batch(rfdetr_api, pil_image, device)
    target = DetectionScoreTarget(
        matched_box_xyxy_orig=matched_box_xyxy_orig,
        class_id_model=class_id_model,
        orig_hw=orig_hw,
        input_hw=input_hw,
    )

    CamCls = EigenCAM if method == "eigen" else GradCAMPlusPlus
    cam = CamCls(model=inner, target_layers=target_layers, reshape_transform=None)
    grayscale_cam = cam(input_tensor=batch, targets=[target])[0, :]

    rgb = np.array(pil_image.convert("RGB").resize((input_hw[1], input_hw[0])), dtype=np.float32) / 255.0
    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(out_png)

    meta = {
        "method": method,
        "orig_hw": list(orig_hw),
        "input_hw": list(input_hw),
        "matched_box_xyxy_orig": list(matched_box_xyxy_orig),
        "class_id_model": int(class_id_model),
        "png": str(out_png.resolve()),
    }
    if metadata_extra:
        meta.update(metadata_extra)
    out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    cleanup = getattr(cam, "cleanup", None)
    if callable(cleanup):
        cleanup()
    else:
        release = getattr(cam, "release", None)
        if callable(release):
            release()
    return grayscale_cam
