"""
EigenCAM dla RF-DETR — implementacja własna bez pytorch-grad-cam.

Metoda (arXiv:2008.00299):
    1. Forward z hookiem na wybraną warstwę → aktywacja [B, C, H, W].
    2. Spłaszczenie przestrzenne → [N, C], centering.
    3. SVD → pierwszy komponent główny (kierunek max wariancji w przestrzeni kanałów).
    4. Projekcja każdego punktu przestrzennego → mapa H × W.
    5. ReLU + normalizacja min-max → interpolacja do orig_hw.

Ważne: EigenCAM jest gradient-free i nie jest query-specific — ta sama mapa dla całego
obrazu. Do oceny per-detekcja ta sama mapa jest ewaluowana osobno dla każdego bbox.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_nn

from xai_metrics import cam_quality_status

from rfdetr_gradcampp_custom import (
    _activation_to_bchw,
    _draw_box,
    _extract_first_tensor,
    default_target_layer_name,
    draw_box_on_heatmap,
    get_module_by_name,
    inspect_candidate_cam_layers,
    match_detection_to_query,
    overlay_cam_on_image,
    preprocess_for_gradcampp,
    resolve_rfdetr_device,
)


# ---------------------------------------------------------------------------
# Hook na aktywacje (bez retencji gradientu — EigenCAM nie wymaga backward)
# ---------------------------------------------------------------------------

class ActivationHook:
    """Context manager przechwytujący wyjście warstwy podczas forward (bez grad)."""

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self.activation: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> "ActivationHook":
        def _hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            tensor = _extract_first_tensor(output)
            if tensor is None:
                raise RuntimeError(
                    "Wyjście warstwy docelowej nie zawiera tensora."
                )
            self.activation = tensor.detach()

        self._handle = self._module.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._handle is not None:
            self._handle.remove()


# ---------------------------------------------------------------------------
# Core EigenCAM
# ---------------------------------------------------------------------------

def compute_eigencam_from_activation(
    activation: torch.Tensor,
    output_hw: tuple[int, int],
) -> np.ndarray:
    """
    EigenCAM z aktywacji jednego obrazu.

    Parametry
    ----------
    activation:
        Tensor aktywacji z warstwy docelowej — dowolny kształt obsługiwany przez
        ``_activation_to_bchw`` ([B, C, H, W] lub [B, N, C]).
    output_hw:
        Docelowy rozmiar (H, W) mapy wyjściowej (zwykle orig_hw obrazu).

    Zwraca
    ------
    cam : np.ndarray float32, kształt output_hw, wartości w [0, 1].
    """
    act = _activation_to_bchw(activation)  # [B, C, H, W]

    if act.shape[0] != 1:
        raise ValueError(
            f"EigenCAM oczekuje batch_size=1, otrzymano {act.shape[0]}."
        )

    act = act[0]  # [C, H, W]
    c, h, w = act.shape

    # [C, H, W] → [N, C] gdzie N = H*W
    features = act.reshape(c, h * w).permute(1, 0).float()  # [N, C]

    # Centering (PCA)
    features = features - features.mean(dim=0, keepdim=True)

    # SVD: features = U S Vh;  pierwszy wiersz Vh = pierwszy komponent główny w R^C
    try:
        _, _, vh = torch.linalg.svd(features, full_matrices=False)
    except RuntimeError:
        # Fallback na CPU jeśli GPU SVD zawodzi
        _, _, vh = torch.linalg.svd(features.detach().cpu(), full_matrices=False)
        vh = vh.to(features.device)

    principal = vh[0]  # [C]

    # Projekcja: każdy piksel → skalar
    cam_flat = features @ principal  # [N]
    cam = cam_flat.reshape(h, w)

    # SVD ma nieokreślony znak — odwróć, jeśli większość jest ujemna
    if float(cam.sum()) < 0:
        cam = -cam

    cam = torch.relu(cam)

    # Normalizacja → interpolacja → normalizacja
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam = cam[None, None, :, :]  # [1, 1, h, w]
    cam = F_nn.interpolate(cam, size=output_hw, mode="bilinear", align_corners=False)
    cam = cam[0, 0]

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def compute_rfdetr_eigencam(
    rfdetr_api: Any,
    pil_image: Any,
    *,
    target_layer_name: str | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], str]:
    """
    EigenCAM dla jednego obrazu.

    Parametry
    ----------
    target_layer_name:
        Pełna nazwa modułu z ``inner.named_modules()``.
        ``None`` → ``default_target_layer_name`` (kanoniczny projektor ``cv2`` lub fallback).

    Zwraca
    ------
    cam           : np.ndarray float32 [H_orig, W_orig] — znormalizowana mapa.
    orig_hw       : (H, W) oryginalnego obrazu.
    input_hw      : (H, W) tensora wejściowego modelu.
    layer_name    : pełna nazwa użytej warstwy (do logowania).
    """
    inner = rfdetr_api.model.model
    inner.eval()

    if target_layer_name is None:
        layer_name = default_target_layer_name(inner)
        target_layer = get_module_by_name(inner, layer_name)
    else:
        target_layer = get_module_by_name(inner, target_layer_name)
        layer_name = target_layer_name

    batch, orig_hw, input_hw = preprocess_for_gradcampp(rfdetr_api, pil_image)

    with torch.no_grad():
        with ActivationHook(target_layer) as hook:
            _ = inner(batch)

        if hook.activation is None:
            raise RuntimeError(
                f"Hook nie przechwycił aktywacji z warstwy '{layer_name}'."
            )

        cam = compute_eigencam_from_activation(
            activation=hook.activation,
            output_hw=orig_hw,
        )

    return cam, orig_hw, input_hw, layer_name


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_eigencam_figure(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    match: dict[str, Any],
    class_name: str,
    layer_name: str,
    cam_quality: str | None = None,
) -> plt.Figure:
    """
    Panel 4-kadrowy: obraz+bbox | CAM (jet) | CAM+bbox | overlay+bbox.

    Analogiczny do ``make_gradcampp_figure`` z ``rfdetr_gradcampp_custom``.
    """
    box = match["box_orig_xyxy"]
    overlay = overlay_cam_on_image(image_rgb, cam)
    cam_box_rgb = draw_box_on_heatmap(cam, tuple(box))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(_draw_box(image_rgb, box))
    axes[0].set_title("Detekcja")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("EigenCAM (obraz globalny)")
    axes[1].axis("off")

    axes[2].imshow(cam_box_rgb)
    axes[2].set_title("EigenCAM + bbox")
    axes[2].axis("off")

    axes[3].imshow(_draw_box(overlay, box))
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    qual_str = f"  quality={cam_quality}" if cam_quality else ""
    layer_short = layer_name.split(".")[-3:] if "." in layer_name else [layer_name]
    fig.suptitle(
        f"EigenCAM  det={match['det_idx']}  query={match['query_id']}  "
        f"class={class_name}  conf={match['pred_confidence']:.3f}  "
        f"IoU(query)={match['matched_iou']:.3f}{qual_str}\n"
        f"layer={'.'.join(layer_short)}"
    )
    plt.tight_layout()
    return fig


def save_eigencam_result(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    match: dict[str, Any],
    class_name: str,
    layer_name: str,
    out_dir: Path,
    stem: str,
    extra_meta: dict | None = None,
    cam_quality: str | None = None,
) -> Path:
    """Zapisuje PNG + npy + JSON dla jednej detekcji; zwraca ścieżkę PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{stem}.npy", cam)

    fig = make_eigencam_figure(
        image_rgb, cam, match, class_name, layer_name, cam_quality=cam_quality
    )
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta: dict[str, Any] = {
        "method": "EigenCAM",
        "target_layer_name": layer_name,
        "note": (
            "EigenCAM jest gradient-free i nie jest query-specific. "
            "Ta sama mapa obrazu jest ewaluowana osobno dla każdej detekcji."
        ),
        **match,
        "class_name": class_name,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return png_path


# ---------------------------------------------------------------------------
# Sweep warstw
# ---------------------------------------------------------------------------

def eigencam_layer_sweep(
    rfdetr_api: Any,
    pil_image: Any,
    pred_detections: Any,
    det_idx: int,
    raw_outputs: dict[str, Any],
    orig_hw: tuple[int, int],
    input_hw: tuple[int, int],
    layer_names: list[str],
    min_match_iou: float = 0.3,
    class_aware: bool = True,
) -> list[dict[str, Any]]:
    """
    Sweep po liście warstw: dla każdej liczy EigenCAM i metryki dla jednej detekcji.

    Zwraca listę słowników z wynikami (``layer_name``, ``pointing_game``,
    ``energy_fraction``, ``status``).
    """
    from xai_metrics import (
        energy_enrichment_inside_bbox,
        energy_fraction_inside_bbox,
        pointing_game_hit,
    )

    results: list[dict[str, Any]] = []

    match = match_detection_to_query(
        pred_detections=pred_detections,
        det_idx=det_idx,
        raw_outputs=raw_outputs,
        orig_hw=orig_hw,
        input_hw=input_hw,
        class_aware=class_aware,
    )

    if match["matched_iou"] < min_match_iou:
        raise RuntimeError(
            f"Detekcja {det_idx}: matched_iou={match['matched_iou']:.3f} < {min_match_iou}"
        )

    box = pred_detections.xyxy[det_idx]
    box_t = tuple(box.tolist())

    for layer_name in layer_names:
        try:
            cam, _, _, _ = compute_rfdetr_eigencam(
                rfdetr_api, pil_image, target_layer_name=layer_name
            )
            hit = pointing_game_hit(cam, box_t)
            ef = energy_fraction_inside_bbox(cam, box_t)
            ee = energy_enrichment_inside_bbox(cam, box_t)
            results.append(
                {
                    "layer_name": layer_name,
                    "pointing_game": int(hit),
                    "energy_fraction": float(ef),
                    "energy_enrichment": float(ee),
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "layer_name": layer_name,
                    "pointing_game": 0,
                    "energy_fraction": 0.0,
                    "energy_enrichment": 0.0,
                    "status": f"failed: {exc}",
                }
            )

    return results
