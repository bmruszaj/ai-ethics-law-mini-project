"""
Własna implementacja GradCAM++ dla RF-DETR — bez pytorch-grad-cam.

Schemat:
    predict() detekcja → IoU match do raw query → target = raw_logit[query, class]
    (opcjonalnie sigmoid(logit)) → backward → GradCAM++ z wybranej warstwy
    Domyślna warstwa CAM (baseline RF-DETR): przestrzenna mapa projektora ``[B,C,H,W]`` —
patrz ``DEFAULT_CAM_TARGET_LAYER_NAME``; przy braku w grafie — fallback do ostatniego ``C2f.cv2``.

Narzędzia:
    inspect_candidate_cam_layers — kandydaci CAM (domyślnie tylko tensory 4D + deduplikacja);
    inspect_cam_candidate_layers — alias powyższego;
    default_target_layer_name — nazwa warstwy domyślnej (kanoniczna lub fallback);
    cam_quality_status — import z ``xai_metrics`` (re-eksport tutaj dla wygody).

Matematyka GradCAM++ (Chattopadhyay et al., arXiv:1710.11063):
    alpha_ij^k = grad^2 / (2*grad^2 + sum_ij(A_ij^k * grad^3))
    weights_k  = sum_ij(alpha_ij^k * relu(grad_ij^k))
    CAM        = relu(sum_k(weights_k * A^k))
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_nn
import torchvision.transforms.functional as F_t
from PIL import Image
from rfdetr.utilities import box_ops

from xai_metrics import cam_quality_status


# ---------------------------------------------------------------------------
# Target-layer resolution (EigenCAM / GradCAM++ baseline)
# ---------------------------------------------------------------------------

# Zalecana przez dokumentację RF-DETR / CAM: pierwszy stage projektora, wyjście [B, C, H, W].
# Jeśli w danej wersji architektury ścieżka nie istnieje, używany jest fallback (ostatni C2f.cv2).
DEFAULT_CAM_TARGET_LAYER_NAME = "backbone.0.projector.stages.0.0.cv2"


def resolve_target_layer(inner: torch.nn.Module) -> torch.nn.Module:
    """Zwraca C2f.cv2 z ostatniego etapu MultiScaleProjector (ta sama warstwa co EigenCAM)."""
    joiner = getattr(inner, "backbone", None)
    if joiner is None:
        raise AttributeError("Model nie ma atrybutu 'backbone' (oczekiwano LWDETR).")
    backbone = next(iter(joiner.children()))
    projector = getattr(backbone, "projector", None)
    if projector is None:
        raise AttributeError("Backbone nie ma 'projector'.")
    stages = getattr(projector, "stages", None)
    if stages is None or len(stages) == 0:
        raise AttributeError("projector.stages jest puste.")
    c2f = stages[-1][0]
    cv2 = getattr(c2f, "cv2", None)
    if cv2 is None:
        raise AttributeError(
            f"Nie znaleziono C2f.cv2. Dostępne atrybuty C2f: {list(vars(c2f).keys())}"
        )
    return cv2


def default_target_layer_name(inner: torch.nn.Module) -> str:
    """Najpierw kanoniczna ścieżka projektora; jeśli jej nie ma w modelu — ostatni ``C2f.cv2``."""
    try:
        get_module_by_name(inner, DEFAULT_CAM_TARGET_LAYER_NAME)
        return DEFAULT_CAM_TARGET_LAYER_NAME
    except KeyError:
        pass
    target = resolve_target_layer(inner)
    for name, module in inner.named_modules():
        if module is target:
            return name
    raise RuntimeError("Nie znaleziono nazwy dla domyślnej warstwy docelowej.")


def get_module_by_name(root: torch.nn.Module, name: str) -> torch.nn.Module:
    for module_name, module in root.named_modules():
        if module_name == name:
            return module
    raise KeyError(f"Brak modułu o nazwie: {name!r}")


def _extract_first_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            t = _extract_first_tensor(item)
            if t is not None:
                return t
    if isinstance(output, dict):
        for item in output.values():
            t = _extract_first_tensor(item)
            if t is not None:
                return t
    return None


def inspect_candidate_cam_layers(
    rfdetr_api: Any,
    pil_image: Image.Image,
    keywords: tuple[str, ...] = ("backbone", "projector", "proj", "encoder"),
    *,
    only_bchw: bool = True,
    dedupe: bool = True,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """
    Jednorazowy forward z hookami: moduły pasujące do ``keywords``.

    Domyślnie zwraca wyłącznie wyjścia **4D** ``[B, C, H, W]`` (naturalna mapa przestrzenna dla CAM),
    bez tokenowych map 3D ViT — bez ``reshape_transform``.

    Zwraca listę ``(pełna_nazwa, klasa_modułu, kształt)``.
    """
    inner = rfdetr_api.model.model
    inner.eval()
    batch, _, _ = preprocess_for_gradcampp(rfdetr_api, pil_image)
    records: list[tuple[str, str, tuple[int, ...]]] = []
    handles: list[Any] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    def make_hook(layer_name: str):
        def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            tensor = _extract_first_tensor(output)
            if tensor is None:
                return
            if only_bchw and tensor.dim() != 4:
                return
            if not only_bchw and tensor.dim() not in (3, 4):
                return
            shape = tuple(tensor.shape)
            key = (layer_name, shape)
            if dedupe and key in seen:
                return
            seen.add(key)
            records.append((layer_name, module.__class__.__name__, shape))

        return hook

    for name, module in inner.named_modules():
        if not name:
            continue
        lower = name.lower()
        if keywords and not any(k in lower for k in keywords):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = inner(batch)

    for h in handles:
        h.remove()

    return records


def inspect_cam_candidate_layers(
    rfdetr_api: Any,
    pil_image: Image.Image,
    **kwargs: Any,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Alias: ``inspect_candidate_cam_layers`` z domyślnym filtrem tylko ``[B,C,H,W]``."""
    return inspect_candidate_cam_layers(rfdetr_api, pil_image, **kwargs)


# ---------------------------------------------------------------------------
# Preprocessing (zgodna z rfdetr predict())
# ---------------------------------------------------------------------------

def resolve_rfdetr_device(rfdetr_api: Any) -> torch.device:
    """
    Urządzenie faktycznych parametrów modułu LWDETR (`next(inner.parameters()).device`).

    Nie używaj samego `rfdetr_api.model.device` — może wskazywać CUDA, podczas gdy backbone
    DINOv2 z Transformers jest na CPU.
    """
    inner = rfdetr_api.model.model
    try:
        return next(inner.parameters()).device
    except StopIteration:
        pass
    attr = getattr(rfdetr_api.model, "device", None)
    if attr is not None:
        return torch.device(attr)
    return torch.device("cpu")


def preprocess_for_gradcampp(
    rfdetr_api: Any,
    pil_image: Image.Image,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """
    Zwraca:
      batch_tensor: [1, 3, res, res] na urządzeniu modelu
      orig_hw:      (H, W) oryginalnego obrazu
      input_hw:     (res, res) tensora wejściowego
    """
    device = resolve_rfdetr_device(rfdetr_api)
    img = pil_image.convert("RGB")
    tensor = F_t.to_tensor(img)
    oh, ow = int(tensor.shape[1]), int(tensor.shape[2])
    res = int(rfdetr_api.model.resolution)
    resized = F_t.resize(tensor, [res, res])
    mean = torch.tensor(rfdetr_api.means, device=device).view(-1, 1, 1)
    std = torch.tensor(rfdetr_api.stds, device=device).view(-1, 1, 1)
    normalized = (resized.to(device) - mean) / std
    batch = normalized.unsqueeze(0)
    return batch, (oh, ow), (res, res)


# ---------------------------------------------------------------------------
# Query matching: predict() detection → raw transformer query
# ---------------------------------------------------------------------------

def _cxcywh_norm_to_xyxy_px(
    boxes_cxcywh: torch.Tensor,
    input_hw: tuple[int, int],
) -> torch.Tensor:
    """[Q, 4] cxcywh normalizowane → [Q, 4] xyxy w pikselach input_hw."""
    ih, iw = input_hw
    xyxy_norm = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)
    scale = xyxy_norm.new_tensor([iw, ih, iw, ih])
    return xyxy_norm * scale


def _iou_xyxy(pred_xyxy: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor:
    """pred_xyxy [Q, 4], box_xyxy [4] → IoU [Q]."""
    ax1, ay1, ax2, ay2 = pred_xyxy.unbind(-1)
    bx1, by1, bx2, by2 = box_xyxy.unbind(-1)
    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)
    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = area_a + area_b - inter
    return torch.where(union > 0.0, inter / union, torch.zeros_like(inter))


def match_detection_to_query(
    pred_detections: Any,
    det_idx: int,
    raw_outputs: dict[str, torch.Tensor],
    orig_hw: tuple[int, int],
    input_hw: tuple[int, int],
    class_aware: bool = True,
) -> dict[str, Any]:
    """
    Dopasowuje detekcję z predict() do raw query przez IoU.

    Konwersja bbox: predict() zwraca xyxy w układzie oryginalnego obrazu,
    raw queries są w układzie input_hw (po resize). Przeliczamy przez scale.

    Zwraca słownik z query_id, matched_iou, logit_class_id i innymi polami
    potrzebnymi do GradCAM++.
    """
    device = raw_outputs["pred_boxes"].device
    oh, ow = orig_hw
    ih, iw = input_hw

    # bbox detekcji w orig przestrzeni → przeskalowany do input
    sx, sy = iw / max(ow, 1e-6), ih / max(oh, 1e-6)
    bx1, by1, bx2, by2 = pred_detections.xyxy[det_idx].tolist()
    box_input = torch.tensor(
        [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy],
        dtype=torch.float32, device=device,
    )

    pred_class_id = int(pred_detections.class_id[det_idx])
    pred_confidence = float(pred_detections.confidence[det_idx])

    logits = raw_outputs["pred_logits"]   # [1, Q, C]
    boxes_raw = raw_outputs["pred_boxes"][0]  # [Q, 4] cxcywh norm

    num_classes = logits.shape[-1]

    # class_id z predict() może być 1-based lub 0-based — sprawdzamy
    if 0 <= pred_class_id < num_classes:
        logit_class_id = pred_class_id
    elif 1 <= pred_class_id <= num_classes:
        logit_class_id = pred_class_id - 1
    else:
        logit_class_id = 0

    query_boxes_input = _cxcywh_norm_to_xyxy_px(boxes_raw, input_hw)
    ious = _iou_xyxy(query_boxes_input, box_input)  # [Q]

    if class_aware:
        probs = logits.sigmoid()[0]                  # [Q, C]
        raw_class_ids = probs.argmax(dim=-1)         # [Q]
        class_mask = raw_class_ids == logit_class_id
        ca_ious = ious.clone()
        ca_ious[~class_mask] = -1.0
        best_q = int(torch.argmax(ca_ious).item())
        best_iou = float(ca_ious[best_q].item())
        if best_iou < 0:
            best_q = int(torch.argmax(ious).item())
            best_iou = float(ious[best_q].item())
    else:
        best_q = int(torch.argmax(ious).item())
        best_iou = float(ious[best_q].item())

    probs_q = logits.sigmoid()[0, best_q, logit_class_id].detach().cpu().item()

    return {
        "det_idx": det_idx,
        "query_id": best_q,
        "matched_iou": best_iou,
        "pred_class_id": pred_class_id,
        "logit_class_id": logit_class_id,
        "pred_confidence": pred_confidence,
        "raw_query_class_score": probs_q,
        "box_orig_xyxy": pred_detections.xyxy[det_idx].tolist(),
        "box_input_xyxy": box_input.detach().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# GradCAM++ core
# ---------------------------------------------------------------------------

class _ActivationHook:
    """Context manager rejestrujący aktywacje i zachowujący grad."""

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self.activation: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> "_ActivationHook":
        def _hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.activation = output
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.activation = item
                        break
            if self.activation is not None:
                self.activation.retain_grad()

        self._handle = self._module.register_forward_hook(_hook)
        return self

    def __exit__(self, *_) -> None:
        if self._handle is not None:
            self._handle.remove()


def _activation_to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """
    Obsługuje [B, C, H, W] i [B, N, C] (tokeny ViT → reshape do kwadratu).
    """
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 3:
        b, n, c = tensor.shape
        root = int(math.isqrt(n))
        if root * root == n:
            h = w = root
            return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        # spróbuj odciąć CLS token
        root = int(math.isqrt(n - 1))
        if root * root == n - 1:
            h = w = root
            return tensor[:, 1:, :].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Nie można reshape tensora aktywacji {tuple(tensor.shape)} do przestrzennej mapy.")
    raise ValueError(f"Nieobsługiwany wymiar aktywacji: {tensor.dim()}")


def compute_gradcampp(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_hw: tuple[int, int],
) -> np.ndarray:
    """
    GradCAM++: alpha_ij^k = grad^2 / (2*grad^2 + sum_ij(A_ij^k * grad^3))
    Zwraca znormalizowaną mapę [0,1] w kształcie output_hw (numpy float32).
    """
    act = _activation_to_bchw(activation)
    grad = _activation_to_bchw(gradient)

    eps = 1e-8
    grad2 = grad.pow(2)
    grad3 = grad.pow(3)

    # mianownik sumy po przestrzeni, keepdim dla kanału
    denom = 2.0 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True)
    alpha = grad2 / (denom + eps)
    # zerujemy alpha tam gdzie gradient = 0 (brak sygnału)
    alpha = torch.where(grad != 0, alpha, torch.zeros_like(alpha))

    weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))          # [B, 1, H, W]

    cam = F_nn.interpolate(cam, size=output_hw, mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)

    lo, hi = cam.min(), cam.max()
    if hi > lo:
        cam = (cam - lo) / (hi - lo)
    else:
        cam = np.zeros_like(cam)

    return cam


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_gradcampp_for_detection(
    rfdetr_api: Any,
    pil_image: Image.Image,
    pred_detections: Any,
    det_idx: int,
    min_match_iou: float = 0.3,
    class_aware: bool = True,
    *,
    target_layer_name: str | None = None,
    target_mode: Literal["logit", "sigmoid_logit"] = "logit",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Oblicza GradCAM++ dla jednej detekcji z predict().

    Parameters
    ----------
    target_layer_name:
        Pełna nazwa modułu z ``inner.named_modules()`` (np. z ``inspect_candidate_cam_layers``).
        ``None`` → ``default_target_layer_name(inner)`` (preferuje ``DEFAULT_CAM_TARGET_LAYER_NAME``).
    target_mode:
        ``logit`` — backward od surowego logitu klasy (standard Grad-CAM).
        ``sigmoid_logit`` — backward od ``sigmoid(logit)`` (inny rozkład gradientu).

    Zwraca:
      cam:   mapa saliencji [0,1] float32, kształt orig_hw (H, W)
      match: słownik z dopasowaniem query + ``target_layer_name``, ``target_mode``
    """
    inner = rfdetr_api.model.model
    inner.eval()

    if target_layer_name is None:
        layer_name_resolved = default_target_layer_name(inner)
        target_layer = get_module_by_name(inner, layer_name_resolved)
    else:
        target_layer = get_module_by_name(inner, target_layer_name)
        layer_name_resolved = target_layer_name

    batch, orig_hw, input_hw = preprocess_for_gradcampp(rfdetr_api, pil_image)

    inner.zero_grad(set_to_none=True)

    with _ActivationHook(target_layer) as hook:
        outputs = inner(batch)

        if isinstance(outputs, tuple):
            outputs = {"pred_logits": outputs[1], "pred_boxes": outputs[0]}

        match = match_detection_to_query(
            pred_detections=pred_detections,
            det_idx=det_idx,
            raw_outputs=outputs,
            orig_hw=orig_hw,
            input_hw=input_hw,
            class_aware=class_aware,
        )

        if match["matched_iou"] < min_match_iou:
            raise RuntimeError(
                f"det={det_idx}: matched_iou={match['matched_iou']:.3f} < min_match_iou={min_match_iou}."
            )

        query_id = match["query_id"]
        class_id = match["logit_class_id"]
        logit_q = outputs["pred_logits"][0, query_id, class_id]
        if target_mode == "sigmoid_logit":
            target_score = logit_q.sigmoid()
        else:
            target_score = logit_q
        target_score.backward(retain_graph=False)

        if hook.activation is None or hook.activation.grad is None:
            raise RuntimeError("Nie udało się pobrać aktywacji/gradientu z warstwy docelowej.")

        cam = compute_gradcampp(
            activation=hook.activation,
            gradient=hook.activation.grad,
            output_hw=orig_hw,
        )

    match_out = {
        **match,
        "target_layer_name": layer_name_resolved,
        "target_mode": target_mode,
    }
    return cam, match_out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_box_on_heatmap(cam: np.ndarray, box_xyxy: tuple[float, ...]) -> np.ndarray:
    """Sanity check: mapa JET + zielony bbox (ten sam układ co ``cam`` — zwykle orig_hw)."""
    cam_rgb = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0.0, 1.0)), cv2.COLORMAP_JET)
    cam_rgb = cv2.cvtColor(cam_rgb, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    cv2.rectangle(cam_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return cam_rgb


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    cam_u8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap, alpha, 0)


def _draw_box(img: np.ndarray, xyxy, color=(0, 255, 0), thickness=3) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def make_gradcampp_figure(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    match: dict[str, Any],
    class_name: str,
    cam_quality: str | None = None,
) -> plt.Figure:
    """Panel: obraz+bbox | CAM | CAM+bbox (sanity) | overlay+bbox."""
    box = match["box_orig_xyxy"]
    overlay = overlay_cam_on_image(image_rgb, cam)
    cam_box_rgb = draw_box_on_heatmap(cam, tuple(box))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(_draw_box(image_rgb, box))
    axes[0].set_title("Detekcja")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GradCAM++")
    axes[1].axis("off")

    axes[2].imshow(cam_box_rgb)
    axes[2].set_title("CAM + bbox (układ współrzędnych)")
    axes[2].axis("off")

    axes[3].imshow(_draw_box(overlay, box))
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    qual = f"  quality={cam_quality}" if cam_quality else ""
    fig.suptitle(
        f"GradCAM++  det={match['det_idx']}  query={match['query_id']}  "
        f"class={class_name}  conf={match['pred_confidence']:.3f}  "
        f"IoU(query)={match['matched_iou']:.3f}{qual}"
    )
    plt.tight_layout()
    return fig


def save_gradcampp_result(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    match: dict[str, Any],
    class_name: str,
    out_dir: Path,
    stem: str,
    extra_meta: dict | None = None,
    cam_quality: str | None = None,
) -> Path:
    """Zapisuje PNG + npy + JSON; zwraca ścieżkę do PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{stem}.npy", cam)

    fig = make_gradcampp_figure(image_rgb, cam, match, class_name, cam_quality=cam_quality)
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "method": "GradCAM++",
        **match,
        "class_name": class_name,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return png_path
