"""
Uruchomienie D-RISE (domyślnie N=1000, ``mask_res=(8,8)``) z opcjonalnym cache na dysku.

Powtarzalność: metoda losuje maski — przed wywołaniem ustaw ``torch.manual_seed(...)``
albo użyj ``cache_dir``, żeby nie przeliczać tych samych obrazów.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import zoom
from torchvision import transforms as T

from rfdetr_drise_wrapper import RFDETRDriseWrapper

from vision_explanation_methods.explanations import drise as drise_mod


def _sal_to_numpy_list(sal: list) -> List[np.ndarray]:
    """Spłaszcza zwrócone przez ``DRISE_saliency`` struktury do listy tablic."""
    out: List[np.ndarray] = []
    for img_maps in sal:
        for item in img_maps:
            if isinstance(item, dict) and "detection" in item:
                out.append(item["detection"].detach().cpu().numpy())
    return out


def _cache_key(image_path: Path, num_masks: int, mask_res: Tuple[int, int], fast: bool) -> str:
    raw = f"{image_path.resolve()}|{num_masks}|{mask_res}|{fast}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def compute_iou_boxes(box1: np.ndarray | Sequence[float], box2: np.ndarray | Sequence[float]) -> float:
    """IoU between two ``[x1, y1, x2, y2]`` boxes (numpy scalars)."""
    b1 = np.asarray(box1, dtype=np.float64).ravel()[:4]
    b2 = np.asarray(box2, dtype=np.float64).ravel()[:4]
    x1_inter = max(b1[0], b2[0])
    y1_inter = max(b1[1], b2[1])
    x2_inter = min(b1[2], b2[2])
    y2_inter = min(b1[3], b2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter_area
    return float(inter_area / (union + 1e-8))


def drise_target_score(detections: Any, target_box: np.ndarray | Sequence[float], target_class_id: int, gamma: float = 1.0) -> float:
    """
    D-RISE-style similarity for RF-DETR (Supervision ``Detections``): same-class predictions only,
    continuous score ``confidence * IoU^gamma`` (max over predictions).

    IoU-weighted matching reduces target switching by assigning lower scores to predictions that
    do not spatially overlap the original target box.
    """
    if detections is None or len(detections) == 0:
        return 0.0
    tb = np.asarray(target_box, dtype=np.float64).ravel()[:4]
    best = 0.0
    for i in range(len(detections)):
        if int(detections.class_id[i]) != int(target_class_id):
            continue
        iou_v = compute_iou_boxes(detections.xyxy[i], tb)
        conf = float(detections.confidence[i])
        score = conf * (iou_v**gamma)
        if score > best:
            best = score
    return float(best)


def _tensor_to_hw_saliency(t: torch.Tensor) -> np.ndarray:
    """``[3,H,W]`` or ``[1,H,W]`` → ``[H,W]`` float32."""
    x = t.detach().cpu().float().numpy()
    if x.ndim == 3:
        if x.shape[0] == 1:
            return x[0].astype(np.float32)
        return x.mean(axis=0).astype(np.float32)
    return x.astype(np.float32)


def normalize_saliency_for_display(raw: np.ndarray) -> np.ndarray:
    """Min–max to ``[0, 1]`` for visualization only (not for cross-detection comparison)."""
    r = raw.astype(np.float64)
    lo, hi = r.min(), r.max()
    if hi - lo < 1e-12:
        return np.zeros_like(r, dtype=np.float32)
    return ((r - lo) / (hi - lo + 1e-8)).astype(np.float32)


def resize_hw_map(map_hw: np.ndarray, target_hw: Tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize ``[H,W]`` map to ``target_hw`` (height, width)."""
    th, tw = target_hw
    if map_hw.shape[0] == th and map_hw.shape[1] == tw:
        return map_hw.astype(np.float32)
    zf = (th / map_hw.shape[0], tw / map_hw.shape[1])
    return zoom(map_hw, zf, order=order).astype(np.float32)


def box_energy_fraction(saliency_hw: np.ndarray, xyxy: np.ndarray | Sequence[float]) -> float:
    """Fraction of total saliency mass inside the axis-aligned target box."""
    h, w = saliency_hw.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(xyxy).ravel()[:4]]
    xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xi2, yi2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inside = float(saliency_hw[yi1:yi2, xi1:xi2].sum())
    total = float(saliency_hw.sum()) + 1e-12
    return inside / total


def expert_energy_fraction(saliency_hw: np.ndarray, expert_mask_hw: np.ndarray) -> float:
    """``sum(saliency * expert_mask) / sum(saliency)`` — share of explanation energy on expert region."""
    m = expert_mask_hw.astype(np.float64)
    if m.shape != saliency_hw.shape:
        raise ValueError(f"expert mask shape {m.shape} != saliency {saliency_hw.shape}")
    num = float((saliency_hw.astype(np.float64) * (m > 0)).sum())
    den = float(saliency_hw.sum()) + 1e-12
    return num / den


def pointing_game_hit(saliency_hw: np.ndarray, expert_mask_hw: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
    """Maximum saliency pixel falls inside binary ``expert_mask``."""
    flat_argmax = int(np.argmax(saliency_hw))
    y, x = np.unravel_index(flat_argmax, saliency_hw.shape)
    hit = bool(expert_mask_hw[y, x] > 0)
    return hit, (int(y), int(x))


def iou_dice_vs_expert(saliency_hw: np.ndarray, expert_mask_hw: np.ndarray, *, quantile: float = 90.0) -> Tuple[float, float]:
    """Threshold saliency at ``quantile`` percentile; IoU and Dice vs binary expert mask."""
    thr = np.percentile(saliency_hw, quantile)
    pred = saliency_hw >= thr
    gt = expert_mask_hw > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / (union + 1e-8))
    dice = float(2.0 * inter / (pred.sum() + gt.sum() + 1e-8))
    return iou, dice


def saliency_map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two equal-shaped maps."""
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_drise_for_image(
    rfdetr_api: Any,
    image_path: str | Path,
    num_classes: int,
    num_masks: int = 1000,
    mask_res: Tuple[int, int] = (8, 8),
    cache_dir: str | Path | None = None,
    fast_mode: bool = False,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> list:
    """
    Zwraca strukturę saliencji z ``vision_explanation_methods`` (lista na obraz → lista dictów z kluczem ``detection``).

    ``fast_mode``: ogranicza ``num_masks`` do co najwyżej 64 (debug).
    """
    image_path = Path(image_path)
    if fast_mode:
        num_masks = min(64, num_masks)

    cache_key = _cache_key(image_path, num_masks, mask_res, fast_mode)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path = cache_dir / f"drise_{cache_key}.npz"
        meta_path = cache_dir / f"drise_{cache_key}.meta.json"
        if npz_path.is_file() and meta_path.is_file():
            loaded = list(np.load(npz_path).values())
            return [[{"detection": torch.from_numpy(arr)} for arr in loaded]]

    if device is None:
        device = str(rfdetr_api.model.device)

    pil = Image.open(image_path).convert("RGB")
    img_tensor = T.ToTensor()(pil).to(device)

    wrapper = RFDETRDriseWrapper(rfdetr_api, num_classes=num_classes)
    detections = wrapper.predict(img_tensor.unsqueeze(0))

    sal = drise_mod.DRISE_saliency(
        model=wrapper,
        image_tensor=img_tensor,
        target_detections=detections,
        number_of_masks=num_masks,
        mask_res=mask_res,
        mask_padding=None,
        device=device,
        verbose=verbose,
    )

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        flat = _sal_to_numpy_list(sal)
        if flat:
            np.savez_compressed(cache_dir / f"drise_{cache_key}.npz", *flat)
            (cache_dir / f"drise_{cache_key}.meta.json").write_text(
                json.dumps(
                    {"image": str(image_path), "num_masks": num_masks, "mask_res": list(mask_res)},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    return sal


def run_drise_per_target(
    rfdetr_api: Any,
    image_path: str | Path,
    *,
    target_box_xyxy: np.ndarray | Sequence[float],
    target_class_id: int,
    target_confidence: float | None = None,
    num_masks: int = 1000,
    mask_res: Tuple[int, int] = (8, 8),
    gamma: float = 1.0,
    center_affinities: bool = True,
    device: str | torch.device | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Per-target D-RISE with explicit RF-DETR scoring: ``max(conf_target_class * IoU(pred, target_box)^gamma)``.

    Returns **raw** fused saliency (no per-map min–max); use ``normalize_saliency_for_display`` only for plots.

    Matches Microsoft D-RISE fusion: weighted sum of masks, optional centering by mean affinity (see ``vision_explanation_methods``).
    """
    if seed is not None:
        torch.manual_seed(seed)

    image_path = Path(image_path)
    if device is None:
        device = str(rfdetr_api.model.device)

    pil = Image.open(image_path).convert("RGB")
    img_tensor = T.ToTensor()(pil).to(device)
    img_size = (img_tensor.shape[-2], img_tensor.shape[-1])

    padding = int(max(img_size[0] / mask_res[0], img_size[1] / mask_res[1])) if mask_res else 0
    if mask_res is None or padding <= 0:
        raise ValueError("invalid mask_res")

    weighted_accum = torch.zeros_like(img_tensor, dtype=torch.float32, device="cpu")
    mask_sum_accum = torch.zeros_like(img_tensor, dtype=torch.float32, device="cpu")
    scores: List[float] = []
    visible_fracs: List[float] = []

    for _ in range(num_masks):
        mask = drise_mod.generate_mask(mask_res, img_size, padding, device)
        masked = drise_mod.fuse_mask(img_tensor, mask)
        pil_m = T.ToPILImage()(masked.cpu().clamp(0, 1))
        with torch.no_grad():
            dets = rfdetr_api.predict(pil_m, threshold=0.01)
        score = drise_target_score(dets, target_box_xyxy, int(target_class_id), gamma=gamma)
        scores.append(score)
        visible_fracs.append(float(mask.mean().item()))

        m_cpu = mask.detach().float().cpu()
        w = score * m_cpu
        weighted_accum += w
        mask_sum_accum += m_cpu

    mean_score = float(np.mean(scores)) if scores else 0.0

    if center_affinities:
        weighted_accum = weighted_accum - mean_score * mask_sum_accum

    raw_hw = _tensor_to_hw_saliency(weighted_accum)

    meta = {
        "target_class": int(target_class_id),
        "target_confidence": float(target_confidence) if target_confidence is not None else None,
        "target_box": [float(x) for x in np.asarray(target_box_xyxy).ravel()[:4]],
        "number_of_masks": int(num_masks),
        "mask_resolution": [int(mask_res[0]), int(mask_res[1])],
        "mask_mean_visible_fraction": float(np.mean(visible_fracs)) if visible_fracs else float("nan"),
        "mean_target_score": float(np.mean(scores)) if scores else 0.0,
        "max_target_score": float(np.max(scores)) if scores else 0.0,
        "matched_detection_rate": float(np.mean(np.array(scores) > 0.0)) if scores else 0.0,
        "gamma": float(gamma),
        "center_affinities": bool(center_affinities),
    }
    if verbose:
        print(meta)

    return {
        "raw_saliency": raw_hw.astype(np.float32),
        "per_mask_scores": scores,
        "meta": meta,
        "pil_size": (pil.size[1], pil.size[0]),
    }


def deletion_insertion_curves(
    rfdetr_api: Any,
    pil_rgb: Image.Image,
    *,
    target_box_xyxy: np.ndarray | Sequence[float],
    target_class_id: int,
    saliency_hw: np.ndarray,
    gamma: float = 1.0,
    predict_threshold: float = 0.01,
    steps: int = 20,
    blur_radius: int = 18,
) -> Dict[str, Any]:
    """
    **Deletion**: progressively replace most salient pixels with a blurred baseline; track ``drise_target_score``.
    **Insertion**: start from blurred image; progressively copy original pixels at most salient locations.

    ``saliency_hw`` must match image spatial size ``[H,W]``.
    """
    device = str(rfdetr_api.model.device)
    orig = T.ToTensor()(pil_rgb.convert("RGB")).to(device)
    h, w = saliency_hw.shape[-2], saliency_hw.shape[-1]
    if orig.shape[-2] != h or orig.shape[-1] != w:
        raise ValueError("saliency map must match image H×W")

    blur_pil = pil_rgb.filter(ImageFilter.GaussianBlur(blur_radius))
    blurred = T.ToTensor()(blur_pil.convert("RGB")).to(device)

    flat_order = np.argsort(-saliency_hw.astype(np.float64).ravel())
    n_pix = h * w
    step_sizes = [int(round(i / steps * n_pix)) for i in range(steps + 1)]

    deletion_scores: List[float] = []
    insertion_scores: List[float] = []

    for k in step_sizes:
        removed = torch.zeros(n_pix, dtype=torch.bool, device=device)
        if k > 0:
            idx_t = torch.as_tensor(flat_order[:k], device=device, dtype=torch.long)
            removed[idx_t] = True
        removed_map = removed.view(h, w)
        rm3 = removed_map.unsqueeze(0).expand_as(orig)
        del_tensor = torch.where(rm3, blurred, orig)
        with torch.no_grad():
            dets_d = rfdetr_api.predict(T.ToPILImage()(del_tensor.cpu().clamp(0, 1)), threshold=predict_threshold)
        deletion_scores.append(drise_target_score(dets_d, target_box_xyxy, int(target_class_id), gamma=gamma))

        ins_tensor = blurred.clone()
        if k > 0:
            idx_t = torch.as_tensor(flat_order[:k], device=device, dtype=torch.long)
            rev = torch.zeros(n_pix, dtype=torch.bool, device=device)
            rev[idx_t] = True
            rev_map = rev.view(h, w)
            rv3 = rev_map.unsqueeze(0).expand_as(orig)
            ins_tensor = torch.where(rv3, orig, blurred)
        with torch.no_grad():
            dets_i = rfdetr_api.predict(T.ToPILImage()(ins_tensor.cpu().clamp(0, 1)), threshold=predict_threshold)
        insertion_scores.append(drise_target_score(dets_i, target_box_xyxy, int(target_class_id), gamma=gamma))

    return {
        "deletion_target_scores": np.array(deletion_scores, dtype=np.float32),
        "insertion_target_scores": np.array(insertion_scores, dtype=np.float32),
        "fractions": np.array([k / max(n_pix, 1) for k in step_sizes], dtype=np.float32),
    }
