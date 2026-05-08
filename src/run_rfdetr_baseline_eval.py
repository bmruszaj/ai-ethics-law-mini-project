#!/usr/bin/env python3
"""
Ewaluacja baseline RF-DETR na zbiorze COCO (jeden checkpoint).

**Protokół referencyjny (to repo):** ``full_dataset_eval`` — inference na wszystkich obrazach
z adnotacjami w podanym JSON COCO i policzenie mAP (pycocotools) oraz F1 per klasa z greedy matching (IoU).

LOO / 5-fold CV jako **trening** pozostają w repozytorium ``dermatoscopy_ai`` (np. ``FoldManager``,
``train_rfdetr_5fold.py``); tutaj wczytujesz już wyuczony checkpoint i raportujesz metryki na wybranym podzbiorze.

Uruchomienie (po ``uv sync --extra rfdetr-xai``):

    python src/run_rfdetr_baseline_eval.py \\
      --coco_json /ścieżka/_annotations.coco.json \\
      --images_dir /ścieżka/do/obrazów \\
      --checkpoint /ścieżka/best_model.pt \\
      --model_size large \\
      --output_dir wyniki/rfdetr_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dermato_ai.coco.dataset_manager import COCODatasetManager  # noqa: E402

from rfdetr_eval_config import (  # noqa: E402
    DetectionEvalConfig,
    coco_bbox_xywh_to_xyxy,
    greedy_match_single_class,
)
from rfdetr_load import load_rfdetr  # noqa: E402


def _sorted_categories(coco: COCO) -> list[dict[str, Any]]:
    return sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])


def _coco_predictions_for_images(
    model: Any,
    coco: COCO,
    images_dir: Path,
    conf_thresh: float,
    idx_to_cat_id: list[int],
) -> tuple[list[dict[str, Any]], int]:
    preds: list[dict[str, Any]] = []
    next_id = 1
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        path = images_dir / info["file_name"]
        if not path.is_file():
            alt = images_dir / "train" / info["file_name"]
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue
        pil = Image.open(path).convert("RGB")
        det = model.predict(pil, threshold=conf_thresh)
        if det is None or len(det) == 0:
            continue
        for i in range(len(det)):
            x1, y1, x2, y2 = det.xyxy[i].tolist()
            w, h = x2 - x1, y2 - y1
            mid = int(det.class_id[i])
            if mid < 0 or mid >= len(idx_to_cat_id):
                continue
            coco_cat = int(idx_to_cat_id[mid])
            preds.append(
                {
                    "id": next_id,
                    "image_id": int(img_id),
                    "category_id": coco_cat,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(det.confidence[i]),
                    "area": float(max(w, 0) * max(h, 0)),
                    "iscrowd": 0,
                }
            )
            next_id += 1
    return preds, next_id


def _accumulate_f1_matches(
    model: Any,
    coco: COCO,
    images_dir: Path,
    cfg: DetectionEvalConfig,
    cat_id_to_idx: dict[int, int],
    num_classes: int,
) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)}

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        path = images_dir / info["file_name"]
        if not path.is_file():
            alt = images_dir / "train" / info["file_name"]
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        pil = Image.open(path).convert("RGB")
        det = model.predict(pil, threshold=cfg.confidence_threshold)
        det_empty = det is None or len(det) == 0
        pred_boxes = [] if det_empty else det.xyxy.tolist()
        pred_scores = [] if det_empty else det.confidence.tolist()
        pred_cls = [] if det_empty else det.class_id.astype(int).tolist()

        gt_by_c: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            # Skip annotations for classes not in our evaluation set
            if ann["category_id"] not in cat_id_to_idx:
                continue
            cid = cat_id_to_idx[ann["category_id"]]
            gt_by_c[cid].append(coco_bbox_xywh_to_xyxy(ann["bbox"]))

        preds_by_c: dict[int, list[tuple[tuple[float, float, float, float], float]]] = defaultdict(list)
        if not det_empty:
            for i, cls in enumerate(pred_cls):
                if cls < 0 or cls >= num_classes:
                    continue
                preds_by_c[int(cls)].append((tuple(pred_boxes[i]), float(pred_scores[i])))

        for c in range(num_classes):
            gt_boxes = gt_by_c.get(c, [])
            plist = preds_by_c.get(c, [])
            if not plist:
                counts[c]["fn"] += len(gt_boxes)
                continue
            pred_boxes_c = [t[0] for t in plist]
            pred_scores_c = [t[1] for t in plist]
            matches = greedy_match_single_class(
                pred_boxes_c,
                pred_scores_c,
                gt_boxes,
                cfg.iou_match_threshold,
            )
            tp = len(matches)
            fp = len(pred_boxes_c) - tp
            fn = len(gt_boxes) - tp
            counts[c]["tp"] += tp
            counts[c]["fp"] += fp
            counts[c]["fn"] += fn

    return counts


def _f1(tp: int, fp: int, fn: int, eps: float = 1e-12) -> float:
    prec = tp / max(tp + fp, eps)
    rec = tp / max(tp + fn, eps)
    return float(2 * prec * rec / max(prec + rec, eps))


def _resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve dataset paths from direct COCO args or dataset_manager archive flow."""
    use_archive_flow = bool(args.zip_path or args.archive_dir or args.use_newest_zip)

    if use_archive_flow:
        if args.zip_path:
            manager = COCODatasetManager.from_archive(archive_path=args.zip_path)
        elif args.archive_dir:
            manager = COCODatasetManager.from_archive(
                directory=args.archive_dir,
                archive_pattern=args.archive_pattern,
            )
        else:
            manager = COCODatasetManager.from_archive()

        coco_json = Path(manager.json_path)
        if args.write_dedup_json:
            dedup_json = args.output_dir / "resolved_dataset.result_dedup.json"
            manager.save(str(dedup_json))
            coco_json = dedup_json

        images_dir = args.images_dir if args.images_dir else (coco_json.parent / "images")
        return coco_json, images_dir

    if args.coco_json is None:
        raise ValueError(
            "Provide --coco_json (and optionally --images_dir), or use one of: "
            "--zip_path, --archive_dir, --use_newest_zip"
        )

    coco_json = args.coco_json
    images_dir = args.images_dir if args.images_dir else (coco_json.parent / "images")
    return coco_json, images_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", type=Path, default=None)
    ap.add_argument("--images_dir", type=Path, default=None)
    ap.add_argument("--zip_path", type=Path, default=None)
    ap.add_argument("--archive_dir", type=Path, default=None)
    ap.add_argument("--archive_pattern", type=str, default="*.zip")
    ap.add_argument("--use_newest_zip", action="store_true")
    ap.add_argument(
        "--write_dedup_json",
        action="store_true",
        help="When loading through archive flow, export deduplicated COCO JSON to output_dir and evaluate with it.",
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model_size", type=str, default="large", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--output_dir", type=Path, default=Path("wyniki/rfdetr_baseline"))
    ap.add_argument("--protocol", type=str, default="full_dataset_eval")
    ap.add_argument("--iou_match", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    cfg = DetectionEvalConfig(
        iou_match_threshold=args.iou_match,
        confidence_threshold=args.conf,
        run_label=args.protocol,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(args.output_dir / "eval_config.json")

    coco_json, images_dir = _resolve_dataset_paths(args)

    coco = COCO(str(coco_json))
    cats = _sorted_categories(coco)
    cat_ids = [c["id"] for c in cats]
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_cat_id = cat_ids
    num_classes = len(cat_ids)

    model = load_rfdetr(args.model_size, args.checkpoint)  # type: ignore[arg-type]
    model_class_names = list(model.class_names) if model.class_names else [str(i) for i in range(num_classes)]
    
    # Handle class mismatch: if model trained on fewer classes than COCO dataset
    # Filter to only evaluate on classes the model knows about
    if len(model_class_names) < num_classes:
        logger.info(f"Model has {len(model_class_names)} classes but COCO has {num_classes}")
        logger.info(f"Model classes: {model_class_names}")
        logger.info(f"Filtering to only evaluate model's known classes")
        # Use only model's class names for evaluation
        class_names = model_class_names
        # Filter COCO categories to match model classes
        filtered_cat_ids = cat_ids[:len(model_class_names)]
        idx_to_cat_id = filtered_cat_ids
        cat_id_to_idx = {cid: i for i, cid in enumerate(filtered_cat_ids)}
        num_classes = len(model_class_names)
    else:
        class_names = model_class_names

    preds, _ = _coco_predictions_for_images(
        model, coco, images_dir, cfg.confidence_threshold, idx_to_cat_id
    )

    coco_gt = coco
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    # Restrict evaluation to only the categories the model was trained on,
    # so mAP is not averaged over other categories with AP=0.
    ev.params.catIds = idx_to_cat_id
    # Also restrict to images that have GT annotations for these categories,
    # matching the training evaluation protocol.
    ev.params.imgIds = coco_gt.getImgIds(catIds=idx_to_cat_id)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    map_5095 = float(ev.stats[0])
    map_50 = float(ev.stats[1])

    f1_counts = _accumulate_f1_matches(model, coco, images_dir, cfg, cat_id_to_idx, num_classes)
    f1_per_class = {class_names[i]: _f1(**f1_counts[i]) for i in range(num_classes)}

    summary = {
        "protocol": args.protocol,
        "checkpoint": str(args.checkpoint.resolve()),
        "coco_json": str(coco_json.resolve()),
        "images_dir": str(images_dir.resolve()),
        "mAP_50_95": map_5095,
        "mAP_50": map_50,
        "f1_per_class": f1_per_class,
        "class_names": class_names,
    }
    (args.output_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "baseline_metrics_per_class.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "f1"])
        for name, sc in f1_per_class.items():
            w.writerow([name, f"{sc:.6f}"])
        w.writerow(["__mAP_50__", f"{map_50:.6f}"])
        w.writerow(["__mAP_50_95__", f"{map_5095:.6f}"])

    print(json.dumps({"wrote": str(args.output_dir), "mAP_50": map_50, "mAP_50_95": map_5095}, indent=2))


if __name__ == "__main__":
    main()
