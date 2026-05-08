#!/usr/bin/env python3
"""
Generuje nakładki PNG: obraz + bbox (GT/pred) + mapa saliencji (heatmap).

Uruchom po zapisaniu map z ``rfdetr_gradcam.run_cam_and_save`` lub własnych tablic ``numpy``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rfdetr_detection_metrics import resize_s_to_hw
from xai_metrics import normalize_saliency_minmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--heatmap_npy", type=Path, help="Opcjonalnie mapa float zapisana jako .npy")
    ap.add_argument("--bbox_gt", type=float, nargs=4, metavar=("x1", "y1", "x2", "y2"))
    ap.add_argument("--bbox_pred", type=float, nargs=4, metavar=("x1", "y1", "x2", "y2"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pil = Image.open(args.image).convert("RGB")
    rgb = np.asarray(pil, dtype=np.float64) / 255.0
    h, w = rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)

    if args.heatmap_npy and args.heatmap_npy.is_file():
        s = np.load(args.heatmap_npy)
        s = resize_s_to_hw(s, h, w)
        heat = normalize_saliency_minmax(s)
        ax.imshow(heat, cmap="jet", alpha=0.45)

    if args.bbox_gt:
        x1, y1, x2, y2 = args.bbox_gt
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2, label="GT"))
    if args.bbox_pred:
        x1, y1, x2, y2 = args.bbox_pred
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", linewidth=2, label="Pred"))

    ax.axis("off")
    if args.bbox_gt or args.bbox_pred:
        ax.legend(loc="upper right")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Zapisano {args.out}")


if __name__ == "__main__":
    main()
