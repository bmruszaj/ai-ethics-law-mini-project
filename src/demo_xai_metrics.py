#!/usr/bin/env python3
"""
Demo: metryki XAI (numpy) + tabela porównawcza + figury SVG (bez matplotlib).

Uruchomienie:
    python3 src/demo_xai_metrics.py

Jeśli ``matplotlib`` jest zainstalowany, dodatkowo zapisze PNG z ``matplotlib``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from xai_metrics import (
    deletion_auc_from_scores,
    energy_fraction_inside_bbox,
    energy_iou_distribution_vs_mask,
    insertion_auc_from_scores,
    pointing_game_hit,
    proxy_insertion_deletion_scores_from_saliency,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "wyniki" / "xai_demo"


def make_three_synthetic_maps(h: int, w: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    y, x = np.ogrid[:h, :w]
    cx, cy = 40, 22

    rise_like = rng.random((h, w)) * 0.4
    rise_like += 1.2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 10**2))

    le_like = np.abs(x - cx) + np.abs(y - cy)
    le_like = 1.0 / (1.0 + le_like * 0.08)

    ch = (np.sin(x * 0.35) * np.sin(y * 0.35)) ** 2
    ch = ch + 2.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 5**2))

    return {"rise_like_noise": rise_like, "legrad_like_radial": le_like, "chefer_like_grid": ch}


def saliency_to_svg_data_uri(s: np.ndarray) -> str:
    """Szary obrazek H×W jako mały PNG base64 przez zlib — uproszczenie: użyj skali kolorów SVG."""
    s = np.asarray(s, dtype=np.float64)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    h, w = s.shape
    cells = []
    step = max(1, h // 32)
    for yy in range(0, h, step):
        for xx in range(0, w, step):
            v = float(s[yy : yy + step, xx : xx + step].mean())
            g = int(255 * (1.0 - v))
            r = int(255 * v)
            b = 120
            cells.append(
                f'<rect x="{xx * 4}" y="{yy * 4}" width="{step * 4}" height="{step * 4}" '
                f'fill="rgb({r},{g},{b})" stroke="none"/>'
            )
    return "\n".join(cells)


def write_svg_overlay(
    smap: np.ndarray,
    bbox: tuple[int, int, int, int],
    path: Path,
    title: str,
) -> None:
    h, w = smap.shape
    scale = 4
    W, H = w * scale, h * scale
    x1, y1, x2, y2 = bbox
    mx = int(np.argmax(smap) % w) * scale + scale // 2
    my = int(np.argmax(smap) // w) * scale + scale // 2
    body = saliency_to_svg_data_uri(smap)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="#111"/>
  <g>{body}</g>
  <rect x="{x1 * scale}" y="{y1 * scale}" width="{(x2 - x1) * scale}" height="{(y2 - y1) * scale}"
        fill="none" stroke="cyan" stroke-width="3"/>
  <circle cx="{mx}" cy="{my}" r="5" fill="yellow" opacity="0.85"/>
  <text x="8" y="20" fill="white" font-size="14" font-family="sans-serif">{title}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    h, w = 64, 64
    bbox = (30, 15, 50, 35)

    maps = make_three_synthetic_maps(h, w, rng)

    compare_rows = []
    for name, s in maps.items():
        pg = pointing_game_hit(s, bbox)
        ef = energy_fraction_inside_bbox(s, bbox)
        eiou = energy_iou_distribution_vs_mask(s, bbox)
        ins_c, del_c = proxy_insertion_deletion_scores_from_saliency(s, n_steps=24)
        ins_auc = insertion_auc_from_scores(ins_c)
        del_auc = deletion_auc_from_scores(del_c)
        compare_rows.append(
            {
                "method_synthetic": name,
                "pointing_game_hit": int(pg),
                "energy_fraction_in_bbox": f"{ef:.4f}",
                "energy_iou_dist_mask": f"{eiou:.4f}",
                "insertion_auc_proxy": f"{ins_auc:.4f}",
                "deletion_auc_proxy": f"{del_auc:.4f}",
            }
        )

    csv_path = OUT / "metrics_demo.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(
            f,
            fieldnames=[
                "pointing_game_hit",
                "energy_fraction_in_bbox",
                "energy_iou_dist_mask",
                "insertion_auc_proxy",
                "deletion_auc_proxy",
            ],
        )
        wtr.writeheader()
        r0 = compare_rows[0]
        wtr.writerow(
            {
                "pointing_game_hit": r0["pointing_game_hit"],
                "energy_fraction_in_bbox": r0["energy_fraction_in_bbox"],
                "energy_iou_dist_mask": r0["energy_iou_dist_mask"],
                "insertion_auc_proxy": r0["insertion_auc_proxy"],
                "deletion_auc_proxy": r0["deletion_auc_proxy"],
            }
        )

    compare_path = OUT / "method_comparison_demo.csv"
    with compare_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(
            f,
            fieldnames=[
                "method_synthetic",
                "pointing_game_hit",
                "energy_fraction_in_bbox",
                "energy_iou_dist_mask",
                "insertion_auc_proxy",
                "deletion_auc_proxy",
            ],
        )
        wtr.writeheader()
        wtr.writerows(compare_rows)

    for i, (name, smap) in enumerate(maps.items()):
        write_svg_overlay(smap, bbox, OUT / f"overlay_method_{i+1}_{name}.svg", name)

    # Opcjonalnie PNG przez matplotlib
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        s0 = list(maps.values())[0]
        ins_c, del_c = proxy_insertion_deletion_scores_from_saliency(s0, n_steps=24)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].imshow(s0, cmap="magma")
        x1, y1, x2, y2 = bbox
        axes[0].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", linewidth=2)
        )
        axes[0].set_title("Saliencja (demo) + bbox")
        axes[0].axis("off")
        axes[1].plot(np.linspace(0, 1, len(ins_c)), ins_c, label="Insertion (proxy P)")
        axes[1].plot(np.linspace(0, 1, len(del_c)), del_c, label="Deletion (proxy P)")
        axes[1].set_xlabel("Ułamek kroków")
        axes[1].set_ylabel("Ułamek masy P w widocznym zbiorze")
        axes[1].legend()
        axes[1].set_title("Krzywe pod AUC (proxy)")
        fig.tight_layout()
        fig.savefig(OUT / "saliency_overlay_demo.png", dpi=150)
        plt.close(fig)
        print(f"Zapisano: {OUT / 'saliency_overlay_demo.png'}")
    except ImportError:
        pass

    print(f"Zapisano: {csv_path}")
    print(f"Zapisano: {compare_path}")
    print(f"Zapisano: {OUT}/overlay_method_*.svg")


if __name__ == "__main__":
    main()
