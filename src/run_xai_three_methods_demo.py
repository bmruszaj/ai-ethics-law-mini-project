#!/usr/bin/env python3
"""
Proof-of-concept: trzy metody XAI na minimalnym ViT + metryki z ``xai_metrics``.

- **RISE** — black-box (losowe maski).
- **LeGrad (surrogate)** — Grad×Input względem pikseli wejścia ( uproszczenie wobec repo WalBouss/LeGrad ).
- **Chefer (rollout)** — rollout macierzy uwagi z jednej warstwy MHA (linia Transformer-MM-Explainability).

Uruchomienie (z opcjonalnym extra ``xai``):

    uv sync --extra xai
    uv run python src/run_xai_three_methods_demo.py

Wyniki: ``wyniki/xai_three_methods_demo/`` (CSV, kilka PNG).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

OUT = _ROOT / "wyniki" / "xai_three_methods_demo"


def main() -> None:
    try:
        import torch
    except ImportError as e:
        print(
            "Brak pakietu torch. Zainstaluj: uv sync --extra xai\n",
            e,
            file=sys.stderr,
        )
        sys.exit(1)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from xai_ins_del import insertion_deletion_aucs
    from xai_metrics import (
        energy_iou_distribution_vs_mask,
        pointing_game_hit,
    )
    from xai_rise_torch import rise_saliency_map
    from xai_toy_vit import TinyViTForXaiDemo, init_demo_weights

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(42)

    img_size, patch = 64, 8
    model = TinyViTForXaiDemo(img_size=img_size, patch_size=patch, dim=96, n_heads=4).to(device)
    init_demo_weights(model)
    model.eval()

    # Syntetyczny obraz z kontrastowym kwadratem (łatwy bbox)
    x = torch.rand(1, 3, img_size, img_size, device=device) * 0.15 + 0.4
    x[:, :, 20:44, 24:48] = torch.linspace(0.2, 0.95, 24, device=device).view(1, 1, 1, 24)
    x.requires_grad_(True)

    bbox_xyxy = (24, 20, 48, 44)  # x1,y1,x2,y2

    def forward_logits(inp: torch.Tensor) -> torch.Tensor:
        logits, _ = model(inp)
        return logits

    # --- LeGrad surrogate: grad * input (kanały sumowane do mapy H×W)
    logits, _ = model(x)
    target = logits.sum()
    model.zero_grad(set_to_none=True)
    target.backward(retain_graph=True)
    g = x.grad.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    x_det = x.detach()

    # --- Chefer-style rollout
    with torch.no_grad():
        _, attn = model(x_det)
    gh = gw = img_size // patch
    rollout = TinyViTForXaiDemo.attention_rollout_map(attn, (gh, gw)).squeeze(0).cpu().numpy()
    # rozciągnięcie patchy -> piksele (nearest)
    rollout_hw = np.kron(rollout, np.ones((patch, patch), dtype=np.float64))
    rollout_hw = rollout_hw[:img_size, :img_size]

    # --- RISE (bez gradientu wejścia)
    rise_map = rise_saliency_map(x_det, lambda t: model(t)[0], n_masks=48, p_keep=0.5, seed=1)
    rise_np = rise_map.cpu().numpy()

    maps = {
        "legrad_gradxinput": g,
        "chefer_rollout": rollout_hw,
        "rise": rise_np,
    }

    rows = []
    for name, smap in maps.items():
        pg = pointing_game_hit(smap, bbox_xyxy)
        eiou = energy_iou_distribution_vs_mask(smap, bbox_xyxy)
        ins_auc, del_auc = insertion_deletion_aucs(
            x_det, smap, lambda t: model(t)[0], n_steps=20
        )
        rows.append(
            {
                "method": name,
                "pointing_game_hit": int(pg),
                "energy_iou_dist_mask": f"{eiou:.4f}",
                "insertion_auc": f"{ins_auc:.4f}",
                "deletion_auc": f"{del_auc:.4f}",
            }
        )

    csv_path = OUT / "metrics_three_methods.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "pointing_game_hit",
                "energy_iou_dist_mask",
                "insertion_auc",
                "deletion_auc",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # Figury: 3 mapy + bbox
    for i, (name, smap) in enumerate(maps.items()):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(smap, cmap="magma")
        x1, y1, x2, y2 = bbox_xyxy
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", linewidth=2)
        )
        ax.set_title(name)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_{i+1}_{name}.png", dpi=120)
        plt.close(fig)

    # Krzywe przykładowe (RISE) — jedna dodatkowa figura
    from xai_ins_del import insertion_deletion_curves

    ins, dels = insertion_deletion_curves(x_det, rise_np, lambda t: model(t)[0], n_steps=20)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.linspace(0, 1, len(ins)), ins, label="Insertion")
    ax.plot(np.linspace(0, 1, len(dels)), dels, label="Deletion")
    ax.set_xlabel("Ułamek kroków")
    ax.set_ylabel("Logit (score)")
    ax.legend()
    ax.set_title("RISE — krzywe faithfulness (toy ViT)")
    fig.tight_layout()
    fig.savefig(OUT / "rise_ins_del_curves.png", dpi=120)
    plt.close(fig)

    print(f"Zapisano: {csv_path}")
    print(f"Zapisano figury w: {OUT}")


if __name__ == "__main__":
    main()
