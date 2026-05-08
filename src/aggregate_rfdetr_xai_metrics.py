#!/usr/bin/env python3
"""
Agregacja per metoda × klasa: średnia EBPG, mIoU, odsetek próbek z Δp > 0.

Wejście: CSV z kolumnami m.in. ``method``, ``class_id`` lub ``class_name``, ``ebpg``, ``miou``, ``delta_p``.
Wyjście: ``wyniki/rfdetr_xai/aggregated_by_method_class.csv`` (domyślnie).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    ap = argparse.ArgumentParser(description="Agregacja metryk XAI detektora.")
    ap.add_argument("--input_csv", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, default=Path("wyniki/rfdetr_xai/aggregated_by_method_class.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "delta_p" not in df.columns:
        raise SystemExit("CSV musi zawierać kolumnę ``delta_p``.")

    df["pct_delta_pos"] = (df["delta_p"] > 0).astype(float)

    group_cols = ["method"]
    if "class_name" in df.columns:
        group_cols.append("class_name")
    elif "class_id" in df.columns:
        group_cols.append("class_id")
    else:
        raise SystemExit("Brak ``class_name`` ani ``class_id``.")

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            ebpg_mean=("ebpg", "mean"),
            miou_mean=("miou", "mean"),
            pct_delta_p_positive=("pct_delta_pos", "mean"),
            n=("delta_p", "count"),
        )
        .reset_index()
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output_csv, index=False)
    print(f"Zapisano: {args.output_csv}")


if __name__ == "__main__":
    main()
