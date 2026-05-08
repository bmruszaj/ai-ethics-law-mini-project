# Katalog wyników (`wyniki/`)

## Co jest zacommitowane (minimalnie pod zaliczenie)

Folder **`xai_demo/`** — deterministyczne wyjście z `src/demo_xai_metrics.py` (metryki na **sztucznych** mapach saliencji; to nie są wyniki treningu RF-DETR na danych klinicznych).

Regeneracja (po `uv sync`, bez `uv run`):

```bash
.venv/bin/python src/demo_xai_metrics.py
```

Pliki: `metrics_demo.csv`, `method_comparison_demo.csv`, `overlay_method_*.svg`, opcjonalnie `saliency_overlay_demo.png` (gdy jest `matplotlib`).

## Co trzymasz lokalnie (nie commituj)

- Checkpointy (`.pt`), duże zbiory, `wyniki/rfdetr_baseline/`, `wyniki/rfdetr_xai/` itd. — zgodnie z `.gitignore`.
- Po własnych eksperymentach: opisz ścieżkę reprodukcji w `README.md` / `EXPERIMENTS.md`; nie wrzucaj plików >10 MB (por. `AGENTS.md`).

## Powiązanie z oceną

[ZASADY_ZALICZENIA_MINI_PROJEKT.md](../docs/context/ZASADY_ZALICZENIA_MINI_PROJEKT.md) wymaga **efektów pracy** i możliwości **odtworzenia** wyników — `xai_demo/` spełnia warstwę techniczną „out of the box”; pełna ścieżka RF-DETR + XAI jest w `README.md` i notebookach.
