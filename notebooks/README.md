# Notebooki — eksperymenty XAI (po treningu)

## Kolejność pracy

1. **RF-DETR i notebook CAM** — checkpoint z ``dermatoscopy_ai``, potem [rfdetr_xai_gradcam.ipynb](rfdetr_xai_gradcam.ipynb) lub własny `.ipynb`; mapy i CSV do `../wyniki/rfdetr_xai/`.
2. **D-RISE / LeGrad / toy ViT** — starsze ścieżki pozostają możliwe; RF-DETR jest ścieżką referencyjną OpenSpec.
3. **Artefakty** — duże wyniki RF-DETR trzymaj **lokalnie** (`wyniki/rfdetr_*` jest ignorowane). Do zaliczenia w repozytorium jest zacommitowany minimalny zestaw **`wyniki/xai_demo/`** (regeneracja: `src/demo_xai_metrics.py`); własne eksperymenty opisz w README i ewentualnie dołącz małe pliki przez dopisanie wyjątków w `.gitignore`.

## Zależności

```bash
uv sync --extra notebooks
uv sync --extra rfdetr-xai   # RF-DETR + CAM + D-RISE
```

## Dlaczego notebook, a nie długi skrypt?

- narracja i **komentarz krok po kroku** dla prowadzącego i dla Ciebie za pół roku,
- **wykresy** bez rozklepywania CLI,
- łatwe **porównanie wariantów** (osobne sekcje / podnotebooki).

Logikę wartościową (metryki, ins/del z forwardem) trzymaj w **`src/`** i importuj w komórkach — wtedy spełniasz kryterium „czytelny, uruchamialny kod” bez duplikacji.
