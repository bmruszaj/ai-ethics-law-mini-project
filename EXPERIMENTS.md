# Przewodnik eksperymentów

Ten dokument zbiera **kolejność kroków**, **wymagane zależności** i **które pliki** otwierać lub uruchamiać. Szczegóły kursu i etyki nadal w `README.md` oraz `PROCESS.md`.

---

## 1. Środowisko

| Cel | Komenda / ustawienie |
|-----|----------------------|
| Rdzeń (jak `dermatoscopy_ai`: torch+cu128, rfdetr, COCO stack, API LLM) + grupa **dev** | `uv sync` |
| Notebooki | `uv sync --extra notebooks` |
| scipy (arkusz pomocniczy / toy ViT) | `uv sync --extra xai` |
| CAM, D-RISE, supervision, tqdm (`grad-cam`, `vision-explanation-methods`) | `uv sync --extra rfdetr-xai` |

Po instalacji uruchamiaj skrypty przez **`.venv/bin/python`** (bez `uv run`), np. `.venv/bin/python src/demo_xai_metrics.py`.

**Repozytorium sąsiednie:** trening i inference RF-DETR często żyją w `../dermatoscopy_ai`. Import `dermato_ai.*` wymaga dodania tego katalogu na `PYTHONPATH` (opis w `README.md`). Checkpoint wczytujesz lokalnie — patrz `src/rfdetr_load.py`.

---

## 2. Baseline detektora (mAP, F1)

**Cel:** policzyć **mAP@50**, **mAP@50:95** i **F1 per klasa** na zbiorze COCO dla jednego checkpointu.

| Co | Plik |
|----|------|
| Skrypt CLI | `src/run_rfdetr_baseline_eval.py` |
| Ładowanie modelu | `src/rfdetr_load.py` |
| Progi IoU / confidence (JSON) | `src/rfdetr_eval_config.py` → zapis `eval_config.json` |
| Wyniki | `wyniki/rfdetr_baseline/` (`baseline_metrics.json`, `baseline_metrics_per_class.csv`) |

**Uruchomienie:** patrz przykład w docstringu skryptu lub sekcja RF-DETR w `README.md`.

**Uwaga:** domyślny protokół w skrypcie to **ewaluacja na całym zbiorze z JSON** (`full_dataset_eval`) — na zbiorze treningowym metryki bywają optymistyczne; zapisz w raporcie, jaki split faktycznie używasz.

---

## 3. Wyjaśnienia oparte na gradientach (EigenCAM, GradCAM++)

**Cel:** mapy uwagi dla wybranej pary GT–pred / bboxu i klasy modelu.

| Co | Plik |
|----|------|
| Warstwa docelowa + uruchomienie CAM + zapis PNG + JSON metadanych | `src/rfdetr_gradcam.py` |
| Cel gradientu (query z max IoU do bboxu, score klasy) | `src/rfdetr_detection_target.py` |
| Notebook (szablon ścieżki) | `notebooks/rfdetr_xai_gradcam.ipynb` |
| Wyniki | np. `wyniki/rfdetr_xai/` (ustaw ścieżki w notebooku / kodzie) |

---

## 4. D-RISE (czarna skrzynka)

**Cel:** mapy saliencji z pakietu Microsoft **vision-explanation-methods**.

| Co | Plik |
|----|------|
| Wrapper pod interfejs D-RISE | `src/rfdetr_drise_wrapper.py` |
| Uruchomienie N masek, tryb szybki, opcjonalny cache | `src/rfdetr_drise_run.py` |

Domyślnie: `num_masks=1000`, `mask_res=(8, 8)`; debug: `fast_mode` (mniej masek). Losowość masek — dla powtarzalności ustaw `torch.manual_seed` albo użyj `cache_dir` (zapis w `rfdetr_drise_run.py`).

---

## 5. Metryki map vs bbox (EBPG, mIoU Otsu, Δp)

**Cel:** liczby na mapach `S` względem prostokąta GT oraz **wiarygodność Δp** (inpainting GT vs losowy box).

| Co | Plik |
|----|------|
| EBPG, mIoU (Otsu), resize mapy, inpainting, Δp / Δp_rand | `src/rfdetr_detection_metrics.py` |
| Agregacja **metoda × klasa** (średnie, %Δp>0) z CSV | `src/aggregate_rfdetr_xai_metrics.py` |

**Wejście do agregacji:** własny CSV z kolumnami m.in. `method`, `class_id` lub `class_name`, `ebpg`, `miou`, `delta_p` (wiersze generujesz po pętli po obrazach / metodach).

---

## 6. Figury (nakładki bbox + heatmapa)

| Co | Plik |
|----|------|
| PNG z matplotlib | `src/generate_rfdetr_xai_figures.py` |

Wymaga obrazu źródłowego; opcjonalnie `.npy` z mapą oraz bbox GT/pred (`--help`).

---

## 7. Materiały pomocnicze (nie jako główny pakiet RF-DETR)

| Cel | Pliki |
|-----|--------|
| Metryki numpy / demo SVG bez detektora | `src/demo_xai_metrics.py`, `src/xai_metrics.py` |
| Insertion/deletion z forwardem (wymaga modelu w kodzie) | `src/xai_ins_del.py` |
| Toy ViT + RISE + surrogate LeGrad/Chefer | `src/run_xai_three_methods_demo.py`, `src/xai_toy_vit.py`, `src/xai_rise_torch.py` |

Traktuj je jako **osobny arkusz wyników** — definicje mogą różnić się od EBPG / mIoU / Δp (`README.md`: sekcja arkusza dodatkowego).

---

## 8. Sugerowana kolejność dnia eksperymentalnego

1. `uv sync --extra rfdetr-xai` (+ notebooki, jeśli potrzebne).  
2. Baseline: `run_rfdetr_baseline_eval.py` → `wyniki/rfdetr_baseline/`.  
3. Na wybranych obrazach: CAM (`rfdetr_gradcam.py` lub notebook).  
4. (Opcjonalnie, czas GPU) D-RISE: `rfdetr_drise_run.py`.  
5. Metryki na zapisanych mapach: `rfdetr_detection_metrics.py` → wiersze CSV → `aggregate_rfdetr_xai_metrics.py`.  
6. Figury do obrony: `generate_rfdetr_xai_figures.py`.  
7. Uzupełnij `README.md` / `PROCESS.md` (wyniki, prompty, decyzje).

---

## 9. Dokumentacja kontekstowa

Szerszy kontekst metryk i formatów: `docs/context/` (np. `EWALUACJA.md`, `RF_DETR_PREDICTIONS_FORMAT.md`, `PROPONYCJA_PROJEKTU.md`).
