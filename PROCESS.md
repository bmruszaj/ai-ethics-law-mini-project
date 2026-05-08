# Dokumentacja procesu

Ten plik dokumentuje **jak** pracowałem/am nad mini-projektem — jakie narzędzia AI wykorzystałem/am, jakie prompty pisałem/am, jakie decyzje podjąłem/łam i co nie zadziałało.

> **PROCESS.md jest tak samo ważny jak kod.** Prowadzący ocenia świadome korzystanie z narzędzi AI — to jest kurs o aspektach AI.

---

## Narzędzia AI

| Narzędzie | Do czego używałem/am |
|-----------|----------------------|
| Cursor (IDE + agent) | Refaktoryzacja repozytorium, workflow OpenSpec (`/opsx-propose`, `/opsx-apply`, `/opsx-archive`), moduły RF-DETR + XAI |
| **ChatGPT 5.5** | Eksploracja idei XAI dla RF-DETR: literatura, metodologia per-detection, D-RISE, Attention Evidence Map, CAM-y, SSGrad-CAM++, metryki (EF, EE, PG), wybór warstw — **poniżej moje prompty i konteksty** z wieloetapowego czatu (bez outputu modelu) |

---

## Prompty

> Nie wklejaj outputu z AI — tylko prompty, które wpisywałeś/aś.

**Kategoria: OpenSpec — propozycja workflow RF-DETR + XAI**

```
/opsx-propose Zaprojektuj w repozytorium: detekcja RF-DETR + trzy metody XAI
(EigenCAM, GradCAM++, D-RISE) na poziomie bboxów i map saliencji; metryki wyjaśnień
EBPG, mIoU (Otsu) oraz Δp względem GT; eksperymenty i baseline mają być spójne
z sąsiednim repozytorium dermatoscopy_ai (checkpoint, adnotacje COCO, opcjonalnie PYTHONPATH do dermato_ai).
```

**Kontekst:** Ustalenie zakresu kodu i artefaktów pod RF-DETR oraz trzy konkretne metody wyjaśnień, bez narracji o „przejściu” z innego typu modelu — chodziło o domknięcie ścieżki detekcji + XAI i raportowanie w `wyniki/` zgodnie z kryteriami kursu.

**Kategoria: OpenSpec — implementacja**

```
/opsx-apply
```

**Kontekst:** Implementacja pod eksperymenty RF-DETR + XAI — skrypty baseline, CAM, D-RISE, metryki detekcyjne (wariant archiwum OpenSpec użyty w trakcie pracy).

**Kategoria: ChatGPT 5.5 — eksploracja idei (wieloetapowy czat)**

```
Czy ktoś już robił XAI na RF-DETR? Jakie są najbliższe kierunki w literaturze (DETR, D-RISE, attention/CAM)?
```

**Kontekst:** Wstęp do tematu: mało gotowych wdrożeń pod RF-DETR; wstępna orientacja: Attention Evidence Map (styl AttentionCAM++), D-RISE, CAM-y jako baseline’y.

```
Robię Deformable Attention Evidence Map na RF-DETR — czy wyjaśniać zawsze query 0, czy query powiązane z konkretną detekcją? Na surowym top-k widzę niskie score (np. query=1, score=0.0125).
```

**Kontekst:** Przejście z arbitralnego query na **predict() → IoU match detekcji do raw query → explain matched query**.

```
Jak ustawić D-RISE pod RF-DETR — globalnie czy per detekcja? Jak zdefiniować target score i jak pisać o IoU matching bez przesady?
```

**Kontekst:** D-RISE **osobno na detekcję**, score ∝ pewność × IoU z boxem; język *IoU-weighted matching reduces target switching*; walidacja przez PG, EF, EE, ins/del, overlap z ekspertem zamiast „mean saliency confirms unique explanations”.

```
Po co w ogóle D-RISE i mapy attention, skoro bbox już pokazuje predykcję?
```

**Kontekst:** Ustalenie narracji: bbox = *co i gdzie*, XAI = *na podstawie czego*; D-RISE vs Attention Evidence Map jako dwa różne „dlaczego”.

```
Potrzebuję kodu GradCAM++ pod RF-DETR: predict → match do raw query → target na logit klasy → backward do warstwy → mapa. Wyniki: PG często False, EF ~0, matched IoU wysokie — co z tego wynika?
```

**Kontekst:** Potwierdzenie, że **matching query działa**, ale GradCAM++ na wybranej warstwie jest **słabym** XAI — zostaje jako baseline, nie jako główna metoda.

```
To samo dla EigenCAM — jak to się ma do per-detection i jak interpretować bardzo niski EF?
```

**Kontekst:** EigenCAM **image-level**, słabo per-detection; rola **gradient-free baseline**, nie metoda od dobrej lokalizacji.

```
Którą warstwę brać pod CAM w RF-DETR — projector, stages, cv2? Co znaczy n=4 w tabeli sweepu vs wybór warstwy?
```

**Kontekst:** Kryterium aktywacji [B,C,H,W]; nie mylić liczby detekcji w raporcie z liczbą warstw; sens „całego projektora” vs warstw wewnętrznych.

```
Chcę Spatial Sensitive Grad-CAM++ z repo Spatial-Sensitive-Grad-CAM pod ten sam pipeline (per-detection). Porównaj z klasycznymi CAM pod kątem PG, EF, EE.
```

**Kontekst:** **SSGrad-CAM++** jako przełom — stabilne mapy (np. EF≈0.6, wysokie EE, wysokie `good_pct`); zmiana narracji: klasyczne CAM słabe, SSGrad mocny baseline.

```
Czym dokładnie różni się Energy Fraction od Energy Enrichment i jak etykietować jakość (good/partial/weak/failed) gdy PG=False ale EF/EE wysokie? Mam sweep na ~87 detekcjach — mam EF/PG/good_pct/failed_pct dla m.1.cv2, stages.0.0.cv2 i całego projektora. Którą warstwę wybrać finalnie, a którą zostawić jako referencyjną?
```

**Kontekst:** EF vs EE przy małych strukturach; sweep: `m.1.cv2` (wyższy mean EF), **`stages.0.0.cv2`** (PG≈0.954, good_pct≈0.943, failed_pct=0) → **wybór finalny**; `backbone.0.projector` jako **warstwa referencyjna / ablacja**.

```python
# Konwencja z czatu / notebooka SSGrad; w src odpowiednik produkcyjny — patrz
# DEFAULT_CAM_TARGET_LAYER_NAME w rfdetr_gradcampp_custom.py (ta sama ścieżka).
SSGRADCAM_TARGET_LAYER_NAME = "backbone.0.projector.stages.0.0.cv2"
CAM_REFERENCE_LAYER_NAME = "backbone.0.projector"
```

**Kontekst (podsumowanie wątku):** finalny trójpak **D-RISE + Attention Evidence Map + SSGrad-CAM++**, przy słabszych baseline’ach GradCAM++ i EigenCAM — bez pełnego transcriptu, tylko moje pytania i ustalenia.

---

## Decyzje

1. **RF-DETR jako model referencyjny** — trening w `dermatoscopy_ai`; w tym repozytorium: baseline, wyjaśnienia i metryki na bboxach. 

2. **Warstwa docelowa CAM (EigenCAM / GradCAM++ w `src/`)** — Kanoniczna ścieżka w `rfdetr_gradcampp_custom.py`: `DEFAULT_CAM_TARGET_LAYER_NAME = "backbone.0.projector.stages.0.0.cv2"`, z fallbackiem na ostatni blok `C2f.cv2` w `MultiScaleProjector`, jeśli ścieżki brak w grafie (RF-DETR 1.6.x). Ten wybór jest **zgodny** z empirycznym sweep’em z eksploracji (ChatGPT + `notebooks/07_SSGradpp.ipynb`). Osobno w notebooku raportuję warstwę **referencyjną** `backbone.0.projector` (narracja / ablacja) — to nie zastępuje domyślnej ścieżki w `src`, tylko ją uzupełnia w eksperymentach.

3. **Normalizacja mapy pod EBPG** — min-max do [0, 1], potem stosunek sumy w bbox GT do sumy globalnej (`src/rfdetr_detection_metrics.py`). Dla starszych metryk w `src/xai_metrics.py`: min-max oraz wariant IoU z rozkładem (suma = 1); insertion/deletion w demie numpy to **proxy** bez forwardu sieci (wersja z modelem: `src/xai_ins_del.py`).

4. **Baseline losowy dla Δp** — `random_bbox_same_area` (losowy prostokąt o tej samej powierzchni co GT).

5. **Dwa „arkusze” metryk** — demo numpy + toy ViT (RISE, surrogate LeGrad/Chefer) osobno od pakietu EBPG / mIoU / Δp na RF-DETR, żeby nie mieszać definicji w raporcie (`README.md`).

6. **Co trafia do GitHub, a co zostaje lokalnie** — `.cursor/`, `.opencode/` w `.gitignore`; `wyniki/` domyślnie ignorowane z wyjątkiem `wyniki/README.md` i `wyniki/xai_demo/*` (mały, reprodukowalny zestaw pod zaliczenie); duże wyniki RF-DETR tylko lokalnie plus opis ścieżki w `README.md` / `EXPERIMENTS.md`.

7. **Eksploracja metod z ChatGPT 5.5 vs stan repozytorium** — W czacie ustaliłem/am m.in. narrację „bbox vs XAI”, regułę `predict() → IoU → explain matched query`, per-detection D-RISE oraz **metodologiczny** trójpak **D-RISE + Attention Evidence Map + SSGrad-CAM++** (przy słabszych baseline’ach GradCAM++ i EigenCAM). **W kodzie tego repo:** moduły w `src/` obejmują m.in. baseline, D-RISE, EigenCAM/GradCAM++ i metryki bbox; **Attention Evidence Map** oraz **SSGrad-CAM++** są rozwijane i dokumentowane przede wszystkim w **notebookach** (`notebooks/04_v3_attention_evidence_map.ipynb`, `notebooks/07_SSGradpp.ipynb` — patrz też `notebooks/WNIOSKI_ANALIZA_NOTEBOOKOW_XAI.md`). W sekcji Prompty używam nazw `SSGRADCAM_TARGET_LAYER_NAME` / `CAM_REFERENCE_LAYER_NAME` jako **etykiety z eksploracji**; w Pythonie `src` odpowiada im **`DEFAULT_CAM_TARGET_LAYER_NAME`** (ta sama ścieżka `…stages.0.0.cv2`) oraz odrębne porównanie z `backbone.0.projector` w notebooku.

---

## Co nie zadziałało

1. **`/opsx-apply` (OpenSpec) — kod z automatu nie był „produkcyjnie gotowy”** — pipeline ze specyfikacji generował szkice oparte na **ogólnych** wzorcach (np. podpięcie `pytorch-grad-cam` „jak do ResNeta”), bez domknięcia RF-DETR: inny graf, `LWDETR` / projektor, rozjazd **`predict()` vs surowe query**, wybór warstwy docelowej. **Obejście:** ręczna iteracja — czytanie forwardu, hooki, własne cele i dopasowanie detekcji do query.

2. **RF-DETR jako świeży detektor — brak gotowców pod XAI** — w praktyce **nie ma** biblioteki typu „`explain(rfdetr)`”; większość tutoriali CAM/D-RISE dotyczy klasyfikacji lub starszych detektorów. Trzeba było **tłumaczyć pomysły z literatury** (DETR, D-RISE, attention) na **konkretny kod** pod Roboflow RF-DETR 1.6.x.

3. **„Zwykłe” metody gradientowe (GradCAM++, EigenCAM) — słaba lokalizacja na RF-DETR** — nawet przy **poprawnym** matchowaniu query wyniki były często słabe (PG, EF), podczas gdy IoU dopasowania było wysokie — sygnał, że problemem nie jest tylko „złe query”, ale **forma mapy** (EigenCAM dodatkowo jest raczej **image-level**). **Obejście:** **Spatial Sensitive Grad-CAM++** (mapa czułości przestrzennej × Grad-CAM++) — w repo głównie w `notebooks/07_SSGradpp.ipynb`.

4. **D-RISE — koszt i tuning** — sensowny wariant wymaga wielu masek i czasu GPU; łatwo o **niedoszacowanie** złożoności przy pierwszych prototypach (globalna mapa zamiast per-detection). Wymusza to **per-detection** target i świadome ograniczenie budżetu obliczeń w eksperymentach.

---

## Iteracje

Kolejne wersje odpowiadają **faktycznej ścieżce** z sekcji **Prompty** i **Co nie zadziałało** (nie tylko „numer wersji w głowie”).

1. **v0 — szablon kursu** — przykłady API LLM, pusty szkielet opisu; punkt wyjścia przed własnym tematem XAI.

2. **v1 — kontekst i pierwsza specyfikacja** — materiały w `docs/context/` (propozycja, metody, ewaluacja, ZASADY zaliczenia); temat **RF-DETR + XAI** — doprecyzowanie wymagań kursu i pierwsze sformalizowanie zakresu (w tym OpenSpec) pod detekcję i trzy metody wyjaśnień.

3. **v2 — OpenSpec pod RF-DETR (`/opsx-propose`, `/opsx-apply`)** — formalny zakres: RF-DETR + EigenCAM / GradCAM++ / D-RISE, metryki EBPG / mIoU / Δp, spójność z `dermatoscopy_ai`. **`/opsx-apply` nie dostarczył gotowca** — szablony z bibliotek XAI nie mapowały się na graf RF-DETR (*Co nie* pkt 1) → **ręczne** dokończenie: hooki, cele, warstwy projektora, `predict()` vs surowe query.

4. **v3 — eksploracja z ChatGPT 5.5** — wieloetapowy czat: literatura (brak gotowców pod RF-DETR — *Co nie* pkt 2), reguła **`predict() → IoU → explain matched query`**, sens **bbox vs XAI**, D-RISE per-detection, słabość **GradCAM++ / EigenCAM** i przejście na **SSGrad-CAM++**, EF vs EE, sweep warstw — patrz **Prompty → ChatGPT 5.5** i *Co nie* pkt 3.

5. **v4 — implementacja „po boju”** — `src/`: baseline, D-RISE, EigenCAM/GradCAM++, metryki bbox (`rfdetr_gradcampp_custom`, `DEFAULT_CAM_TARGET_LAYER_NAME`); **notebooki** dla Attention Evidence Map i SSGrad (`04_v3_attention_evidence_map.ipynb`, `07_SSGradpp.ipynb`, `WNIOSKI_ANALIZA_NOTEBOOKOW_XAI.md`). **LeGrad** na pełnym transformerze świadomie **nie** realizowany (*Co nie* pkt 7).

6. **v5 — D-RISE i infrastruktura** — iteracje na **koszt obliczeń**, per-detection target, ograniczenie budżetu masek (*Co nie* pkt 4); równolegle frustrujące **`uv sync` / PyPI** na części środowisk (*Co nie* pkt 6).

7. **v6 — dokumentacja i oddanie** — `EXPERIMENTS.md`, README z checklistą ZASADY, PROCESS z promptami i „porażkami”, `pyproject.toml` + `uv.lock` zsynchronizowane ze `dermatoscopy_ai`, `wyniki/xai_demo/` w repo, uruchomienie przez `.venv/bin/python` (bez `uv run`), usunięcie mylących placeholderów „treningu” z dem.

**Czas pracy (opcjonalnie):** 20h
