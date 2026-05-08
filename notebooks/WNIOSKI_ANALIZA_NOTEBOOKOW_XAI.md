# Wnioski z analizy notebooków XAI (RF-DETR, klasa „Yellow globules (ulcer)”)

Dokument zbiera **wnioski wyłącznie z zapisanych wykonań** notebooków: `02_baseline_eval.ipynb`, `03_gradcam_analysis.ipynb`, `04_drise_analysis.ipynb`, `04_v3_attention_evidence_map.ipynb`, `05_gradcampp.ipynb`, `06_eigencam.ipynb`, `07_SSGradpp.ipynb`.  
Wspólny kontekst: checkpoint `rfdetr_large_best_.../checkpoint_best_ema.pth`, COCO po deduplikacji (~217 obrazów, 10 kategorii w etykietach, model **1 klasy** — ewaluacja / XAI skupione na kategorii zgodnej z checkpointem), środowisko z logami wskazującymi m.in. na **CPU** dla części wag (DINOv2 przez Transformers).

---

## 1. `02_baseline_eval.ipynb` — baseline detekcji (mAP)

**Cel:** uruchomienie `run_rfdetr_baseline_eval.py` i zapis metryk COCO dla wybranej kategorii (zgodnie z logiem: filtrowanie do kategorii 9 = „Yellow globlues (ulcer)” przy 10 kategoriach w JSON).

**Wyniki liczbowe (z wyjścia notebooka):**

- **mAP@[0.50:0.95]** ≈ **0.646**
- **mAP@0.50** ≈ **0.955**
- AP dla obszarów: small ~0.306, medium ~0.611, large ~0.738
- AR (maxDets=100) ~0.679 (all), z rozbiciem na rozmiary bboxów

**Wnioski:**

- Model na wybranej klasie ma **bardzo wysokie AP@50**, co sugeruje dobrą separację przy luźniejszym IoU; pełniejszy próg 0.50:0.95 jest niższy — typowe dla trudniejszej lokalizacji lub zmienności granic.
- Małe obiekty (small) mają wyraźnie niższy AP niż large — przy interpretacji XAI warto pamiętać o **skali obiektu** i rozdzielczości map względem maski / bboxa.
- Ostrzeżenia RF-DETR (patch size, liczba klas w konfiguracji vs checkpoint) nie blokują inferencji, ale przy publikacji warto **jawnie opisać** `num_classes=1` i mapowanie kategorii COCO.

---

## 2. `03_gradcam_analysis.ipynb` — EigenCAM / GradCAM++ (`pytorch_grad_cam`)

**Cel:** eksploracja architektury (warstwa docelowa: `backbone[0][-1]` / `MultiScaleProjector`), wizualizacja na losowej próbie obrazów, porównanie EigenCAM vs GradCAM++.

**Obserwacje z wykonania:**

- Na **pierwszym** obrazie z pipeline nie było detekcji; na **10 losowych** obrazach: 6/10 miało detekcje — pokazuje **dużą częstość braku predykcji** przy użytym progu / próbie.
- **EigenCAM** w sekcji porównawczej **nie wygenerował map** dla 6/6 obrazów z powodu błędu PyTorch: *view created in no_grad … modified inplace with grad mode* — **0/6 sukcesów**; dalsza wizualizacja Eigen vs GradCAM++ padła z `ValueError` przy `plt.subplots(n, …)` dla `n=0`.
- **GradCAM** (identyfikacja warstwy) dla 5 obrazów zakończyła się sukcesem; zapisano m.in. `gradcam_results.png`.
- Podsumowanie drukowane po częściowym sukcesie: „EigenCAM: 5 images, Grad-CAM: 5 images” — **niespójne z wcześniejszym błędem EigenCAM** (część ścieżki działała na innej konfiguracji / kolejności komórek); **EigenCAM w tej ścieżce nie jest wiarygodny bez naprawy inplace / trybu grad**.

**Wnioski:**

- Notebook 03 jest przydatny jako **szkoleniowy / diagnostyczny**, ale **EigenCAM przez `pytorch_grad_cam` wymaga poprawy technicznej** zanim można go cytować jako wynik eksperymentalny.
- Własne notebooki **05–07** (implementacje bez zewnętrznego grad-cam) lepiej dokumentują **metryki i ograniczenia** niż ta wczesna komórka 03.

---

## 3. `04_drise_analysis.ipynb` — D-RISE (black-box, per-detekcja)

**Cel:** saliencja przez losowe maski; **target score** = ciągła miara (np. conf × IoU z bboxem docelowym), surowe mapy do metryk, normalizacja tylko do wizualizacji; krzywe deletion/insertion; zbieżność liczby masek.

**Wyniki / logi:**

- Przykładowy obraz: **4 detekcje** tej samej klasy.
- **Zbieżność (demo, krótki budżet masek):** dla harmonogramu 64 → 128 → 256 masek, korelacja Pearsona między kolejnymi budżetami: **~0.787** (128 vs 64), **~0.888** (256 vs 128) — rosnąca stabilność mapy przy większym N (pełny raport wymaga harmonogramu z dokumentacji, np. 500–5000).
- **Tabela per-detekcja (128 masek, siatka 8×8):** `matched_detection_rate` ~**0.72–0.88** — przy niższej wartości wyjaśnienie bywa **bardziej szumne** (często maska „gasi” detekcję).
- **Energia saliencji w bboxie (`box_energy_in_target_box`):** silny rozrzut między detekcjami (**~0.086 – ~0.49**) — różna **jakość lokalizacji** wyjaśnienia dla sąsiednich predykcji na jednym obrazie.
- Log: ostrzeżenie o **class_id poza [0,1)** przy `predict()` — sygnał do dopilnowania spójności typów / mapowania klas w skryptach D-RISE.

**Wnioski metodologiczne (zgodnie z treścią notebooka):**

- Oceniać D-RISE przez **nakładanie na bbox / eksperta** oraz **deletion/insertion**, a nie przez samą średnią intensywność mapy.
- **IoU w score** ogranicza „przeskakiwanie” na inne obiekty — ważne przy wielu detekcjach na skórze.

---

## 4. `04_v3_attention_evidence_map.ipynb` — mapa „evidence” z deformable attention (MSDeformAttn)

**Cel:** XAI **świadomy architektury**: rejestracja `sampling_locations` + `attention_weights` z `MSDeformAttn`, dobór **query** dopasowanego do wiersza `predict()` przez **IoU** (preferencja zgodności klasy), mapy rzadkie + Gaussian splat (tylko wizualizacja).

**Wyniki:**

- Dla wybranego obrazu (`536d69b3-…jpg`): **1 detekcja**, dopasowanie **query_id=0**, **matched_iou ≈ 1.0**, zgodność klasy **class_aware_ok=True**.
- Przechwycono **4 wywołania** warstw MSDeformAttn; zapisano m.in. `*_attention_raw_sparse.npy`, `*_gaussian_sigma12.npy`, overlay i heatmapę.
- **Batch 10 losowych indeksów:** 6/10 obrazów z poprawnym pipeline (reszta: brak detekcji), **wszystkie** przetworzone przypadki: **query=0**, IoU **1.00** — w tej próbie **stabilne mapowanie** na pierwszy slot query (nie uogólniać bez szerszej statystyki).
- Sekcja diagnostyczna: heatmapy **niskiej pewności** slotów wyłącznie jako **kontrola negatywna** (poprawne metodologicznie oddzielenie od XAI per detekcja).

**Wnioski:**

- To jedyna z analizowanych metod, która **bezpośrednio** wizualizuje mechanizm **dekodera** (sampling + wagi), a nie gradient ani maskowanie wejścia.
- Pipeline **predict → IoU → query** jest spójny z opisem w notebooku i daje **powtarzalne metadane** (JSON z ścieżkami plików).

---

## 5. `05_gradcampp.ipynb` — GradCAM++ (własna implementacja, `rfdetr_gradcampp_custom.py`)

**Cel:** target = surowy logit `[query_id, class_id]` po dopasowaniu bboxa z `predict()` do `pred_boxes`; warstwa domyślna: **`backbone.0.projector.stages.0.0.cv2`**; metryki: Pointing game, energy fraction (EF), energy enrichment (EE).

**Wyniki (zapisane wykonanie):**

- Obraz testowy z **4 detekcjami**; **matched_iou ~0.997–0.999** dla wszystkich — dopasowanie query **poprawne**.
- Dla **pierwszej detekcji (det=0):** Pointing game **False**, **EF = 0**, EE = 0, etykieta **`failed`** — mimo prawie idealnego IoU mapa **nie koncentruje energii w bboxie**.
- **Batch na tym samym obrazie (4 detekcje):** nadal **`failed`** w heurystyce jakości; część detekcji ma **EE > 1** (np. 1.92, 8.17) przy **nadal niskim** EF i braku trafienia Pointing game — zgodnie z notatką w notebooku: to sugeruje **słabą użyteczność GradCAM++ na tej warstwie** dla RF-DETR+DETR, a nie błąd matchingu.
- **Sweep warstw (n=4 detekcje na jednym obrazie):** wiele warstw projektora ma **pointing_game_acc = 0.25** lub 0; **żadna** nie rozwiązuje problemu w pełni w tej mini-pętli; warstwy agregujące typu `backbone` → **all failed**.
- **Batch wieloobrazowy (fragment, MAX_IMAGES=20 → łącznie 26 rekordów w tabeli):** dla rekordów `status=ok`: średnia **Pointing game ≈ 0.231**, średnia **EF ≈ 0.0187**, **matched_iou średnio ~0.999** — **wysokie IoU, niska lokalizacja CAM**.
- EigenCAM w notebooku: **`RUN_EIGEN_CAM=False`** — porównanie z EigenCAM **nie zostało uruchomione** w zapisanym stanie.

**Wnioski:**

- **GradCAM++ na domyślnym `C2f.cv2`** jest **słabym kandydatem** do „proofu” lokalizacji dla tego modelu — metryki z notebooka to **eksplicytny wynik negatywny** przy poprawnym dopasowaniu query.
- Sensowna ścieżka dalsza: **sweep** (już częściowo zrobiony) + porównanie z **D-RISE / attention evidence / SSGrad++**, a nie poleganie wyłącznie na GradCAM++.

---

## 6. `06_eigencam.ipynb` — EigenCAM (własna implementacja, `rfdetr_eigencam.py`)

**Cel:** **SVD** na aktywacjach warstwy (taka sama co GradCAM++), **bez gradientu**, **jedna mapa na obraz**; metryki bbox liczone **per detekcja** na tej samej mapie (metodologicznie: mapa **nie jest query-specific**).

**Wyniki:**

- Ten sam obraz co w 05: dla **det=0** — Pointing game **False**, EF **~0.0001**, jakość **`failed`**.
- **Wszystkie 4 detekcje** na obrazie: **`failed`** przy bardzo małym EF.
- **Pełny batch 217 obrazów:** 345 rekordów, **ok=215**; **Pointing game średnio ~0.093**, **EF średnio ~0.0233**, **matched_iou ~0.9993**; rozkład jakości: **failed 184**, **weak 28**, **good 3**.
- Po podsumowaniu: **TypeError** na `describe(numeric_only=True)` — różnica API **pandas** w środowisku (komórka kończy się błędem, ale agregaty wyżej zostały wypisane).

**Wnioski:**

- EigenCAM na tej warstwie i tym zadaniu **w większości przypadków nie wskazuje bboxa** (świadomie: metoda opisuje **dominantną przestrzenną wariancję cech**, nie „przyczynę” danej detekcji).
- Do pracy naukowej: **nie traktować** EigenCAM jako równoważnego wyjaśnienia per-query; nadaje się raczej do **globalnego** zarysu aktywacji backboneu / projektora, z ostrożną interpretacją przy wielu lesionach.

---

## 7. `07_SSGradpp.ipynb` — SSGrad-CAM++ (spatial sensitivity × GradCAM++)

**Cel:** jak GradCAM++, ale z mnożeniem przez **mapę wrażliwości przestrzennej** z `|∇A|` (implementacja własna, literatura: SSGrad-CAM++, CVPR Workshop 2024).

**Wyniki (kluczowe liczby z zapisu):**

- **det=0** (ta sama scena co 05/06): Pointing game **False**, ale **EF ≈ 0.185**, **EE ≈ 61.3**, jakość **`weak`** — wyraźna poprawa w **koncentracji energii względem 05**, choć szczyt nadal poza prostym testem „hit”.
- **Wszystkie 4 detekcje na obrazie:** 2× **`good`** (PG hit, wysokie EF/EE), 2× **`weak`** — metoda **rozróżnia** detekcje tam, gdzie GradCAM++ był jednolicie słaby.
- **Batch 20 obrazów (pierwsza detekcja):** 7 obrazów z detekcją; z agregacji `ok_df`: **Pointing game acc ≈ 0.857**, **średni EF ≈ 0.397**, **EE średnio ~114**, **`good_pct` ≈ 85.7%**, **`failed_pct` = 0** — **silny kontrast** względem GradCAM++ na tej samej warstwie.
- **Sweep warstw (n=4, jeden obraz):** najwyższe PG dla części podwarstw `m.1.cv2` itd.; **ostrożnie** (mały n).
- **Średnie po 217 obrazach** (sekcja podsumowania w MD komórki 47): dla kandydatów:
  - `projector` / `stages.0`: **EF ≈ 0.426**, **EE ≈ 61.6**, **PG ≈ 0.770**
  - **`stages.0.0.cv2`:** **EF ≈ 0.457**, **EE ≈ 75.3**, **PG ≈ 0.954** ← **najwyższy Pointing Game**
  - `m.1.cv2`: wyższe EF (~0.557), niższa stabilność / mniej „elegancka” warstwa do uzasadnienia
- **Rekomendacja zapisana w notebooku:** docelowa warstwa **`backbone.0.projector.stages.0.0.cv2`**, referencyjna **`backbone.0.projector`**.

**Wnioski:**

- **SSGrad-CAM++** jest **najlepszym kandydatem** spośród gradientowych CAMów testowanych tutaj na **lokalizacji zgodnej z bboxem** (metryki + argument o warstwie w `MultiScaleProjector`).
- Moduł **czułości przestrzennej** realnie adresuje problem „płaskiego” GradCAM++ na ViT+DETR w tej konfiguracji.
- Nadal: część detekcji zostaje w **`weak`** — warto raportować **rozkład** jakości, nie tylko średnią.

---

## 8. Synteza porównawcza

| Obszar | Baseline (02) | D-RISE (04) | Attention evidence (04 v3) | GradCAM++ (05) | EigenCAM (06) | SSGrad++ (07) |
|--------|----------------|------------|----------------------------|----------------|----------------|---------------|
| Rola | Metryki detekcji | Black-box, maski | Wewnętrzny mechanizm dekodera | Gradient + aktywacje | Wariancja cech (globalnie) | Gradient + aktyw. + \|∇A\| |
| Silna strona (z logów) | Wysokie AP@50 | Per-detekcja, IoU w score, krzywe del/ins | Bezpośrednia interpretacja samplingów | Formalne powiązanie z logitem query | Szybkość (brak backward) | Wysokie PG/EF w batchach, stabilna warstwa |
| Słaba / ograniczenie | AP 0.5:0.95 < AP@50 | Koszt masek; jakość zależy od `matched_detection_rate` | Tylko dekoder; nie zastępuje black-box | **Słaba lokalizacja** mimo IoU≈1 | **Nie per-query**; dominacja `failed` | Czasami `weak`; koszt sweepu |

**Praktyczna rekomendacja (do rozdziału XAI w pracy):**

1. **Detekcja / wiarygodność modelu:** cytować **02**.  
2. **Wyjaśnienie zgodne z post-processingiem i query:** **04 v3** (attention evidence) + **07 SSGrad++** na `projector.stages.0.0.cv2`.  
3. **Wyjaśnienie niezależne od gradientu / wewnętrznej struktury:** **D-RISE** z raportem zbieżności masek i deletion/insertion.  
4. **GradCAM++ / EigenCAM** na domyślnej warstwie: traktować jako **wynik negatywny** lub **pomocniczy** — nie jako główny dowód lokalizacji bez dodatkowych warstw / metod.

---

## 9. Problemy techniczne do naprawy (żeby powtórzenia były czyste)

- **03:** EigenCAM + błąd **inplace / no_grad**; potencjalna niespójność liczników sukcesów między komórkami.  
- **06:** `DataFrame.describe(numeric_only=True)` — niekompatybilność z wersją **pandas** w venv.  
- **Wspólne logi:** `predict()` / D-RISE — dopilnowanie **zakresu class_id** i komunikatów RF-DETR o **90 vs 1 klasach**.

---

## 10. Ścieżki artefaktów

- **Figury w repo (README):** [`wyniki/notebook_xai_przyklady/`](../wyniki/notebook_xai_przyklady/) — skopiowane z zapisu notebooków; nie zależą od osobnego katalogu `rfdetr-xai/`.
- **Po ponownym uruchomieniu:** ścieżki zapisu ustawiasz w komórkach notebooków (np. `wyniki/rfdetr_baseline/`, `wyniki/rfdetr_xai/`, podkatalogi pod CAM / D-RISE / attention / SSGrad) — zależnie od maszyny i konfiguracji.

---

*Plik wygenerowany jako podsumowanie istniejących outputów w repozytorium; po ponownym uruchomieniu notebooków liczby mogą się zmienić przy innych seedach, progach lub checkpointach.*
