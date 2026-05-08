# Metody wyjaśnialności planowane w projekcie

W ramach projektu planowana jest **ocena i porównanie trzech metod wyjaśnialności** (XAI) pod kątem zgodności z lokalizacją struktur dermatoskopowych (np. w odniesieniu do wyników detektora RF-DETR oraz anotacji). Poniżej zestawienie metod, uzasadnień wyboru oraz skrócony opis mechanizmu działania.

---

## 1. LeGrad — metoda główna

**Źródła:** Bousselham et al., ICCV 2025 · [arXiv:2404.03214](https://arxiv.org/abs/2404.03214) · [GitHub: WalBouss/LeGrad](https://github.com/WalBouss/LeGrad)

**Dlaczego ta metoda:** W literaturze jest to podejście **jawnie testowane na SigLIP**. Natywnie uwzględnia **attentional pooler** używany w SigLIP. Autorzy raportują ok. **3,6× lepszy** wynik względem Grad-CAM na SigLIP (np. **25,4 vs 7,0 p-mIoU** na OpenImagesV7). Metoda jest **szybka** (rzędu ~5 ms na obraz) i **otwartoźródłowa**.

**Zasada działania:** Obliczane są gradienty predykcji względem map uwagi w kolejnych warstwach, następnie uśredniane (z obcięciem ReLU). Prostsza niż Chefer, przy zachowaniu lepszej wierności przestrzennej w kontekście SigLIP.

---

## 2. Chefer 2021b (Generic Attention-model Explainability) — metoda pomocnicza

**Źródła:** Chefer et al., ICCV 2021 Oral · [arXiv:2103.15679](https://arxiv.org/abs/2103.15679) · [GitHub: hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)

**Dlaczego ta metoda:** Podejście **pokazane bezpośrednio na CLIP** — ta sama **rodzina architektury** co SigLIP. Mapy są **rozróżniające względem klasy** (class-discriminative): różne prompty / klasy dają różne mapy. Metoda jest **dobrze zacytowana** i szeroko opisana w literaturze.

**Zasada działania:** Kombinacja **gradient × attention** w kolejnych warstwach, agregacja w stylu **rollout** (mnożenie macierzy). Nieco bardziej złożona niż LeGrad, z mocniejszym umocowaniem teoretycznym.

---

## 3. RISE — linia odniesienia (czarna skrzynka)

**Źródła:** Petsiuk et al., BMVC 2018 · [arXiv:1806.07421](https://arxiv.org/abs/1806.07421)

**Dlaczego ta metoda:** **Całkowicie niezależna od architektury** — nie wymaga dostępu do gradientów ani mechanizmu attention. Daje **niezależną walidację**: jeśli **LeGrad** (i ewentualnie Chefer) **zgadzają się z RISE** co do istotnych regionów, argument za spójnością wyjaśnienia jest silniejszy. **Wolniejsza** (rzędu ~4000 przejść do przodu na obraz), ale **akceptowalna** do ewaluacji na ograniczonym zbiorze (np. **206 obrazów** w planowanym zakresie).

**Zasada działania:** Losowe maskowanie fragmentów obrazu i agregacja wpływu masek na wyjście modelu (podejście typu **black-box** oparte na perturbacjach wejścia).

---

## Rola trzech metod w badaniu

| Metoda   | Rola w projekcie        | Charakterystyka |
|----------|-------------------------|-----------------|
| **LeGrad** | Główna, dopasowana do SigLIP | Szybka, gradientowa, uwzględnia attentional pooler |
| **Chefer** | Uzupełnienie, CLIP/SigLIP  | Class-discriminative, rollout gradient–attention |
| **RISE**   | Baseline black-box        | Bez gradientów/attention, kosztowna obliczeniowo |

Szczegóły implementacji (warstwy, hiperparametry RISE, metryki zgodności z bounding boxami) należy opisać w kodzie / notebooku oraz w `PROCESS.md` po ustaleniu eksperymentu.

---

*Dokument uzupełnia propozycję projektu o konkretny zestaw metod XAI; wyniki porównań należy dodać po przeprowadzeniu ewaluacji.*
