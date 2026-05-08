# Ewaluacja wyjaśnień XAI

Dokument zapisuje **metryki ewaluacji** uzgodnione na potrzeby projektu (porównanie map wyjaśnialności z anotacjami eksperckimi w postaci bounding boxów oraz ocena wierności modelu względem predykcji).

---

## 1. Pointing Game

**Pytanie:** Czy punkt o **największej saliencji** (maksimum mapy wyjaśnienia) wpada **do wnętrza prostokąta eksperckiego** (bounding boxa)?

**Interpretacja:** Prosta miara **lokalizacyjna** — sprawdza, czy model „wskazuje” co najmniej jeden najważniejszy punkt w regionie uznanym za istotny w anotacji. Typowo liczona jako **udział trafień** (accuracy / hit rate) po stronie obrazów lub struktur.

**Kontekst:** Dobrze nadaje się do szybkiej oceny zgodności wyjaśnienia z bboxem; **nie** uwzględnia rozłożenia saliencji w całym obszarze (tylko jeden punkt).

---

## 2. Energy IoU (Intersection over Union energii)

**Pytanie:** **Jaka część całkowitej „energii” saliencji** (np. suma wartości mapy po normalizacji) znajduje się **wewnątrz** bounding boxa względem energii w całym obrazie / union z bboxem — w zależności od przyjętej definicji (w literaturze warianty opierają się na **nakładaniu się masy prawdopodobieństwa** mapy z regionem GT).

**Interpretacja:** Miara **pokrycia przestrzennego** — nagradza wyjaśnienia, które koncentrują saliencję w obrębie anotacji, a nie rozlewają ją po całym obrazie.

**Kontekst:** Uzupełnia Pointing Game o informację o **rozłożeniu** ważności, nie tylko o jednym maksimum.

---

## 3. Insertion / Deletion — AUC

**Pytanie (idea):** Jak zmienia się **pewność predykcji modelu**, gdy do wejścia wprowadzamy obraz **stopniowo** — od pustego / rozmytego do pełnego (**insertion**), albo gdy **usuwamy** (maskujemy) najważniejsze piksele według mapy (**deletion**)?

**Metryka:** **Powierzchnia pod krzywą** (AUC) w funkcji ułamka wprowadzonych / usuniętych pikseli — miara **wierności (faithfulness)** wyjaśnienia względem **samego modelu**, bez odwołania do bboxów.

**Interpretacja:** Jeśli wyjaśnienie jest spójne z tym, co naprawdę napędza sieć, **insertion** powinno szybko rosnąć przy dodawaniu najważniejszych regionów, a **deletion** — szybko obniżać predykcję przy usuwaniu tych regionów. Wysokie AUC insertion i niskie / odpowiednie zachowanie deletion (w zależności od definicji w skrypcie) wskazują na lepszą zgodność mapy z mechanizmem decyzyjnym.

**Kontekst:** **Niezależne od anotacji** eksperckich — przydatne do porównania metod XAI (LeGrad, Chefer, RISE) na równych zasadach oraz do wykrycia map „ładnych”, ale **niefaithful** względem modelu.

---

## Podsumowanie ról metryk

| Metryka | Zależność od bboxów | Co głównie mierzy |
|--------|----------------------|-------------------|
| **Pointing Game** | Tak (ekspert) | Czy maksimum saliencji trafia w region GT |
| **Energy IoU** | Tak (ekspert) | Jaka część energii saliencji leży w bboxie |
| **Insertion / Deletion AUC** | Nie | Wierność mapy wobec zachowania modelu |

Szczegóły implementacji (normalizacja map, kolejność pikseli, liczba kroków krzywej, uśrednianie po klasach) należy opisać w kodzie / `PROCESS.md` po ustaleniu eksperymentu.

---

*Dokument odzwiercida ustalenia z notatek projektowych; po uruchomieniu eksperymentów warto uzupełnić tabele wyników i odnośniki do plików w `wyniki/`.*
