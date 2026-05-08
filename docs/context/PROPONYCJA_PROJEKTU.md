# Propozycja projektu

## Tytuł

**Wyjaśnialność modeli detekcji struktur dermatoskopowych z wykorzystaniem metod XAI**

## Opis projektu

Projekt stanowi część pracy magisterskiej i dotyczy analizy obrazów dermatoskopowych z wykorzystaniem metod sztucznej inteligencji. Dysponujemy zbiorem danych zawierającym zdjęcia zmian skórnych wraz z anotacjami struktur dermatoskopowych (w postaci bounding boxów). Na tej podstawie trenowany jest model klasyfikacyjny rozwiązujący zadanie **wieloetykietowej klasyfikacji obecności struktur** (czy dana struktura występuje na obrazie), a równolegle wykorzystywany jest **model detekcyjny (RF-DETR)**, który lokalizuje te struktury na obrazie.

Głównym celem projektu jest zbadanie, **na ile decyzje modelu klasyfikacyjnego są zgodne z faktycznymi strukturami obecnymi na obrazie**. W tym celu wykorzystane zostaną metody **wyjaśnialnej sztucznej inteligencji (XAI)**, takie jak **Grad-CAM**, które pozwalają określić, na które fragmenty obrazu model „zwraca uwagę”. Następnie obszary wskazywane przez XAI zostaną **porównane z wynikami detekcji** (bounding boxami) uzyskanymi przez model RF-DETR.

## Cel badawczy (skrót)

- Ocenić spójność między mapami uwagi / wyjaśnieniami XAI a lokalizacją struktur z detektora.
- Wnioskować o tym, czy klasyfikator opiera się na fragmentach obrazu zgodnych z medycznie interpretowalnymi strukturami dermoskopowymi.

## Kontekst techniczny

| Element | Opis |
|--------|------|
| Dane | Obrazy dermatoskopowe, anotacje struktur (bounding boxy). |
| Klasyfikacja | Model wieloetykietowy: obecność poszczególnych struktur na obrazie. |
| Detekcja | RF-DETR — lokalizacja struktur. |
| XAI | m.in. Grad-CAM — wizualizacja obszarów wpływających na decyzję klasyfikatora. |
| Porównanie | Mapy XAI vs. boxy z detektora (np. nakładanie, metryki przestrzenne — do ustalenia w implementacji). |

## Aspekty etyczne, społeczne i regulacyjne

Projekt odnosi się do aspektów etycznych i społecznych AI, ponieważ dotyczy **zastosowań w medycynie**, gdzie kluczowe znaczenie ma **przejrzystość** i możliwość **uzasadnienia** działania modelu. Wyniki mogą wskazać, czy model rzeczywiście opiera swoje decyzje na **medycznie istotnych strukturach**, co jest istotne w kontekście:

- budowania **zaufania** do systemów wspomagających diagnostykę,
- wymagań **transparentności** wynikających z regulacji, w tym **Rozporządzenia UE o sztucznej inteligencji (AI Act)**,
- odpowiedzialnego wdrażania narzędzi AI przy ocenie zmian skórnych (m.in. unikanie „pozornej” pewności bez sensownej podstawy wizualnej).

## Oczekiwane artefakty (orientacyjnie)

- Opis pipeline’u: dane → klasyfikacja + detekcja → XAI → porównanie.
- Wizualizacje i/lub metryki porównawcze (do umieszczenia w katalogu `wyniki/` po uzyskaniu wyników).
- Wnioski dotyczące zgodności wyjaśnień z detekcją oraz implikacji dla przejrzystości i ryzyk (np. rozjechanie się XAI od rzeczywistej lokalizacji struktur).

## Ograniczenia i dalsze kierunki

- Jakość wyjaśnień XAI zależy od architektury i warstw, na których Grad-CAM jest stosowany; interpretacja wymaga ostrożności klinicznej.
- Detektor i klasyfikator mogą się różnić pod względem błędów; porównanie XAI–detekcja nie zastępuje walidacji klinicznej.
- Możliwe rozszerzenia: inne metody XAI, formalne metryki IoU / korelacji z maskami, ocena z udziałem ekspertów.

## Powiązanie z kursem (mini-projekt)

Mini-projekt w repozytorium kursu „Aspekty prawne, społeczne i etyczne w sztucznej inteligencji” może dokumentować powyższy wątek przede wszystkim pod kątem **przejrzystości, zaufania, AI Act i odpowiedzialnego AI w medycynie**, a szczegóły implementacyjne i pełne wyniki magisterki — w pracy dyplomowej i powiązanych materiałach źródłowych.

---

*Dokument opisuje propozycję projektu; metryki i rysunki wynikowe należy uzupełnić po przeprowadzeniu eksperymentów.*
