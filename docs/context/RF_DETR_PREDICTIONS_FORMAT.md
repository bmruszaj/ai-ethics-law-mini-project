# Format predykcji RF-DETR (plan integracji z XAI)

W pierwszej iteracji mini-projektu ewaluacja względem **bboxów GT (COCO)** jest źródłem prawdy przestrzenniej — bez plików z detektora.

## Docelowy format JSON (jeden plik lub jeden plik na obraz)

Przykład listy detekcji per obraz (współrzędne w pikselach, jak w COCO `bbox` można przekonwertować na `xyxy`):

```json
{
  "images": [
    {
      "file_name": "przykład.jpg",
      "image_id": 101,
      "detections": [
        {
          "category_name": "Blue-gray globules",
          "bbox_xyxy": [120.0, 80.0, 220.0, 190.0],
          "score": 0.87
        }
      ]
    }
  ],
  "model": "rf-detr",
  "schema_version": 1
}
```

## Mapowanie na metryki

- **Pointing Game / Energy IoU:** użyj `bbox_xyxy` z detektora zamiast GT — te same funkcje w `src/xai_metrics.py`.
- **Porównanie GT vs detektor:** osobno policz metryki dla GT i dla boxów RF-DETR; różnica interpretuj jako **wpływ błędów lokalizacji** na ocenę wyjaśnień (fałszywe trafienia / pominięcia struktury).

## Ograniczenia

- Brak standaryzacji między zespołami — dopasuj `file_name` / `image_id` do kluczy w `result.json` zbioru.
- Progi `score` filtrują szum; zapisz użyty próg w `PROCESS.md`.
