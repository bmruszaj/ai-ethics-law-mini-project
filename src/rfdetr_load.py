"""
Ładowanie RF-DETR z checkpointu. Architektura i preprocess inference pochodzą z pakietu ``rfdetr``
(te same konwencje co trening w repozytorium ``dermatoscopy_ai``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

ModelSize = Literal["nano", "small", "medium", "large"]


def _model_ctor(model_size: str) -> type:
    """Zwraca klasę modelu RF-DETR dla podanego rozmiaru."""
    key = model_size.lower()

    # Wzorzec jak w skrypcie modalowym: import klasy zależnie od rozmiaru.
    if key == "nano":
        from rfdetr import RFDETRNano

        return RFDETRNano
    if key == "small":
        from rfdetr import RFDETRSmall

        return RFDETRSmall
    if key == "medium":
        from rfdetr import RFDETRMedium

        return RFDETRMedium
    if key == "large":
        from rfdetr import RFDETRLarge

        return RFDETRLarge

    raise ValueError(
        f"Nieobsługiwany model_size={model_size!r}; dozwolone: ['large', 'medium', 'nano', 'small']"
    )


def load_rfdetr(model_size: ModelSize, checkpoint_path: str | Path, **model_kwargs: Any) -> Any:
    """
    Buduje model RF-DETR danego rozmiaru i ładuje wagi z ``checkpoint_path``.

    Checkpoint powinien być zgodny z formatem RF-DETR (np. ``best_model.pt`` z treningu:
    klucz ``model`` ze ``state_dict``).
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt}")

    ctor = _model_ctor(model_size)
    # ``pretrain_weights`` — ścieżka do wag (jak w ``rfdetr`` / ``dermatoscopy_ai``).
    return ctor(pretrain_weights=str(ckpt.resolve()), **model_kwargs)
