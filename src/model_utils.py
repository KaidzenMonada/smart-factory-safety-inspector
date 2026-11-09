"""Model loading helpers for the Smart Factory Safety Inspector pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config.dataset import CLASS_NAMES
from config.paths import BEST_MODEL


class ModelClassMismatchError(ValueError):
    """Raised when model class metadata does not match the project schema."""


def normalize_model_names(names: Any) -> list[str]:
    """Return model class names in class-id order."""

    if isinstance(names, dict):
        return [
            str(names[key])
            for key in sorted(names, key=lambda item: int(item))
        ]

    return [str(name) for name in names]


def validate_model_classes(model: YOLO) -> None:
    """Validate that a YOLO model matches the final five-class schema."""

    model_names = normalize_model_names(model.names)
    expected_names = list(CLASS_NAMES)

    if model_names != expected_names:
        raise ModelClassMismatchError(
            "Model class names do not match the final Smart Factory "
            "Safety Inspector schema. "
            f"Expected {expected_names}, got {model_names}."
        )


def load_project_model(model_path: str | Path = BEST_MODEL) -> YOLO:
    """Load the project model and validate final project classes."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained Smart Factory Safety Inspector model weights were "
            "not found:\n"
            f"{model_path}\n"
            "Train the model with train.py and place/export the final "
            "weights at models/best.pt."
        )

    model = YOLO(model_path)
    validate_model_classes(model)

    return model
