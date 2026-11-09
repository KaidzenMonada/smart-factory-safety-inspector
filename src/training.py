"""
YOLO training helpers for the final Smart Factory Safety Inspector dataset.
"""

from __future__ import annotations

from datetime import datetime

from ultralytics import YOLO

from config.paths import RUNS_DIR, YAML_FILE
from config.training_config import (
    AMP,
    BATCH_SIZE,
    CACHE,
    CLOSE_MOSAIC,
    DEGREES,
    DETERMINISTIC,
    DEVICE,
    EPOCHS,
    FLIPLR,
    FLIPUD,
    HSV_H,
    HSV_S,
    HSV_V,
    IMAGE_SIZE,
    MIXUP,
    MODELS,
    MOSAIC,
    OPTIMIZER,
    PATIENCE,
    PERSPECTIVE,
    PLOTS,
    PRETRAINED,
    SAVE,
    SAVE_PERIOD,
    SCALE,
    SEED,
    SHEAR,
    TRANSLATE,
    VERBOSE,
    WORKERS,
)


def load_model(model_name: str) -> YOLO:
    """
    Load a pretrained YOLO model.
    """

    print(f"\nLoading model: {model_name}")

    model = YOLO(model_name)

    print(f"OK: {model_name} loaded successfully.")

    return model


def get_training_config(model_name: str) -> dict:
    """
    Build the training configuration dictionary for YOLO.
    """

    if not YAML_FILE.exists():
        raise FileNotFoundError(
            "Final dataset data.yaml not found:\n"
            f"{YAML_FILE}"
        )

    return {
        # Dataset
        "data": str(YAML_FILE),

        # Core training settings
        "imgsz": IMAGE_SIZE,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "patience": PATIENCE,
        "optimizer": OPTIMIZER,

        # Data augmentation
        "hsv_h": HSV_H,
        "hsv_s": HSV_S,
        "hsv_v": HSV_V,
        "degrees": DEGREES,
        "translate": TRANSLATE,
        "scale": SCALE,
        "fliplr": FLIPLR,
        "flipud": FLIPUD,
        "mosaic": MOSAIC,
        "close_mosaic": CLOSE_MOSAIC,
        "mixup": MIXUP,
        "shear": SHEAR,
        "perspective": PERSPECTIVE,

        # Hardware
        "device": DEVICE,
        "workers": WORKERS,
        "amp": AMP,
        "cache": CACHE,
        "deterministic": DETERMINISTIC,
        "pretrained": PRETRAINED,

        # Reproducibility
        "seed": SEED,

        # Output
        "plots": PLOTS,
        "save": SAVE,
        "save_period": SAVE_PERIOD,
        "project": str(RUNS_DIR),
        "name": model_name.replace(".pt", ""),
        "exist_ok": False,
        "verbose": VERBOSE,
    }


def train_model(model_name: str):
    """
    Train a single YOLO model.
    """

    model = load_model(model_name)
    training_config = get_training_config(model_name)

    print(f"\nStarting training: {model_name}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")

    results = model.train(**training_config)

    print(f"OK: Training completed: {model_name}")

    return results


def train_all_models() -> None:
    """
    Train all YOLO models defined in the configuration.
    """

    print("=" * 70)
    print("YOLO Smart Factory Safety Inspector Training Experiment")
    print("=" * 70)

    start_time = datetime.now()

    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {YAML_FILE}")
    print(f"Models to train: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print()

    failures = 0

    for index, model_name in enumerate(MODELS, start=1):
        print("-" * 70)
        print(f"Model {index}/{len(MODELS)}")
        print(f"Architecture: {model_name}")
        print("-" * 70)

        try:
            train_model(model_name)

        except Exception as error:
            failures += 1
            print(f"\nFAILED: Training failed for {model_name}")
            print(f"Error: {error}")

        print()

    finish_time = datetime.now()
    total_time = finish_time - start_time

    print("=" * 70)
    print("Training Experiment Completed")
    print("=" * 70)
    print(f"Finished: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training Time: {total_time}")

    if failures:
        print(f"\nTraining workflow finished with {failures} failure(s).")
    else:
        print("\nTraining workflow finished successfully.")
