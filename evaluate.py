"""
Final model evaluation.

This module evaluates the selected YOLO model on the test dataset.
"""

import csv
import json

from config.dataset import CLASS_NAMES
from config.paths import (
    BEST_MODEL,
    YAML_FILE,
    EVALUATION_DIR,
)

from config.training_config import (
    IMAGE_SIZE,
    DEVICE,
    WORKERS,
    PLOTS,
    VERBOSE,
)
from src.model_utils import load_project_model


def load_model():
    """
    Load the trained YOLO model.
    """

    print("Loading trained model...")

    model = load_project_model(BEST_MODEL)

    print(f"Model loaded: {BEST_MODEL.name}")

    return model


def run_evaluation(model):
    """
    Evaluate the trained model on the test dataset.
    """

    print("\nStarting model evaluation...")

    if not YAML_FILE.exists():
        raise FileNotFoundError(
            "Final Smart Factory Safety Inspector data.yaml "
            "not found:\n"
            f"{YAML_FILE}\n"
            "Add and validate the final dataset before evaluation."
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    results = model.val(
        data=YAML_FILE,
        split="test",
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=PLOTS,
        verbose=VERBOSE,
        project=EVALUATION_DIR,
        name=BEST_MODEL.stem,
        exist_ok=True,
        save_json=False,
        save_txt=False,
    )

    print("Evaluation completed.")

    return results


def save_results(results):
    """
    Save evaluation results for further analysis.
    """

    save_dir = results.save_dir

    json_path = save_dir / "evaluation_results.json"
    csv_path = save_dir / "evaluation_results.csv"

    metrics = results.box
    per_class = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "class": class_name,
                "precision": float(metrics.p[class_index]),
                "recall": float(metrics.r[class_index]),
                "map50": float(metrics.ap50[class_index]),
                "map50_95": float(metrics.ap[class_index]),
            }
        )

    summary = {
        "model": str(BEST_MODEL),
        "dataset": str(YAML_FILE),
        "split": "test",
        "image_size": IMAGE_SIZE,
        "overall": {
            "precision": float(metrics.mp),
            "recall": float(metrics.mr),
            "map50": float(metrics.map50),
            "map50_95": float(metrics.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            key: float(value)
            for key, value in results.speed.items()
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "precision",
                "recall",
                "map50",
                "map50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(per_class)

    print(f"\nEvaluation results saved:")
    print(f"JSON : {json_path.name}")
    print(f"CSV  : {csv_path.name}")


def print_results(results):
    """
    Display evaluation metrics.
    """

    print("\n" + "=" * 50)
    print("FINAL MODEL EVALUATION")
    print("=" * 50)

    metrics = results.box

    print(f"Precision : {metrics.mp:.4f}")
    print(f"Recall    : {metrics.mr:.4f}")
    print(f"mAP50     : {metrics.map50:.4f}")
    print(f"mAP50-95  : {metrics.map:.4f}")


def main():
    """
    Run the complete evaluation pipeline.
    """

    model = load_model()

    results = run_evaluation(model)

    save_results(results)

    print_results(results)


if __name__ == "__main__":
    main()
