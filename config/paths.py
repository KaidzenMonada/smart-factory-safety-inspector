"""
Project paths.

This module contains all directory and file paths used throughout the project.
"""

from pathlib import Path

# ==================================================
# Project Root
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================================================
# Project Directories
# ==================================================

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"
SRC_DIR = PROJECT_ROOT / "src"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# ==================================================
# Dataset Configuration
# ==================================================

FINAL_DATASET_DIR = DATA_DIR / "final_dataset"
DATASET_DIR = FINAL_DATASET_DIR

YAML_FILE = DATASET_DIR / "data.yaml"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

# ==================================================
# Evaluation
# ==================================================

BEST_MODEL = MODELS_DIR / "best.pt"

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
