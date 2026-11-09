"""
Training configuration for YOLO model comparison.

All models use the same training settings to ensure
a fair comparison between the configured candidate models.
"""

# ==================================================
# Models to Compare
# ==================================================

MODELS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
]
# ==================================================
# Image Parameters
# ==================================================

IMAGE_SIZE = 640

# ==================================================
# Training Parameters
# ==================================================

EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 16

# ==================================================
# Optimizer
# ==================================================

OPTIMIZER = "auto"

# ==================================================
# Data Augmentation
# ==================================================

# Color augmentation
HSV_H = 0.015
HSV_S = 0.5
HSV_V = 0.3

# Geometric augmentation
DEGREES = 5.0
TRANSLATE = 0.05
SCALE = 0.2

# Flipping
FLIPLR = 0.5
FLIPUD = 0.0

# Multi-image augmentation
MOSAIC = 0.5
CLOSE_MOSAIC = 10
MIXUP = 0.0

# Disabled geometric transformations
SHEAR = 0.0
PERSPECTIVE = 0.0

# ==================================================
# Hardware
# ==================================================

DEVICE = 0
WORKERS = 8
AMP = True
CACHE = False
DETERMINISTIC = True
PRETRAINED = True

# ==================================================
# Reproducibility
# ==================================================

SEED = 42

# ==================================================
# Output Options
# ==================================================

PLOTS = True
SAVE = True
SAVE_PERIOD = -1
VERBOSE = True
