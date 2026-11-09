"""
Main training script.

Run:

    python train.py
"""

from src.training import train_all_models


def main() -> None:
    """
    Run the complete YOLO training experiment.
    """

    train_all_models()


if __name__ == "__main__":
    main()
