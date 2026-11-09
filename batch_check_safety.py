"""
Batch image safety compliance checker.

Run:

    python batch_check_safety.py
"""

from check_safety import select_required_ppe
from src.batch import process_batch
from src.model_utils import load_project_model


def main():
    """
    Run batch compliance evaluation for a folder of images.
    """

    model = load_project_model()

    required_ppe = select_required_ppe()

    input_dir = input("\nEnter image folder path: ").strip()

    summary = process_batch(
        model=model,
        input_dir=input_dir,
        required_ppe=required_ppe,
    )

    print("\nBatch Compliance Assessment")
    print("=" * 50)
    print(f"Processed: {summary['processed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Outputs: {summary['output_dir']}")

    if summary["csv_report"] is not None:
        print(f"CSV report: {summary['csv_report']}")

    if summary["errors"]:
        print("\nFailed images:")

        for error in summary["errors"]:
            print(f"- {error['image']}: {error['error']}")


if __name__ == "__main__":
    main()
