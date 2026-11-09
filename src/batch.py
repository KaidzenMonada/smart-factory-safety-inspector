"""
Batch safety compliance evaluation utilities.
"""

import csv
from datetime import datetime
from pathlib import Path

from config.paths import PREDICTIONS_DIR
from src.inference import evaluate_image
from src.path_utils import safe_filename_stem, safe_is_file
from src.visualization import save_prediction_image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


CSV_FIELDS = [
    "timestamp",
    "image_filename",
    "image_path",
    "overall_status",
    "persons_detected",
    "compliant_persons",
    "attention_persons",
    "required_ppe",
    "ppe_not_detected_summary",
    "unassigned_ppe",
    "detected_objects",
    "total_detected_objects",
    "annotated_image_path",
]


def format_ppe_not_detected_summary(ppe_not_detected_summary):
    """
    Return a compact person-level PPE-not-detected summary for CSV output.
    """

    if not ppe_not_detected_summary:
        return ""

    return "; ".join(
        f"Worker {item['person_id']}: {', '.join(item['missing_ppe'])}"
        for item in ppe_not_detected_summary
    )


def format_unassigned_ppe(unassigned_ppe):
    """
    Return a compact class list for PPE that could not be assigned to a person.
    """

    if not unassigned_ppe:
        return ""

    return ";".join(
        detection["class"]
        for detection in unassigned_ppe
    )


def build_csv_row(
    image_path,
    detections,
    required_ppe,
    compliance_result,
    annotated_image_path,
):
    """
    Build one batch_compliance_results.csv row.
    """

    image_path = Path(image_path)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_filename": image_path.name,
        "image_path": str(image_path),
        "overall_status": compliance_result["status"],
        "persons_detected": compliance_result["persons_detected"],
        "compliant_persons": compliance_result["compliant_persons"],
        "attention_persons": compliance_result["attention_persons"],
        "required_ppe": ";".join(required_ppe),
        "ppe_not_detected_summary": format_ppe_not_detected_summary(
            compliance_result["ppe_not_detected_summary"]
        ),
        "unassigned_ppe": format_unassigned_ppe(
            compliance_result["unassigned_ppe"]
        ),
        "detected_objects": ";".join(compliance_result["detected"]),
        "total_detected_objects": len(detections),
        "annotated_image_path": str(annotated_image_path),
    }


def save_csv_report(rows, csv_path):
    """
    Save batch image evaluation rows to one CSV report.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def find_image_files(input_dir):
    """
    Find supported image files in a directory tree.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing images.

    Returns
    -------
    list[Path]
        Sorted recursive list of image paths.
    """

    input_dir = Path(input_dir)

    return sorted(
        path
        for path in input_dir.rglob("*")
        if safe_is_file(path)
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def get_batch_output_dir(input_dir, output_dir=None):
    """
    Return the output directory for a batch inference input folder.

    By default, batch outputs are grouped under outputs/predictions/ using the
    input folder name. An explicit output_dir is still accepted for scripts that
    need a custom destination.
    """

    if output_dir is not None:
        return Path(output_dir)

    source_folder = Path(input_dir)
    return PREDICTIONS_DIR / safe_filename_stem(source_folder.name)


def process_batch(
    model,
    input_dir,
    required_ppe,
    output_dir=None,
):
    """
    Evaluate all supported images in a folder tree.

    Processing continues if one image fails. Successful evaluations save
    annotated images and one CSV report for the batch.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded YOLO model.

    input_dir : str or Path
        Folder tree of images to evaluate.

    required_ppe : list[str]
        Required PPE classes.

    output_dir : str or Path, optional
        Base directory for batch outputs. Defaults to
        outputs/predictions/<input-folder-name>/.

    Returns
    -------
    dict
        Batch summary with successful reports, errors, and output paths.
    """

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found:\n{input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder:\n{input_dir}"
        )

    output_dir = get_batch_output_dir(
        input_dir=input_dir,
        output_dir=output_dir,
    )
    annotated_dir = output_dir
    csv_path = output_dir / "batch_compliance_results.csv"

    image_files = find_image_files(input_dir)
    csv_rows = []
    errors = []

    total_images = len(image_files)

    print(f"\nFound {total_images} supported image(s).")

    for index, image_path in enumerate(image_files, start=1):
        relative_image_path = image_path.relative_to(input_dir)
        image_output_dir = annotated_dir / relative_image_path.parent

        print(f"[{index}/{total_images}] Processing {relative_image_path}...")

        try:
            evaluation = evaluate_image(
                model=model,
                image_path=image_path,
                required_ppe=required_ppe,
            )

            compliance_result = evaluation["compliance_result"]

            saved_image = save_prediction_image(
                image=evaluation["result"].orig_img,
                compliance_result=compliance_result,
                output_path=image_output_dir,
                detections=evaluation["detections"],
                output_filename=image_path.name,
            )

            csv_rows.append(
                build_csv_row(
                    image_path=image_path,
                    detections=evaluation["detections"],
                    required_ppe=required_ppe,
                    compliance_result=compliance_result,
                    annotated_image_path=saved_image,
                )
            )

        except Exception as error:
            errors.append(
                {
                    "image": str(image_path),
                    "error": str(error),
                }
            )
            print(f"Failed: {error}")

    save_csv_report(
        rows=csv_rows,
        csv_path=csv_path,
    )

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "csv_report": csv_path,
        "processed": len(csv_rows),
        "failed": len(errors),
        "errors": errors,
        "reports": csv_rows,
    }
