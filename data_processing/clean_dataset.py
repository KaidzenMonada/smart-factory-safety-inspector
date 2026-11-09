"""Apply the confirmed final dataset corrections.

This script edits only confirmed same-source copies and annotation fixes
identified during EDA/manual review. It intentionally does not perform broad
cleaning, inference, relabeling, or heuristic repairs.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"


CROSS_SPLIT_FILES_TO_REMOVE = [
    "valid/images/ppe_1093_png_jpg.rf.39bee1143ebd7fb2f505b7fc537d696b.jpg",
    "train/images/ppe_1223_png_jpg.rf.c6084b2e9e071641a4ff5bcc4462e2f0.jpg",
    "train/images/3d928d51-9274-449b-aa62-cebd0a80a09c.jpg",
]


WITHIN_SPLIT_FILES_TO_REMOVE = [
    "train/images/aug_3406438.jpg",
    "train/images/aug_4361292.jpg",
    "train/images/18eb0286-8f8e-4079-8b1f-9b6f92c1c0b0.jpg",
    "train/images/a01634.jpg",
    "train/images/aug_1430145.jpg",
    "train/images/aug_4714368.jpg",
    "train/images/aug_8597202.jpg",
    "train/images/bpart2_001190.jpg",
    "test/images/546fecd0-2f3b-4536-a053-e58ad96169a8.jpg",
    "train/images/aug_9307744.jpg",
    "train/images/5d45b2ac-8094-4ec0-93ef-1619d617f77a.jpg",
    "train/images/aug_6980863.jpg",
    "train/images/aug_1904601.jpg",
    "train/images/aug_5662293.jpg",
]


FILES_TO_REMOVE = CROSS_SPLIT_FILES_TO_REMOVE + WITHIN_SPLIT_FILES_TO_REMOVE


LABEL_EDITS = {
    "test/labels/b000408.txt": {
        "remove": [
            "4 0.220570 0.644078 0.303328 0.711844",
            "4 0.546250 0.619008 0.464062 0.761984",
        ],
        "add": [
            "0 0.225000 0.621875 0.312500 0.756250",
            "0 0.546094 0.618750 0.464062 0.762500",
        ],
    },
    "train/labels/3e4ff635-42f7-4f3c-9a38-b76bbce01d08.txt": {
        "remove": [],
        "add": [
            "0 0.543750 0.546094 0.303125 0.495312",
            "0 0.753125 0.571094 0.237500 0.442188",
        ],
    },
    "train/labels/bpart2_001190.txt": {
        "remove": [],
        "add": [
            "0 0.417969 0.589063 0.310937 0.821875",
            "0 0.618750 0.577344 0.400000 0.845313",
        ],
    },
    "train/labels/helmet-13-_jpg.rf.318b5cdc177f145efd3e22cfb5c85101.txt": {
        "remove": [
            "4 0.584797 0.535562 0.823313 0.928656",
        ],
        "add": [
            "0 0.596875 0.390625 0.806250 0.671875",
        ],
    },
}


def image_to_label_path(image_relative_path: str) -> Path:
    image_path = Path(image_relative_path)
    return DATASET_DIR / image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def removed_label_paths() -> set[str]:
    return {image_to_label_path(path).relative_to(DATASET_DIR).as_posix() for path in FILES_TO_REMOVE}


def validate_inputs() -> None:
    for image_relative_path in FILES_TO_REMOVE:
        image_exists = (DATASET_DIR / image_relative_path).is_file()
        label_exists = image_to_label_path(image_relative_path).is_file()
        if image_exists != label_exists:
            raise FileNotFoundError(f"Image/label removal mismatch: {image_relative_path}")

    removed_labels = removed_label_paths()
    for label_relative_path, edit in LABEL_EDITS.items():
        label_path = DATASET_DIR / label_relative_path
        if not label_path.is_file():
            if label_relative_path in removed_labels:
                continue
            raise FileNotFoundError(f"Required label file not found: {label_relative_path}")
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for row in edit["remove"]:
            if lines.count(row) > 1:
                raise ValueError(f"Duplicate removable row in {label_relative_path}: {row}")
            if row not in lines and not all(add_row in lines for add_row in edit["add"]):
                raise ValueError(f"Expected row or completed edit in {label_relative_path}: {row}")
        for row in edit["add"]:
            if lines.count(row) > 1:
                raise ValueError(f"Duplicate added row in {label_relative_path}: {row}")


def apply_label_edits() -> None:
    removed_labels = removed_label_paths()
    for label_relative_path, edit in LABEL_EDITS.items():
        label_path = DATASET_DIR / label_relative_path
        if not label_path.is_file() and label_relative_path in removed_labels:
            continue
        lines = label_path.read_text(encoding="utf-8").splitlines()
        original = lines.copy()
        for row in edit["remove"]:
            if row in lines:
                lines.remove(row)
        for row in edit["add"]:
            if row not in lines:
                lines.append(row)
        if lines != original:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_confirmed_same_source_files() -> int:
    removed_count = 0
    for image_relative_path in FILES_TO_REMOVE:
        image_path = DATASET_DIR / image_relative_path
        label_path = image_to_label_path(image_relative_path)
        if image_path.exists() and label_path.exists():
            image_path.unlink()
            label_path.unlink()
            removed_count += 1
    return removed_count


def main() -> None:
    validate_inputs()
    apply_label_edits()
    removed_count = remove_confirmed_same_source_files()
    print("Final dataset processing complete.")
    print(f"Removed image/label pairs in this run: {removed_count}")
    print(f"Confirmed image/label pairs in final processing: {len(FILES_TO_REMOVE)}")
    print(f"Edited label files: {len(LABEL_EDITS)}")


if __name__ == "__main__":
    main()
