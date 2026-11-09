"""Dataset structure audit for the final PPE dataset."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageOps

from config.dataset import CLASS_NAMES, PPE_CLASS_NAMES
from config.paths import (
    DATASET_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    TEST_IMAGES,
    TEST_LABELS,
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS,
    YAML_FILE,
)
from config.training_config import IMAGE_SIZE


SPLITS = ("train", "valid", "test")
SPLIT_DIRS = {
    "train": (TRAIN_IMAGES, TRAIN_LABELS, "train"),
    "valid": (VALID_IMAGES, VALID_LABELS, "val"),
    "test": (TEST_IMAGES, TEST_LABELS, "test"),
}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_ID_TO_NAME = dict(enumerate(CLASS_NAMES))
CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}
PPE_CLASS_IDS = tuple(CLASS_NAME_TO_ID[name] for name in PPE_CLASS_NAMES)

EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
TABLES_DIR = EDA_OUTPUT_DIR / "tables"
FIGURES_DIR = EDA_OUTPUT_DIR / "figures"
EXAMPLES_DIR = EDA_OUTPUT_DIR / "examples"
SUMMARIES_DIR = EDA_OUTPUT_DIR / "summaries"


def project_relative(path: Path) -> str:
    """Return a project-relative path for stable generated outputs."""

    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def prepare_output_dirs(clean: bool = False) -> None:
    """Create the simplified EDA output directories."""

    if clean and EDA_OUTPUT_DIR.exists():
        target = EDA_OUTPUT_DIR.resolve()
        if OUTPUTS_DIR.resolve() not in target.parents:
            raise RuntimeError(f"Unexpected EDA output path: {target}")
        shutil.rmtree(target)

    for directory in (TABLES_DIR, FIGURES_DIR, EXAMPLES_DIR, SUMMARIES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write one CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write one JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return a file-content SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint() -> dict[str, Any]:
    """Return a deterministic content fingerprint for the dataset."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for path in sorted(p for p in DATASET_DIR.rglob("*") if p.is_file()):
        digest.update(path.relative_to(DATASET_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        file_count += 1
        total_bytes += path.stat().st_size

    return {
        "dataset_dir": project_relative(DATASET_DIR),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }


def audit_dataset() -> dict[str, Any]:
    """Inspect split structure, image-label pairing, and readable image dimensions."""

    dataset_yaml = yaml.safe_load(YAML_FILE.read_text(encoding="utf-8"))
    image_files, label_files = _scan_split_files()
    dimensions = _read_dimensions(image_files)
    structure = _structure_summary(image_files, label_files, dimensions)
    dimension_summary = _dimension_summary(dimensions)

    write_dataframe(structure, TABLES_DIR / "dataset_structure_summary.csv")
    write_dataframe(dimension_summary, TABLES_DIR / "image_dimension_summary.csv")
    _plot_file_counts(structure)
    _plot_dimensions(dimensions)

    summary = {
        "dataset_yaml": dataset_yaml,
        "class_names": list(CLASS_NAMES),
        "image_size_for_training": IMAGE_SIZE,
        "total_image_files": int(structure.loc[structure["split"] == "overall", "image_files"].iloc[0]),
        "total_label_files": int(structure.loc[structure["split"] == "overall", "label_files"].iloc[0]),
        "missing_label_files": int(structure.loc[structure["split"] == "overall", "missing_label_files"].iloc[0]),
        "missing_image_files": int(structure.loc[structure["split"] == "overall", "missing_image_files"].iloc[0]),
        "unsupported_image_files": int(structure.loc[structure["split"] == "overall", "unsupported_image_files"].iloc[0]),
        "unreadable_images": int(structure.loc[structure["split"] == "overall", "unreadable_images"].iloc[0]),
        "empty_label_files": int(structure.loc[structure["split"] == "overall", "empty_label_files"].iloc[0]),
        "yaml_paths_consistent": _yaml_paths_consistent(dataset_yaml),
        "split_stem_overlap_count": _split_stem_overlap_count(image_files),
        "image_dimension": dimension_summary[dimension_summary["split"] == "overall"].iloc[0].to_dict(),
    }

    return {
        "dataset_yaml": dataset_yaml,
        "image_files": image_files,
        "label_files": label_files,
        "dimensions": dimensions,
        "structure_summary": structure,
        "dimension_summary": dimension_summary,
        "summary": summary,
    }


def _scan_split_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for split, (image_dir, label_dir, _) in SPLIT_DIRS.items():
        for path in sorted(image_dir.iterdir()):
            if path.is_file():
                image_rows.append(
                    {
                        "split": split,
                        "image_file": project_relative(path),
                        "image_name": path.name,
                        "stem": path.stem,
                        "extension": path.suffix.lower(),
                        "is_supported_extension": path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS,
                        "size_bytes": path.stat().st_size,
                        "_path": path,
                    }
                )

        for path in sorted(label_dir.iterdir()):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                non_empty_lines = sum(1 for line in text.splitlines() if line.strip())
                label_rows.append(
                    {
                        "split": split,
                        "label_file": project_relative(path),
                        "label_name": path.name,
                        "stem": path.stem,
                        "is_empty_label_file": non_empty_lines == 0,
                        "non_empty_line_count": non_empty_lines,
                        "_path": path,
                    }
                )

    return pd.DataFrame(image_rows), pd.DataFrame(label_rows)


def _read_dimensions(image_files: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in image_files.to_dict("records"):
        width = height = aspect_ratio = None
        is_readable = False
        error = ""

        if record["is_supported_extension"]:
            try:
                with Image.open(record["_path"]) as image:
                    image = ImageOps.exif_transpose(image)
                    width, height = image.size
                    image.verify()
                aspect_ratio = width / height if height else math.nan
                is_readable = True
            except Exception as exc:  # noqa: BLE001 - record the dataset problem, do not stop.
                error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "split": record["split"],
                "image_file": record["image_file"],
                "is_supported_extension": record["is_supported_extension"],
                "is_readable": is_readable,
                "image_width": width,
                "image_height": height,
                "aspect_ratio": aspect_ratio,
                "read_error": error,
            }
        )

    return pd.DataFrame(rows)


def _structure_summary(
    image_files: pd.DataFrame,
    label_files: pd.DataFrame,
    dimensions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for split in (*SPLITS, "overall"):
        images = image_files if split == "overall" else image_files[image_files["split"] == split]
        labels = label_files if split == "overall" else label_files[label_files["split"] == split]
        dims = dimensions if split == "overall" else dimensions[dimensions["split"] == split]

        image_stems = set(images["stem"])
        label_stems = set(labels["stem"])
        rows.append(
            {
                "split": split,
                "image_files": len(images),
                "label_files": len(labels),
                "paired_image_label_stems": len(image_stems & label_stems),
                "missing_label_files": len(image_stems - label_stems),
                "missing_image_files": len(label_stems - image_stems),
                "unsupported_image_files": int((~images["is_supported_extension"]).sum()) if not images.empty else 0,
                "readable_images": int(dims["is_readable"].sum()) if not dims.empty else 0,
                "unreadable_images": int((dims["is_supported_extension"] & ~dims["is_readable"]).sum()) if not dims.empty else 0,
                "empty_label_files": int(labels["is_empty_label_file"].sum()) if not labels.empty else 0,
            }
        )

    return pd.DataFrame(rows)


def _dimension_summary(dimensions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for split in (*SPLITS, "overall"):
        group = dimensions[dimensions["is_readable"]]
        if split != "overall":
            group = group[group["split"] == split]

        rows.append(
            {
                "split": split,
                "readable_images": len(group),
                "width_min": int(group["image_width"].min()) if not group.empty else 0,
                "width_median": float(group["image_width"].median()) if not group.empty else 0.0,
                "width_max": int(group["image_width"].max()) if not group.empty else 0,
                "height_min": int(group["image_height"].min()) if not group.empty else 0,
                "height_median": float(group["image_height"].median()) if not group.empty else 0.0,
                "height_max": int(group["image_height"].max()) if not group.empty else 0,
                "aspect_ratio_min": float(group["aspect_ratio"].min()) if not group.empty else 0.0,
                "aspect_ratio_median": float(group["aspect_ratio"].median()) if not group.empty else 0.0,
                "aspect_ratio_max": float(group["aspect_ratio"].max()) if not group.empty else 0.0,
                "distinct_dimensions": int(group[["image_width", "image_height"]].drop_duplicates().shape[0]) if not group.empty else 0,
            }
        )

    return pd.DataFrame(rows)


def _yaml_paths_consistent(dataset_yaml: dict[str, Any]) -> bool:
    if dataset_yaml.get("path") != ".":
        return False

    for split, (image_dir, label_dir, yaml_key) in SPLIT_DIRS.items():
        yaml_dir = (DATASET_DIR / dataset_yaml.get(yaml_key, "")).resolve()
        if yaml_dir != image_dir.resolve() or not label_dir.exists():
            return False

    return True


def _split_stem_overlap_count(image_files: pd.DataFrame) -> int:
    split_counts = image_files.groupby("stem")["split"].nunique()
    return int((split_counts > 1).sum())


def _plot_file_counts(structure: pd.DataFrame) -> None:
    split_rows = structure[structure["split"].isin(SPLITS)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x_positions = range(len(split_rows))
    ax.bar([x - 0.18 for x in x_positions], split_rows["image_files"], width=0.36, label="Images")
    ax.bar([x + 0.18 for x in x_positions], split_rows["label_files"], width=0.36, label="Label files")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(split_rows["split"])
    ax.set_ylabel("File count")
    ax.set_title("Image and Label File Counts by Split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_file_counts_by_split.png", dpi=150)
    plt.close(fig)


def _plot_dimensions(dimensions: pd.DataFrame) -> None:
    readable = dimensions[dimensions["is_readable"]]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for split in SPLITS:
        group = readable[readable["split"] == split]
        ax.scatter(group["image_width"], group["image_height"], s=18, alpha=0.55, label=split)
    ax.set_xlabel("Image width (pixels)")
    ax.set_ylabel("Image height (pixels)")
    ax.set_title("Image Dimensions by Split")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_dimensions_scatter.png", dpi=150)
    plt.close(fig)
