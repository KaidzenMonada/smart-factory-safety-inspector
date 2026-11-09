"""Deterministic visual examples for manual EDA review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from config.dataset import PERSON_CLASS_NAME, PPE_CLASS_NAMES
from config.paths import PROJECT_ROOT
from eda.dataset_audit import EXAMPLES_DIR, TABLES_DIR, project_relative, write_dataframe
from src.path_utils import safe_filename_stem


BOX_COLORS = {
    "person": "#1F77B4",
    "helmet": "#F2C94C",
    "vest": "#2CA02C",
    "gloves": "#D62728",
    "goggles": "#17BECF",
}
HIGHLIGHT_COLOR = "#FF00FF"
MAX_EDGE = 1050
PAIR_PANEL_HEIGHT = 500


def generate_visual_examples(
    annotations: pd.DataFrame,
    validation_issues: pd.DataFrame,
    bbox_candidates: pd.DataFrame,
    person_ppe_results: dict[str, Any],
    duplicate_results: dict[str, Any],
) -> pd.DataFrame:
    """Render a small, deterministic set of representative examples."""

    usable = annotations[annotations["is_usable_annotation"]].copy()
    image_stats = _image_stats(usable)
    assignments = person_ppe_results["assignments"]
    unassigned = person_ppe_results["unassigned"]
    multiple = person_ppe_results["multiple_same_class"]
    near_pairs = duplicate_results["near_duplicate_candidates"]
    records: list[dict[str, Any]] = []

    def add_image(category: str, image_file: str, reason: str, source_table: str, highlights: list[str] | None = None) -> None:
        if not image_file or not (PROJECT_ROOT / image_file).exists():
            return
        index = sum(1 for row in records if row["category"] == category) + 1
        output = EXAMPLES_DIR / f"{category}_{index:02d}_{safe_filename_stem(Path(image_file).stem)}.png"
        _save_annotated_image(usable, image_file, output, category, highlights or [])
        records.append(_manifest_row(category, output, image_file, "", reason, source_table, highlights or []))

    def add_pair(category: str, row: dict[str, Any], reason: str, source_table: str) -> None:
        image_a = row.get("image_a", "")
        image_b = row.get("image_b", "")
        if not image_a or not image_b:
            return
        index = sum(1 for item in records if item["category"] == category) + 1
        output = EXAMPLES_DIR / f"{category}_{index:02d}_{safe_filename_stem(Path(image_a).stem)}__{safe_filename_stem(Path(image_b).stem)}.png"
        _save_pair_image(usable, image_a, image_b, output, row)
        records.append(_manifest_row(category, output, image_a, image_b, reason, source_table, []))

    normal = image_stats[(image_stats["person_count"] == 1) & (image_stats["ppe_count"].between(1, 5))]
    for row in normal.sort_values(["split", "total_annotations", "image_file"]).head(1).to_dict("records"):
        add_image("normal_annotation", row["image_file"], "One person with a modest number of PPE labels.", "class_counts_by_split.csv")

    multi_dense = image_stats[(image_stats["person_count"] >= 2) | (image_stats["ppe_count"] >= 6)]
    for row in multi_dense.sort_values(["total_annotations", "split", "image_file"], ascending=[False, True, True]).head(2).to_dict("records"):
        add_image("multi_person_dense_scene", row["image_file"], f"{row['person_count']} people and {row['ppe_count']} PPE labels.", "class_counts_by_split.csv")

    small_ppe = bbox_candidates[
        bbox_candidates["candidate_type"].isin(["very_small_box", "small_ppe_box"])
        & bbox_candidates["class_name"].isin(PPE_CLASS_NAMES)
    ].sort_values(["box_sqrt_area_640", "split", "image_file", "row_number"])
    for row in _unique_images(small_ppe, 3):
        add_image("small_ppe", row["image_file"], f"{row['class_name']} small-object candidate.", "bbox_review_candidates.csv", [row["row_id"]])

    boundary = validation_issues[validation_issues["issue_type"].str.contains("boundary_overflow", na=False)]
    for row in _unique_images(boundary.sort_values(["split", "image_file", "row_number"]), 2):
        add_image("boundary_overflow", row["image_file"], f"{row['issue_type']} review case.", "boundary_overflow_rows.csv", [row["row_id"]])

    for row in _unique_images(unassigned.sort_values(["relationship_category", "split", "image_file"]), 2):
        add_image("unassigned_ppe", row["image_file"], f"{row['ppe_class']} is unassigned: {row['relationship_category']}.", "unassigned_ppe.csv", [row["ppe_row_id"]])

    multiple_sample = multiple.copy()
    multiple_sample["priority"] = multiple_sample["ppe_class"].map({"helmet": 0, "vest": 1, "goggles": 2, "gloves": 3}).fillna(9)
    for row in _unique_images(multiple_sample.sort_values(["priority", "associated_box_count", "split", "image_file"], ascending=[True, False, True, True]), 2):
        highlights = _same_class_rows(assignments, row)
        add_image("multiple_same_class_ppe", row["image_file"], f"Person {row['person_id']} has {row['associated_box_count']} {row['ppe_class']} boxes.", "multiple_same_class_ppe_per_person.csv", highlights)

    for row in near_pairs.sort_values(["cross_split", "perceptual_distance", "image_a"], ascending=[False, True, True]).head(2).to_dict("records"):
        add_pair("near_duplicate_candidate", row, "Perceptual hash candidate; not a confirmed duplicate.", "near_duplicate_candidates.csv")

    manifest = pd.DataFrame(records)
    write_dataframe(manifest, TABLES_DIR / "visual_examples_manifest.csv")
    return manifest


def _manifest_row(
    category: str,
    output: Path,
    image_a: str,
    image_b: str,
    reason: str,
    source_table: str,
    highlights: list[str],
) -> dict[str, str]:
    return {
        "category": category,
        "example_file": project_relative(output),
        "source_image_a": image_a,
        "source_image_b": image_b,
        "reason": reason,
        "highlighted_rows": "; ".join(highlights),
        "source_table": source_table,
    }


def _image_stats(annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, image_file), group in annotations.groupby(["split", "image_file"], sort=True):
        rows.append(
            {
                "split": split,
                "image_file": image_file,
                "person_count": int((group["class_name"] == PERSON_CLASS_NAME).sum()),
                "ppe_count": int(group["class_name"].isin(PPE_CLASS_NAMES).sum()),
                "total_annotations": len(group),
            }
        )
    return pd.DataFrame(rows)


def _unique_images(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in df.to_dict("records"):
        if row["image_file"] in seen:
            continue
        seen.add(row["image_file"])
        rows.append(row)
        if len(rows) == limit:
            break
    return rows


def _same_class_rows(assignments: pd.DataFrame, row: dict[str, Any]) -> list[str]:
    group = assignments[
        (assignments["split"] == row["split"])
        & (assignments["image_file"] == row["image_file"])
        & (assignments["assigned_person_id"] == row["person_id"])
        & (assignments["ppe_class"] == row["ppe_class"])
    ]
    return sorted(group["ppe_row_id"].tolist())


def _save_annotated_image(
    annotations: pd.DataFrame,
    image_file: str,
    output: Path,
    title: str,
    highlights: list[str],
) -> None:
    image = _annotated_image(annotations, image_file, title, set(highlights))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _annotated_image(annotations: pd.DataFrame, image_file: str, title: str, highlights: set[str]) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(PROJECT_ROOT / image_file)).convert("RGB")
    original_width, original_height = image.size
    scale = min(MAX_EDGE / original_width, MAX_EDGE / original_height, 1.0)
    if scale < 1.0:
        image = image.resize((int(original_width * scale), int(original_height * scale)), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row in annotations[annotations["image_file"] == image_file].sort_values("row_number").to_dict("records"):
        color = HIGHLIGHT_COLOR if row["row_id"] in highlights else BOX_COLORS.get(row["class_name"], "white")
        line_width = 5 if row["row_id"] in highlights else 3
        box = (
            row["x_min"] * original_width * scale,
            row["y_min"] * original_height * scale,
            row["x_max"] * original_width * scale,
            row["y_max"] * original_height * scale,
        )
        draw.rectangle(box, outline=color, width=line_width)
        _draw_label(draw, box, row["class_name"], color, font)

    banner_height = 32
    framed = Image.new("RGB", (image.width, image.height + banner_height), "white")
    framed.paste(image, (0, banner_height))
    banner = ImageDraw.Draw(framed)
    banner.rectangle((0, 0, framed.width, banner_height), fill="#222222")
    banner.text((8, 9), f"{title.replace('_', ' ').title()} | {image_file}"[:160], fill="white", font=font)
    return framed


def _draw_label(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], text: str, color: str, font: ImageFont.ImageFont) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x_min, y_min, _, _ = box
    label_y = max(0, y_min - height - 4)
    draw.rectangle((x_min, label_y, x_min + width + 6, label_y + height + 4), fill=color)
    draw.text((x_min + 3, label_y + 2), text, fill="black", font=font)


def _save_pair_image(annotations: pd.DataFrame, image_a: str, image_b: str, output: Path, row: dict[str, Any]) -> None:
    left = _resize_to_height(_annotated_image(annotations, image_a, "near duplicate A", set()), PAIR_PANEL_HEIGHT)
    right = _resize_to_height(_annotated_image(annotations, image_b, "near duplicate B", set()), PAIR_PANEL_HEIGHT)
    padding = 14
    header_height = 36
    canvas = Image.new("RGB", (left.width + right.width + padding * 3, PAIR_PANEL_HEIGHT + header_height + padding * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, header_height), fill="#222222")
    draw.text((padding, 11), f"Near-Duplicate Candidate | distance {row['perceptual_distance']} | cross split: {row['cross_split']}", fill="white", font=font)
    canvas.paste(left, (padding, header_height + padding))
    canvas.paste(right, (left.width + padding * 2, header_height + padding))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    return image.resize((int(image.width * height / image.height), height), Image.Resampling.LANCZOS)
