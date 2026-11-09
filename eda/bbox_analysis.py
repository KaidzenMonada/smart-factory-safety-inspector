"""Bounding-box size and geometry analysis."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.dataset import CLASS_NAMES, PPE_CLASS_NAMES
from config.training_config import IMAGE_SIZE
from eda.dataset_audit import FIGURES_DIR, TABLES_DIR, write_dataframe


VERY_SMALL_OBJECT_EDGE_PX = 16.0
SMALL_OBJECT_EDGE_PX = 32.0
MEDIUM_OBJECT_EDGE_PX = 96.0
VERY_THIN_EDGE_PX = 4.0
EXTREME_ASPECT_RATIO = 6.0
UNUSUALLY_LARGE_NORM_AREA = 0.80


def analyze_bounding_boxes(dataset_state: dict[str, Any], annotations: pd.DataFrame) -> dict[str, Any]:
    """Measure usable boxes and flag review candidates without calling them errors."""

    measurements = _measure_boxes(dataset_state, annotations)
    summary_by_class = _summary_by_class(measurements)
    review_candidates = _review_candidates(measurements)

    write_dataframe(summary_by_class, TABLES_DIR / "bbox_summary_by_class.csv")
    write_dataframe(review_candidates, TABLES_DIR / "bbox_review_candidates.csv")

    _plot_estimated_sizes(measurements)
    _plot_size_categories(measurements)

    ppe_small = review_candidates[
        review_candidates["candidate_type"].isin(["very_small_box", "small_ppe_box"])
        & review_candidates["class_name"].isin(PPE_CLASS_NAMES)
    ]
    summary = {
        "image_size_for_training": IMAGE_SIZE,
        "total_measured_boxes": len(measurements),
        "review_candidate_count": len(review_candidates),
        "review_candidate_counts": review_candidates["candidate_type"].value_counts().astype(int).to_dict(),
        "small_or_very_small_ppe_candidate_count": len(ppe_small),
        "very_small_boxes_by_class": {
            row["class_name"]: int(row["very_small_boxes"]) for row in summary_by_class.to_dict("records")
        },
        "small_boxes_by_class": {
            row["class_name"]: int(row["small_boxes"]) for row in summary_by_class.to_dict("records")
        },
    }

    return {
        "measurements": measurements,
        "summary_by_class": summary_by_class,
        "review_candidates": review_candidates,
        "summary": summary,
    }


def _measure_boxes(dataset_state: dict[str, Any], annotations: pd.DataFrame) -> pd.DataFrame:
    usable = annotations[annotations["is_usable_annotation"]].copy()
    dimensions = dataset_state["dimensions"][["split", "image_file", "is_readable", "image_width", "image_height"]]
    data = usable.merge(dimensions, on=["split", "image_file"], how="left")
    data = data[data["is_readable"]].copy()

    data["box_area_norm"] = data["box_width_norm"] * data["box_height_norm"]
    data["box_aspect_ratio"] = data["box_width_norm"] / data["box_height_norm"]
    data["box_width_px"] = data["box_width_norm"] * data["image_width"]
    data["box_height_px"] = data["box_height_norm"] * data["image_height"]
    scale = np.minimum(IMAGE_SIZE / data["image_width"], IMAGE_SIZE / data["image_height"])
    data["box_width_640"] = data["box_width_px"] * scale
    data["box_height_640"] = data["box_height_px"] * scale
    data["box_sqrt_area_640"] = np.sqrt(data["box_width_640"] * data["box_height_640"])
    data["size_category_640"] = data["box_sqrt_area_640"].apply(_size_category)

    return data.sort_values(["split", "image_file", "row_number"])


def _size_category(sqrt_area: float) -> str:
    if sqrt_area < VERY_SMALL_OBJECT_EDGE_PX:
        return "very_small"
    if sqrt_area < SMALL_OBJECT_EDGE_PX:
        return "small"
    if sqrt_area < MEDIUM_OBJECT_EDGE_PX:
        return "medium"
    return "large"


def _summary_by_class(measurements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_name in CLASS_NAMES:
        group = measurements[measurements["class_name"] == class_name]
        rows.append(
            {
                "class_name": class_name,
                "box_count": len(group),
                "width_640_median": float(group["box_width_640"].median()) if not group.empty else 0.0,
                "height_640_median": float(group["box_height_640"].median()) if not group.empty else 0.0,
                "sqrt_area_640_median": float(group["box_sqrt_area_640"].median()) if not group.empty else 0.0,
                "sqrt_area_640_p10": float(group["box_sqrt_area_640"].quantile(0.10)) if not group.empty else 0.0,
                "sqrt_area_640_p90": float(group["box_sqrt_area_640"].quantile(0.90)) if not group.empty else 0.0,
                "very_small_boxes": int((group["size_category_640"] == "very_small").sum()),
                "small_boxes": int((group["size_category_640"] == "small").sum()),
                "medium_boxes": int((group["size_category_640"] == "medium").sum()),
                "large_boxes": int((group["size_category_640"] == "large").sum()),
            }
        )
    return pd.DataFrame(rows)


def _review_candidates(measurements: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in measurements.to_dict("records"):
        candidates = []
        min_edge = min(record["box_width_640"], record["box_height_640"])
        max_edge = max(record["box_width_640"], record["box_height_640"])
        aspect_ratio = record["box_aspect_ratio"]
        inverse_aspect_ratio = 1.0 / aspect_ratio if aspect_ratio else float("inf")

        if record["box_sqrt_area_640"] < VERY_SMALL_OBJECT_EDGE_PX:
            candidates.append("very_small_box")
        elif record["class_name"] in PPE_CLASS_NAMES and record["box_sqrt_area_640"] < SMALL_OBJECT_EDGE_PX:
            candidates.append("small_ppe_box")
        if min_edge < VERY_THIN_EDGE_PX and max_edge >= VERY_SMALL_OBJECT_EDGE_PX:
            candidates.append("unusually_thin_box")
        if aspect_ratio >= EXTREME_ASPECT_RATIO or inverse_aspect_ratio >= EXTREME_ASPECT_RATIO:
            candidates.append("extreme_aspect_ratio")
        if record["box_area_norm"] >= UNUSUALLY_LARGE_NORM_AREA:
            candidates.append("unusually_large_box")

        for candidate in candidates:
            rows.append(
                {
                    "split": record["split"],
                    "image_file": record["image_file"],
                    "label_file": record["label_file"],
                    "row_number": record["row_number"],
                    "row_id": record["row_id"],
                    "class_name": record["class_name"],
                    "candidate_type": candidate,
                    "review_level": "HEURISTIC_REVIEW",
                    "box_width_norm": record["box_width_norm"],
                    "box_height_norm": record["box_height_norm"],
                    "box_area_norm": record["box_area_norm"],
                    "box_aspect_ratio": record["box_aspect_ratio"],
                    "box_width_640": record["box_width_640"],
                    "box_height_640": record["box_height_640"],
                    "box_sqrt_area_640": record["box_sqrt_area_640"],
                }
            )

    return pd.DataFrame(rows).sort_values(["candidate_type", "class_name", "split", "image_file", "row_number"])


def _plot_estimated_sizes(measurements: pd.DataFrame) -> None:
    data_by_class = [measurements[measurements["class_name"] == class_name]["box_sqrt_area_640"] for class_name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.boxplot(data_by_class, labels=CLASS_NAMES, showfliers=False)
    ax.axhline(VERY_SMALL_OBJECT_EDGE_PX, color="#E15759", linestyle="--", linewidth=1.1, label="Very small")
    ax.axhline(SMALL_OBJECT_EDGE_PX, color="#F28E2B", linestyle="--", linewidth=1.1, label="Small")
    ax.set_ylabel("Square-root box area at 640 input (pixels)")
    ax.set_title("Estimated Object Size After Letterbox Resize")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bbox_estimated_size_640_by_class.png", dpi=150)
    plt.close(fig)


def _plot_size_categories(measurements: pd.DataFrame) -> None:
    pivot = (
        measurements.pivot_table(index="class_name", columns="size_category_640", values="row_id", aggfunc="count", fill_value=0)
        .reindex(list(CLASS_NAMES))
        .reindex(columns=["very_small", "small", "medium", "large"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bottom = np.zeros(len(pivot))
    colors = {"very_small": "#E15759", "small": "#F28E2B", "medium": "#59A14F", "large": "#4E79A7"}
    for category in pivot.columns:
        ax.bar(pivot.index, pivot[category], bottom=bottom, label=category.replace("_", " "), color=colors[category])
        bottom += pivot[category].to_numpy()
    ax.set_ylabel("Box count")
    ax.set_title("Object Size Categories by Class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bbox_size_category_by_class.png", dpi=150)
    plt.close(fig)
