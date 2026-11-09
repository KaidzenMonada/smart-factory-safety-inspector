"""Class distribution and co-occurrence analysis."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.dataset import CLASS_NAMES, PERSON_CLASS_NAME, PPE_CLASS_NAMES
from eda.dataset_audit import CLASS_ID_TO_NAME, FIGURES_DIR, SPLITS, TABLES_DIR, write_dataframe


def analyze_class_distribution(dataset_state: dict[str, Any], annotations: pd.DataFrame) -> dict[str, Any]:
    """Count usable annotations by class, split, image, and class co-occurrence."""

    usable = annotations[annotations["is_usable_annotation"]].copy()
    image_counts = _image_counts(dataset_state, usable)
    class_counts = _class_counts(usable)
    split_counts = _class_counts_by_split(usable)
    cooccurrence = _cooccurrence(usable)

    write_dataframe(class_counts, TABLES_DIR / "class_counts.csv")
    write_dataframe(split_counts, TABLES_DIR / "class_counts_by_split.csv")
    write_dataframe(cooccurrence, TABLES_DIR / "class_cooccurrence.csv")

    _plot_class_counts(split_counts)
    _plot_cooccurrence(cooccurrence, usable)

    nonzero = class_counts[class_counts["annotation_count"] > 0]["annotation_count"]
    summary = {
        "total_usable_annotations": len(usable),
        "overall_class_counts": {row["class_name"]: int(row["annotation_count"]) for row in class_counts.to_dict("records")},
        "class_counts_by_split": {
            f"{row['split']}:{row['class_name']}": int(row["annotation_count"])
            for row in split_counts.to_dict("records")
        },
        "imbalance_ratio_max_to_min_nonzero": float(nonzero.max() / nonzero.min()) if len(nonzero) else 0.0,
        "most_common_class": _class_extreme(class_counts, ascending=False),
        "rarest_class": _class_extreme(class_counts, ascending=True),
        "annotations_per_image": {
            "min": int(image_counts["total_annotations"].min()),
            "median": float(image_counts["total_annotations"].median()),
            "max": int(image_counts["total_annotations"].max()),
        },
        "people_per_image": {
            "images_with_no_person": int((image_counts["person_count"] == 0).sum()),
            "images_with_multiple_people": int((image_counts["person_count"] >= 2).sum()),
            "max_people_in_image": int(image_counts["person_count"].max()),
        },
        "ppe_per_image": {
            "images_with_no_ppe": int((image_counts["ppe_count"] == 0).sum()),
            "max_ppe_objects_in_image": int(image_counts["ppe_count"].max()),
        },
    }

    return {
        "usable_annotations": usable,
        "image_counts": image_counts,
        "class_counts": class_counts,
        "class_counts_by_split": split_counts,
        "cooccurrence": cooccurrence,
        "summary": summary,
    }


def _class_counts(annotations: pd.DataFrame) -> pd.DataFrame:
    counts = annotations["class_name"].value_counts().to_dict()
    total = len(annotations)
    rows = []

    for class_id, class_name in CLASS_ID_TO_NAME.items():
        count = int(counts.get(class_name, 0))
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "annotation_count": count,
                "proportion": count / total if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _class_counts_by_split(annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split in SPLITS:
        split_rows = annotations[annotations["split"] == split]
        split_total = len(split_rows)
        counts = split_rows["class_name"].value_counts().to_dict()
        for class_id, class_name in CLASS_ID_TO_NAME.items():
            count = int(counts.get(class_name, 0))
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "annotation_count": count,
                    "split_annotation_total": split_total,
                    "split_proportion": count / split_total if split_total else 0.0,
                }
            )

    return pd.DataFrame(rows)


def _image_counts(dataset_state: dict[str, Any], annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = annotations.groupby(["split", "image_file", "class_name"]).size().reset_index(name="count")

    for image in dataset_state["image_files"].to_dict("records"):
        group = grouped[(grouped["split"] == image["split"]) & (grouped["image_file"] == image["image_file"])]
        counts = {row["class_name"]: int(row["count"]) for row in group.to_dict("records")}
        rows.append(
            {
                "split": image["split"],
                "image_file": image["image_file"],
                "person_count": counts.get(PERSON_CLASS_NAME, 0),
                "ppe_count": sum(counts.get(name, 0) for name in PPE_CLASS_NAMES),
                "total_annotations": sum(counts.values()),
            }
        )

    return pd.DataFrame(rows)


def _cooccurrence(annotations: pd.DataFrame) -> pd.DataFrame:
    image_classes = annotations.groupby(["split", "image_file"])["class_name"].apply(lambda values: set(values)).reset_index()
    rows = []

    for class_a, class_b in combinations(CLASS_NAMES, 2):
        with_a = image_classes[image_classes["class_name"].apply(lambda classes: class_a in classes)]
        with_b = image_classes[image_classes["class_name"].apply(lambda classes: class_b in classes)]
        with_both = image_classes[image_classes["class_name"].apply(lambda classes: class_a in classes and class_b in classes)]
        union_count = len(pd.concat([with_a, with_b]).drop_duplicates(["split", "image_file"]))
        rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "images_with_class_a": len(with_a),
                "images_with_class_b": len(with_b),
                "images_with_both": len(with_both),
                "jaccard": len(with_both) / union_count if union_count else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _class_extreme(class_counts: pd.DataFrame, ascending: bool) -> dict[str, Any]:
    row = class_counts.sort_values(["annotation_count", "class_id"], ascending=[ascending, True]).iloc[0]
    return {"class_name": row["class_name"], "annotation_count": int(row["annotation_count"])}


def _plot_class_counts(split_counts: pd.DataFrame) -> None:
    pivot = split_counts.pivot(index="class_name", columns="split", values="annotation_count").loc[list(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x_positions = np.arange(len(pivot))
    width = 0.24
    for index, split in enumerate(SPLITS):
        ax.bar(x_positions + (index - 1) * width, pivot[split], width=width, label=split)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Annotation count")
    ax.set_title("Class Annotation Counts by Split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_counts_by_split.png", dpi=150)
    plt.close(fig)


def _plot_cooccurrence(cooccurrence: pd.DataFrame, annotations: pd.DataFrame) -> None:
    matrix = pd.DataFrame(0, index=list(CLASS_NAMES), columns=list(CLASS_NAMES), dtype=int)
    image_classes = annotations.groupby(["split", "image_file"])["class_name"].apply(lambda values: set(values)).reset_index()

    for class_name in CLASS_NAMES:
        matrix.loc[class_name, class_name] = int(image_classes["class_name"].apply(lambda classes: class_name in classes).sum())
    for row in cooccurrence.to_dict("records"):
        matrix.loc[row["class_a"], row["class_b"]] = int(row["images_with_both"])
        matrix.loc[row["class_b"], row["class_a"]] = int(row["images_with_both"])

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("Class Co-occurrence by Image")
    for row_index in range(len(CLASS_NAMES)):
        for col_index in range(len(CLASS_NAMES)):
            ax.text(col_index, row_index, matrix.iloc[row_index, col_index], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Image count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_cooccurrence_heatmap.png", dpi=150)
    plt.close(fig)
