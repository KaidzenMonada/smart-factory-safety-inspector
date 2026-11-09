"""Ground-truth person/PPE relationship analysis."""

from __future__ import annotations

from math import hypot
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.dataset import PERSON_CLASS_NAME, PPE_CLASS_NAMES
from eda.dataset_audit import FIGURES_DIR, TABLES_DIR, write_dataframe
from src.compliance import bbox_center, choose_person_for_ppe, normalized_center_distance


NEAR_PERSON_BOUNDARY_DISTANCE_NORM = 0.03
CLEARLY_FAR_FROM_PERSON_DISTANCE_NORM = 0.10


def analyze_person_ppe_relationships(annotations: pd.DataFrame) -> dict[str, Any]:
    """Assign PPE to people with the preserved center-containment rule."""

    usable = annotations[annotations["is_usable_annotation"]].copy()
    assignment_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []
    multiple_rows: list[dict[str, Any]] = []

    for (split, image_file), group in usable.groupby(["split", "image_file"], sort=True):
        people = _person_records(group)
        ppe_items = _ppe_records(group)
        label_file = group["label_file"].iloc[0]

        for ppe in ppe_items:
            owner = choose_person_for_ppe(ppe, people)
            if owner is None:
                unassigned_rows.append(_unassigned_record(split, image_file, label_file, ppe, people))
                continue

            owner["assigned_ppe"].append(ppe)
            assignment_rows.append(
                {
                    "split": split,
                    "image_file": image_file,
                    "label_file": label_file,
                    "ppe_row_id": ppe["row_id"],
                    "ppe_row_number": ppe["row_number"],
                    "ppe_class": ppe["class"],
                    "assigned_person_id": owner["person_id"],
                    "person_row_id": owner["person_row_id"],
                }
            )

        for person in people:
            counts = {class_name: 0 for class_name in PPE_CLASS_NAMES}
            for ppe in person["assigned_ppe"]:
                counts[ppe["class"]] += 1
            person_rows.append(
                {
                    "split": split,
                    "image_file": image_file,
                    "person_id": person["person_id"],
                    "total_assigned_ppe": sum(counts.values()),
                    **{f"{class_name}_count": counts[class_name] for class_name in PPE_CLASS_NAMES},
                }
            )
            for class_name, count in counts.items():
                if count > 1:
                    multiple_rows.append(
                        {
                            "split": split,
                            "image_file": image_file,
                            "label_file": label_file,
                            "person_id": person["person_id"],
                            "person_row_id": person["person_row_id"],
                            "ppe_class": class_name,
                            "associated_box_count": count,
                            "review_level": "HEURISTIC_REVIEW",
                            "interpretation": "HEURISTIC: multiple same-class PPE boxes are visual review candidates, not confirmed annotation errors.",
                        }
                    )

    assignments = pd.DataFrame(assignment_rows)
    unassigned = pd.DataFrame(unassigned_rows)
    person_counts = pd.DataFrame(person_rows)
    multiple_same_class = pd.DataFrame(multiple_rows)
    unassigned_summary = _unassigned_summary(unassigned)
    multiple_summary = _multiple_summary(multiple_same_class)

    write_dataframe(unassigned, TABLES_DIR / "unassigned_ppe.csv")
    write_dataframe(unassigned_summary, TABLES_DIR / "unassigned_ppe_summary.csv")
    write_dataframe(multiple_same_class, TABLES_DIR / "multiple_same_class_ppe_per_person.csv")
    write_dataframe(multiple_summary, TABLES_DIR / "multiple_same_class_ppe_summary.csv")

    _plot_assignment_counts(assignments, unassigned)
    _plot_unassigned_reasons(unassigned)
    _plot_multiple_same_class(multiple_same_class)

    summary = {
        "association_rule": "PPE center must fall inside a person box; if several people contain it, the smallest person box wins, then nearest center, then person order.",
        "total_ppe_annotations_considered": len(assignments) + len(unassigned),
        "assigned_ppe_count": len(assignments),
        "unassigned_ppe_count": len(unassigned),
        "unassigned_ppe_by_class": unassigned["ppe_class"].value_counts().astype(int).to_dict() if not unassigned.empty else {},
        "unassigned_ppe_images": int(unassigned["image_file"].nunique()) if not unassigned.empty else 0,
        "unassigned_ppe_with_no_person_annotation": int((unassigned["relationship_category"] == "no_person_annotation").sum()) if not unassigned.empty else 0,
        "unassigned_ppe_near_person_boundary": int((unassigned["relationship_category"] == "near_person_boundary").sum()) if not unassigned.empty else 0,
        "unassigned_ppe_clearly_far": int((unassigned["relationship_category"] == "clearly_far_from_person_box").sum()) if not unassigned.empty else 0,
        "people_with_any_assigned_ppe": int((person_counts["total_assigned_ppe"] > 0).sum()) if not person_counts.empty else 0,
        "people_with_no_assigned_ppe": int((person_counts["total_assigned_ppe"] == 0).sum()) if not person_counts.empty else 0,
        "multiple_same_class_ppe_person_cases": len(multiple_same_class),
        "multiple_same_class_ppe_by_class": multiple_same_class["ppe_class"].value_counts().astype(int).to_dict() if not multiple_same_class.empty else {},
    }

    return {
        "assignments": assignments,
        "unassigned": unassigned,
        "unassigned_summary": unassigned_summary,
        "multiple_same_class": multiple_same_class,
        "multiple_summary": multiple_summary,
        "summary": summary,
    }


def _person_records(group: pd.DataFrame) -> list[dict[str, Any]]:
    people = group[group["class_name"] == PERSON_CLASS_NAME].sort_values("row_number")
    records = []
    for index, row in enumerate(people.to_dict("records"), start=1):
        records.append(
            {
                "person_id": index,
                "person_row_id": row["row_id"],
                "person_bbox": [row["x_min"], row["y_min"], row["x_max"], row["y_max"]],
                "assigned_ppe": [],
            }
        )
    return records


def _ppe_records(group: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in group[group["class_name"].isin(PPE_CLASS_NAMES)].sort_values("row_number").to_dict("records"):
        bbox = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        center_x, center_y = bbox_center(bbox)
        records.append(
            {
                "row_id": row["row_id"],
                "row_number": row["row_number"],
                "class": row["class_name"],
                "bbox_xyxy": bbox,
                "center_x": center_x,
                "center_y": center_y,
            }
        )
    return records


def _point_to_box_distance(point: tuple[float, float], bbox: list[float]) -> float:
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return hypot(max(x_min - x, 0.0, x - x_max), max(y_min - y, 0.0, y - y_max))


def _unassigned_record(
    split: str,
    image_file: str,
    label_file: str,
    ppe: dict[str, Any],
    people: list[dict[str, Any]],
) -> dict[str, Any]:
    nearest_person = None
    nearest_edge_distance = None
    nearest_center_distance = None
    ppe_center = (ppe["center_x"], ppe["center_y"])

    if people:
        nearest_edge_distance, nearest_center_distance, _, nearest_person = min(
            (
                _point_to_box_distance(ppe_center, person["person_bbox"]),
                normalized_center_distance(ppe_center, person["person_bbox"]),
                person["person_id"],
                person,
            )
            for person in people
        )

    if not people:
        category = "no_person_annotation"
    elif nearest_edge_distance <= NEAR_PERSON_BOUNDARY_DISTANCE_NORM:
        category = "near_person_boundary"
    elif nearest_edge_distance >= CLEARLY_FAR_FROM_PERSON_DISTANCE_NORM:
        category = "clearly_far_from_person_box"
    else:
        category = "outside_person_box"

    return {
        "split": split,
        "image_file": image_file,
        "label_file": label_file,
        "ppe_row_id": ppe["row_id"],
        "ppe_row_number": ppe["row_number"],
        "ppe_class": ppe["class"],
        "person_count_in_image": len(people),
        "has_person_annotation": bool(people),
        "nearest_person_id": nearest_person["person_id"] if nearest_person else None,
        "nearest_person_center_distance_norm": nearest_center_distance,
        "nearest_person_edge_distance_norm": nearest_edge_distance,
        "ppe_center_x": ppe["center_x"],
        "ppe_center_y": ppe["center_y"],
        "relationship_category": category,
        "review_level": "REVIEW",
    }


def _unassigned_summary(unassigned: pd.DataFrame) -> pd.DataFrame:
    if unassigned.empty:
        return pd.DataFrame(columns=["split", "ppe_class", "relationship_category", "unassigned_ppe_count", "affected_images"])

    return (
        unassigned.groupby(["split", "ppe_class", "relationship_category"], dropna=False)
        .agg(unassigned_ppe_count=("ppe_row_id", "size"), affected_images=("image_file", "nunique"))
        .reset_index()
        .sort_values(["split", "ppe_class", "relationship_category"])
    )


def _multiple_summary(multiple_same_class: pd.DataFrame) -> pd.DataFrame:
    if multiple_same_class.empty:
        return pd.DataFrame(columns=["split", "ppe_class", "associated_box_count", "person_count", "affected_images"])

    return (
        multiple_same_class.groupby(["split", "ppe_class", "associated_box_count"], dropna=False)
        .agg(person_count=("person_row_id", "size"), affected_images=("image_file", "nunique"))
        .reset_index()
        .sort_values(["split", "ppe_class", "associated_box_count"])
    )


def _plot_assignment_counts(assignments: pd.DataFrame, unassigned: pd.DataFrame) -> None:
    classes = list(PPE_CLASS_NAMES)
    assigned_counts = assignments["ppe_class"].value_counts().to_dict() if not assignments.empty else {}
    unassigned_counts = unassigned["ppe_class"].value_counts().to_dict() if not unassigned.empty else {}
    x_positions = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x_positions - 0.18, [assigned_counts.get(name, 0) for name in classes], width=0.36, label="Assigned")
    ax.bar(x_positions + 0.18, [unassigned_counts.get(name, 0) for name in classes], width=0.36, label="Unassigned")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(classes)
    ax.set_ylabel("PPE annotation count")
    ax.set_title("PPE Assignment to Person Boxes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "person_ppe_assignment_counts.png", dpi=150)
    plt.close(fig)


def _plot_unassigned_reasons(unassigned: pd.DataFrame) -> None:
    if unassigned.empty:
        return

    counts = unassigned["relationship_category"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(counts.index, counts.values, color="#F28E2B")
    ax.set_ylabel("Unassigned PPE count")
    ax.set_title("Unassigned PPE Review Categories")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "unassigned_ppe_by_reason.png", dpi=150)
    plt.close(fig)


def _plot_multiple_same_class(multiple_same_class: pd.DataFrame) -> None:
    if multiple_same_class.empty:
        return

    counts = multiple_same_class["ppe_class"].value_counts().reindex(PPE_CLASS_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(counts.index, counts.values, color="#4E79A7")
    ax.set_ylabel("Person count")
    ax.set_title("People with Multiple Same-Class PPE Boxes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multiple_same_class_ppe_by_class.png", dpi=150)
    plt.close(fig)
