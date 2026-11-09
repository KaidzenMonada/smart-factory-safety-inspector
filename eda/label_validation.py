"""YOLO label validation for the final PPE dataset."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from eda.dataset_audit import CLASS_ID_TO_NAME, TABLES_DIR, write_dataframe


BOUNDARY_OVERFLOW_TINY_LIMIT = 0.001

FAIL_ISSUE_TYPES = {
    "token_count_invalid",
    "numeric_parse_error",
    "invalid_class_id",
    "non_finite_class_id",
    "non_finite_x_center",
    "non_finite_y_center",
    "non_finite_width",
    "non_finite_height",
    "x_center_out_of_range",
    "y_center_out_of_range",
    "width_out_of_range",
    "height_out_of_range",
    "zero_width",
    "zero_height",
    "negative_width",
    "negative_height",
}

ISSUE_META = {
    "token_count_invalid": ("malformed_annotation", "FAIL"),
    "numeric_parse_error": ("malformed_annotation", "FAIL"),
    "invalid_class_id": ("invalid_class", "FAIL"),
    "non_finite_class_id": ("non_finite_value", "FAIL"),
    "non_finite_x_center": ("non_finite_value", "FAIL"),
    "non_finite_y_center": ("non_finite_value", "FAIL"),
    "non_finite_width": ("non_finite_value", "FAIL"),
    "non_finite_height": ("non_finite_value", "FAIL"),
    "x_center_out_of_range": ("normalized_value_out_of_range", "FAIL"),
    "y_center_out_of_range": ("normalized_value_out_of_range", "FAIL"),
    "width_out_of_range": ("normalized_value_out_of_range", "FAIL"),
    "height_out_of_range": ("normalized_value_out_of_range", "FAIL"),
    "zero_width": ("invalid_box_dimension", "FAIL"),
    "zero_height": ("invalid_box_dimension", "FAIL"),
    "negative_width": ("invalid_box_dimension", "FAIL"),
    "negative_height": ("invalid_box_dimension", "FAIL"),
    "boundary_overflow_tiny": ("boundary_overflow_review", "REVIEW"),
    "boundary_overflow_large": ("boundary_overflow_review", "WARNING"),
    "duplicate_annotation_row": ("duplicate_annotation", "REVIEW"),
}

ISSUE_COLUMNS = [
    "split",
    "image_file",
    "label_file",
    "row_number",
    "row_id",
    "class_id",
    "class_name",
    "raw_annotation_values",
    "issue_type",
    "issue_family",
    "severity",
    "field",
    "value",
    "token_count",
    "first_duplicate_row",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "boundary_overflow_magnitude",
    "details",
]


def validate_yolo_labels(dataset_state: dict[str, Any]) -> dict[str, Any]:
    """Validate every non-empty YOLO row and keep parsed annotations in memory."""

    image_lookup = {
        (row["split"], row["stem"]): row["image_file"]
        for row in dataset_state["image_files"].to_dict("records")
    }
    annotation_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for label in dataset_state["label_files"].to_dict("records"):
        seen_rows: dict[str, int] = {}
        image_file = image_lookup.get((label["split"], label["stem"]), "")
        for row_number, raw_line in enumerate(label["_path"].read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            raw = raw_line.strip()
            if not raw:
                continue

            row = _base_annotation(label, image_file, row_number, raw)
            row_issues: list[str] = []
            tokens = raw.split()

            if raw in seen_rows:
                _add_issue(issue_rows, row, "duplicate_annotation_row", row_issues, "annotation_row", raw, len(tokens), seen_rows[raw])
            else:
                seen_rows[raw] = row_number

            values = _parse_tokens(tokens, row, issue_rows, row_issues)
            if values is not None:
                _validate_values(row, values, issue_rows, row_issues)

            row["is_duplicate_annotation_row"] = "duplicate_annotation_row" in row_issues
            row["has_label_issue"] = bool(row_issues)
            row["is_usable_annotation"] = _is_usable(row, row_issues)
            annotation_rows.append(row)

    annotations = pd.DataFrame(annotation_rows)
    issues = pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)
    issue_summary = _issue_summary(issues)
    boundary_rows = issues[issues["issue_type"].isin(["boundary_overflow_tiny", "boundary_overflow_large"])].copy()

    write_dataframe(issues, TABLES_DIR / "label_validation_issues.csv")
    write_dataframe(issue_summary, TABLES_DIR / "label_validation_issue_summary.csv")
    write_dataframe(boundary_rows, TABLES_DIR / "boundary_overflow_rows.csv")

    summary = {
        "total_annotation_rows": len(annotations),
        "usable_annotation_rows": int(annotations["is_usable_annotation"].sum()) if not annotations.empty else 0,
        "fail_issue_rows": int(issues[issues["issue_type"].isin(FAIL_ISSUE_TYPES)]["row_id"].nunique()) if not issues.empty else 0,
        "fail_issue_count": int(issues["issue_type"].isin(FAIL_ISSUE_TYPES).sum()) if not issues.empty else 0,
        "tiny_boundary_overflow_rows": int((annotations["is_tiny_boundary_overflow"]).sum()) if not annotations.empty else 0,
        "large_boundary_overflow_rows": int((annotations["is_large_boundary_overflow"]).sum()) if not annotations.empty else 0,
        "duplicate_annotation_row_count": int((annotations["is_duplicate_annotation_row"]).sum()) if not annotations.empty else 0,
        "boundary_overflow_tiny_limit": BOUNDARY_OVERFLOW_TINY_LIMIT,
    }

    return {
        "annotations": annotations,
        "issues": issues,
        "issue_summary": issue_summary,
        "summary": summary,
    }


def _base_annotation(label: dict[str, Any], image_file: str, row_number: int, raw: str) -> dict[str, Any]:
    return {
        "split": label["split"],
        "image_file": image_file,
        "label_file": label["label_file"],
        "row_number": row_number,
        "row_id": f"{label['split']}:{label['label_name']}:{row_number}",
        "class_id": None,
        "class_name": "",
        "raw_annotation_values": raw,
        "x_center": None,
        "y_center": None,
        "box_width_norm": None,
        "box_height_norm": None,
        "x_min": None,
        "y_min": None,
        "x_max": None,
        "y_max": None,
        "boundary_overflow_magnitude": None,
        "is_tiny_boundary_overflow": False,
        "is_large_boundary_overflow": False,
    }


def _parse_tokens(
    tokens: list[str],
    row: dict[str, Any],
    issues: list[dict[str, Any]],
    row_issues: list[str],
) -> list[float] | None:
    if len(tokens) != 5:
        _add_issue(issues, row, "token_count_invalid", row_issues, token_count=len(tokens))
        return None

    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        _add_issue(issues, row, "numeric_parse_error", row_issues, token_count=len(tokens), details=str(exc))
        return None


def _validate_values(
    row: dict[str, Any],
    values: list[float],
    issues: list[dict[str, Any]],
    row_issues: list[str],
) -> None:
    class_value, x_center, y_center, width, height = values
    fields = {
        "class_id": class_value,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }

    for field, value in fields.items():
        if not math.isfinite(value):
            _add_issue(issues, row, f"non_finite_{field}", row_issues, field=field, value=value, token_count=5)

    if any(issue.startswith("non_finite_") for issue in row_issues):
        return

    class_id = int(class_value) if class_value.is_integer() else None
    if class_id not in CLASS_ID_TO_NAME:
        _add_issue(issues, row, "invalid_class_id", row_issues, "class_id", class_value, 5)
    else:
        row["class_id"] = class_id
        row["class_name"] = CLASS_ID_TO_NAME[class_id]

    row.update(
        {
            "x_center": x_center,
            "y_center": y_center,
            "box_width_norm": width,
            "box_height_norm": height,
        }
    )

    for field, value in (("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)):
        if value < 0.0 or value > 1.0:
            _add_issue(issues, row, f"{field}_out_of_range", row_issues, field, value, 5)

    for field, value in (("width", width), ("height", height)):
        if value == 0.0:
            _add_issue(issues, row, f"zero_{field}", row_issues, field, value, 5)
        elif value < 0.0:
            _add_issue(issues, row, f"negative_{field}", row_issues, field, value, 5)

    if width <= 0.0 or height <= 0.0:
        return

    row["x_min"] = x_center - width / 2.0
    row["y_min"] = y_center - height / 2.0
    row["x_max"] = x_center + width / 2.0
    row["y_max"] = y_center + height / 2.0
    overflow = max(0.0, -row["x_min"], -row["y_min"], row["x_max"] - 1.0, row["y_max"] - 1.0)
    row["boundary_overflow_magnitude"] = overflow

    if overflow > 0.0:
        issue_type = "boundary_overflow_tiny" if overflow <= BOUNDARY_OVERFLOW_TINY_LIMIT else "boundary_overflow_large"
        row["is_tiny_boundary_overflow"] = issue_type == "boundary_overflow_tiny"
        row["is_large_boundary_overflow"] = issue_type == "boundary_overflow_large"
        _add_issue(
            issues,
            row,
            issue_type,
            row_issues,
            token_count=5,
            x_min=row["x_min"],
            y_min=row["y_min"],
            x_max=row["x_max"],
            y_max=row["y_max"],
            boundary_overflow_magnitude=overflow,
        )


def _add_issue(
    issues: list[dict[str, Any]],
    row: dict[str, Any],
    issue_type: str,
    row_issues: list[str],
    field: str = "",
    value: Any = "",
    token_count: int | None = None,
    first_duplicate_row: int | None = None,
    x_min: float | None = None,
    y_min: float | None = None,
    x_max: float | None = None,
    y_max: float | None = None,
    boundary_overflow_magnitude: float | None = None,
    details: str = "",
) -> None:
    family, severity = ISSUE_META[issue_type]
    issues.append(
        {
            "split": row["split"],
            "image_file": row["image_file"],
            "label_file": row["label_file"],
            "row_number": row["row_number"],
            "row_id": row["row_id"],
            "class_id": row.get("class_id"),
            "class_name": row.get("class_name", ""),
            "raw_annotation_values": row["raw_annotation_values"],
            "issue_type": issue_type,
            "issue_family": family,
            "severity": severity,
            "field": field,
            "value": value,
            "token_count": token_count,
            "first_duplicate_row": first_duplicate_row,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "boundary_overflow_magnitude": boundary_overflow_magnitude,
            "details": details,
        }
    )
    row_issues.append(issue_type)


def _is_usable(row: dict[str, Any], row_issues: list[str]) -> bool:
    required = ("class_id", "x_center", "y_center", "box_width_norm", "box_height_norm")
    return (
        all(row.get(field) is not None for field in required)
        and row["box_width_norm"] > 0.0
        and row["box_height_norm"] > 0.0
        and all(0.0 <= row[field] <= 1.0 for field in ("x_center", "y_center", "box_width_norm", "box_height_norm"))
        and not any(issue in FAIL_ISSUE_TYPES for issue in row_issues)
    )


def _issue_summary(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return pd.DataFrame(
            columns=[
                "issue_type",
                "issue_family",
                "severity",
                "issue_count",
                "affected_annotation_rows",
                "affected_images",
                "affected_label_files",
            ]
        )

    return (
        issues.groupby(["issue_type", "issue_family", "severity"], dropna=False)
        .agg(
            issue_count=("issue_type", "size"),
            affected_annotation_rows=("row_id", "nunique"),
            affected_images=("image_file", "nunique"),
            affected_label_files=("label_file", "nunique"),
        )
        .reset_index()
        .sort_values(["severity", "issue_type"])
    )
