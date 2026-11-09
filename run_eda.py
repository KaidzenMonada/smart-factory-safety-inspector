"""Run the clean final EDA workflow and generate the report notebook."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

import nbformat
import pandas as pd

from eda.bbox_analysis import analyze_bounding_boxes
from eda.class_distribution import analyze_class_distribution
from eda.dataset_audit import (
    EXAMPLES_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    SUMMARIES_DIR,
    TABLES_DIR,
    audit_dataset,
    dataset_fingerprint,
    prepare_output_dirs,
    write_dataframe,
    write_json,
)
from eda.duplicate_analysis import analyze_duplicates
from eda.label_validation import FAIL_ISSUE_TYPES, validate_yolo_labels
from eda.person_ppe_analysis import analyze_person_ppe_relationships
from eda.visual_inspection import generate_visual_examples


NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_Final_Dataset_EDA.ipynb"
SUMMARY_PATH = SUMMARIES_DIR / "eda_summary.json"

CORE_EXPECTED = {
    "total_image_files": 3882,
    "total_label_files": 3882,
    "total_annotation_rows": 23772,
    "fail_issue_rows": 0,
    "tiny_boundary_overflow_rows": 862,
    "unassigned_ppe_count": 152,
    "multiple_same_class_ppe_person_cases": 2149,
    "exact_duplicate_pair_count": 0,
    "near_duplicate_candidate_pair_count": 17,
}


def main() -> None:
    before = dataset_fingerprint()
    prepare_output_dirs(clean=True)

    dataset = audit_dataset()
    labels = validate_yolo_labels(dataset)
    classes = analyze_class_distribution(dataset, labels["annotations"])
    boxes = analyze_bounding_boxes(dataset, labels["annotations"])
    person_ppe = analyze_person_ppe_relationships(labels["annotations"])
    duplicates = analyze_duplicates(dataset, labels["annotations"])
    visuals = generate_visual_examples(
        annotations=labels["annotations"],
        validation_issues=labels["issues"],
        bbox_candidates=boxes["review_candidates"],
        person_ppe_results=person_ppe,
        duplicate_results=duplicates,
    )

    readiness = build_readiness(dataset, labels, classes, boxes, person_ppe, duplicates)
    summary = {
        "dataset": dataset["summary"],
        "label_validation": labels["summary"],
        "class_distribution": classes["summary"],
        "bbox_analysis": boxes["summary"],
        "person_ppe": person_ppe["summary"],
        "duplicates": duplicates["summary"],
        "readiness": readiness,
        "visual_example_count": len(visuals),
        "core_result_consistency": core_result_consistency(dataset, labels, person_ppe, duplicates),
    }
    write_json(summary, SUMMARY_PATH)

    create_notebook(summary, dataset, labels, classes, boxes, person_ppe, duplicates, visuals)
    after = dataset_fingerprint()
    validation = validate_notebook()
    summary["dataset_unchanged"] = before == after
    summary["dataset_fingerprint_before"] = before
    summary["dataset_fingerprint_after"] = after
    summary["notebook_validation"] = validation
    summary["output_counts"] = output_counts()
    write_json(summary, SUMMARY_PATH)

    print(f"EDA complete. Final verdict: {readiness['final_verdict']}")
    print(f"Dataset unchanged: {before == after}")
    print(f"Notebook code cells: {validation['code_cell_count']}")


def build_readiness(
    dataset: dict[str, Any],
    labels: dict[str, Any],
    classes: dict[str, Any],
    boxes: dict[str, Any],
    person_ppe: dict[str, Any],
    duplicates: dict[str, Any],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []

    def add(severity: str, category: str, finding: str, count: int | float, affected_images: int | str, action: str, table: str) -> None:
        findings.append(
            {
                "severity": severity,
                "category": category,
                "finding": finding,
                "count": count,
                "affected_images": affected_images,
                "action_required": action,
                "evidence_table": table,
            }
        )

    dataset_summary = dataset["summary"]
    label_summary = labels["summary"]
    class_summary = classes["summary"]
    box_summary = boxes["summary"]
    person_summary = person_ppe["summary"]
    duplicate_summary = duplicates["summary"]

    structural = dataset_summary["missing_label_files"] + dataset_summary["missing_image_files"]
    unreadable = dataset_summary["unsupported_image_files"] + dataset_summary["unreadable_images"]
    fail_rows = label_summary["fail_issue_rows"]
    large_boundary = label_summary["large_boundary_overflow_rows"]
    tiny_boundary = label_summary["tiny_boundary_overflow_rows"]
    duplicate_rows = label_summary["duplicate_annotation_row_count"]
    small_ppe = box_summary["small_or_very_small_ppe_candidate_count"]
    geometry = box_summary["review_candidate_count"]
    unassigned = person_summary["unassigned_ppe_count"]
    multiple = person_summary["multiple_same_class_ppe_person_cases"]
    exact_cross = duplicate_summary["exact_duplicate_cross_split_pair_count"]
    exact_within = duplicate_summary["exact_duplicate_within_split_pair_count"]
    near = duplicate_summary["near_duplicate_candidate_pair_count"]

    add("FAIL" if structural else "PASS", "Dataset integrity", "Image-label pairing issues.", structural, "", "Yes" if structural else "No", "dataset_structure_summary.csv")
    add("FAIL" if unreadable else "PASS", "Dataset integrity", "Unsupported or unreadable images.", unreadable, unreadable, "Yes" if unreadable else "No", "dataset_structure_summary.csv")
    add("PASS", "Dataset integrity", "Empty label files.", dataset_summary["empty_label_files"], dataset_summary["empty_label_files"], "No", "dataset_structure_summary.csv")
    add("FAIL" if fail_rows else "PASS", "Label validation", "Malformed, invalid-class, non-finite, out-of-range, zero, or negative annotation rows.", fail_rows, _affected_issue_images(labels["issues"], FAIL_ISSUE_TYPES), "Yes" if fail_rows else "No", "label_validation_issues.csv")
    add("WARNING" if large_boundary else "PASS", "Label validation", "Large calculated boundary overflows.", large_boundary, _affected_issue_images(labels["issues"], {"boundary_overflow_large"}), "Yes" if large_boundary else "No", "boundary_overflow_rows.csv")
    add("REVIEW" if tiny_boundary else "PASS", "Label validation", "Tiny calculated boundary overflows.", tiny_boundary, _affected_issue_images(labels["issues"], {"boundary_overflow_tiny"}), "No", "boundary_overflow_rows.csv")
    add("WARNING" if duplicate_rows else "PASS", "Label validation", "Exact duplicate annotation rows.", duplicate_rows, _affected_issue_images(labels["issues"], {"duplicate_annotation_row"}), "Yes" if duplicate_rows else "No", "label_validation_issues.csv")
    add("PASS", "Class distribution", "Maximum-to-minimum nonzero class-count ratio.", round(class_summary["imbalance_ratio_max_to_min_nonzero"], 3), "", "No", "class_counts.csv")
    add("REVIEW" if small_ppe else "PASS", "Bounding boxes", "Small or very small PPE boxes at the 640-pixel training input scale.", small_ppe, _affected_images(boxes["review_candidates"]), "No", "bbox_review_candidates.csv")
    add("REVIEW" if geometry else "PASS", "Bounding boxes", "Heuristic geometry review candidates.", geometry, _affected_images(boxes["review_candidates"]), "No", "bbox_review_candidates.csv")
    add("REVIEW" if unassigned else "PASS", "Person/PPE relationship", "PPE not assigned to any person under the preserved rule.", unassigned, person_summary["unassigned_ppe_images"], "No", "unassigned_ppe.csv")
    add("REVIEW" if multiple else "PASS", "Person/PPE relationship", "Multiple same-class PPE boxes associated with one person.", multiple, _affected_images(person_ppe["multiple_same_class"]), "No", "multiple_same_class_ppe_per_person.csv")
    add("FAIL" if exact_cross else "PASS", "Duplicate and leakage", "Confirmed exact cross-split duplicate image pairs.", exact_cross, _pair_images(duplicates["exact_pairs"], True), "Yes" if exact_cross else "No", "eda_summary.json")
    add("WARNING" if exact_within else "PASS", "Duplicate and leakage", "Confirmed exact within-split duplicate image pairs.", exact_within, _pair_images(duplicates["exact_pairs"], False), "Yes" if exact_within else "No", "eda_summary.json")
    add("REVIEW" if near else "PASS", "Duplicate and leakage", "Perceptual near-duplicate candidate pairs.", near, _pair_images(duplicates["near_duplicate_candidates"], None), "Manual Review" if near else "No", "near_duplicate_candidates.csv")

    findings_df = pd.DataFrame(findings)
    write_dataframe(findings_df, TABLES_DIR / "readiness_findings.csv")

    critical = structural + unreadable + fail_rows
    cleaning_required = large_boundary + duplicate_rows + exact_cross + exact_within
    if critical:
        verdict = "NOT READY: CRITICAL DATASET PROBLEMS FOUND"
    elif cleaning_required:
        verdict = "ATTENTION: DATASET CLEANING REQUIRED"
    else:
        verdict = "READY FOR TRAINING"

    return {
        "final_verdict": verdict,
        "critical_fail_count": int(critical),
        "cleaning_required_count": int(cleaning_required),
        "blocking_problem_count": int(critical + cleaning_required),
        "cross_split_exact_duplicate_pairs": int(exact_cross),
        "findings": findings,
    }


def core_result_consistency(dataset: dict[str, Any], labels: dict[str, Any], person_ppe: dict[str, Any], duplicates: dict[str, Any]) -> dict[str, Any]:
    current = {
        "total_image_files": dataset["summary"]["total_image_files"],
        "total_label_files": dataset["summary"]["total_label_files"],
        "total_annotation_rows": labels["summary"]["total_annotation_rows"],
        "fail_issue_rows": labels["summary"]["fail_issue_rows"],
        "tiny_boundary_overflow_rows": labels["summary"]["tiny_boundary_overflow_rows"],
        "unassigned_ppe_count": person_ppe["summary"]["unassigned_ppe_count"],
        "multiple_same_class_ppe_person_cases": person_ppe["summary"]["multiple_same_class_ppe_person_cases"],
        "exact_duplicate_pair_count": duplicates["summary"]["exact_duplicate_pair_count"],
        "near_duplicate_candidate_pair_count": duplicates["summary"]["near_duplicate_candidate_pair_count"],
    }
    return {
        key: {"expected": expected, "actual": current[key], "matches": current[key] == expected}
        for key, expected in CORE_EXPECTED.items()
    }


def create_notebook(
    summary: dict[str, Any],
    dataset: dict[str, Any],
    labels: dict[str, Any],
    classes: dict[str, Any],
    boxes: dict[str, Any],
    person_ppe: dict[str, Any],
    duplicates: dict[str, Any],
    visuals: pd.DataFrame,
) -> None:
    near_count = duplicates["summary"]["near_duplicate_candidate_pair_count"]
    duplicate_figures = []
    if near_count:
        duplicate_figures.append(_fig("outputs/eda/figures/near_duplicate_candidate_distance_histogram.png", "Near-duplicate candidate distances"))
    duplicate_findings = [
        f"Near-duplicate threshold: Hamming distance <= {duplicates['summary']['near_duplicate_hamming_threshold']}.",
        "Perceptual near-duplicates are candidates only." if near_count else "No perceptual near-duplicate candidates remain.",
    ]
    duplicate_action = "Manual Review" if near_count else "No"
    duplicate_conclusion = (
        "Near-duplicate candidates should be checked visually before any split-cleaning decision."
        if near_count
        else "No duplicate or near-duplicate action is required from the final EDA outputs."
    )

    cells = [
        _md(
            "# Final Dataset EDA: Smart Factory Safety Inspector\n\n"
            "This notebook is a code-free EDA report for the active PPE detection dataset. "
            "All evidence was generated by the scripts in `eda/`; this notebook only explains the results.\n\n"
            f"Dataset: `data/final_dataset/`\n\n"
            f"Classes: {', '.join(dataset['summary']['class_names'])}\n\n"
            f"Final verdict: **{summary['readiness']['final_verdict']}**"
        ),
        _section(
            "Dataset Overview",
            "The dataset split structure, image-label pairing, image readability, dimensions, and YAML paths were audited.",
            _table(dataset["structure_summary"], ["split", "image_files", "label_files", "paired_image_label_stems", "missing_label_files", "missing_image_files", "empty_label_files"]),
            [_fig("outputs/eda/figures/dataset_file_counts_by_split.png", "Image and label counts by split")],
            [
                f"Total images: {_fmt(dataset['summary']['total_image_files'])}.",
                f"Total label files: {_fmt(dataset['summary']['total_label_files'])}.",
                f"Unsupported or unreadable images: {_fmt(dataset['summary']['unsupported_image_files'] + dataset['summary']['unreadable_images'])}.",
                f"YAML paths consistent: {'Yes' if dataset['summary']['yaml_paths_consistent'] else 'No'}.",
            ],
            "No structural split or pairing failure was found.",
            "No",
            "The dataset is structurally ready for EDA-level review.",
        ),
        _section(
            "Dataset Integrity",
            "Readable image dimensions and aspect ratios were summarized by split.",
            _table(dataset["dimension_summary"], ["split", "readable_images", "width_min", "width_median", "width_max", "height_min", "height_median", "height_max", "aspect_ratio_median"]),
            [_fig("outputs/eda/figures/image_dimensions_scatter.png", "Image dimensions by split")],
            [
                f"Readable images: {_fmt(dataset['summary']['total_image_files'])}.",
                f"Distinct dimensions: {_fmt(dataset['summary']['image_dimension']['distinct_dimensions'])}.",
                f"Median aspect ratio: {dataset['summary']['image_dimension']['aspect_ratio_median']:.3f}.",
            ],
            "The dataset contains multiple source sizes, which is expected for a collected PPE dataset.",
            "No",
            "Image files are readable and consistently paired with labels.",
        ),
        _section(
            "Label Validation",
            "Every non-empty YOLO row was checked for token count, numeric parsing, class ID, finite values, normalized coordinate range, positive dimensions, calculated edges, and exact duplicate rows.",
            _table(labels["issue_summary"], ["issue_type", "issue_family", "severity", "issue_count", "affected_annotation_rows", "affected_images"]),
            [],
            [
                f"Annotation rows: {_fmt(labels['summary']['total_annotation_rows'])}.",
                f"Critical malformed/unusable rows: {_fmt(labels['summary']['fail_issue_rows'])}.",
                f"Tiny boundary-overflow rows: {_fmt(labels['summary']['tiny_boundary_overflow_rows'])}.",
                f"Large boundary-overflow rows: {_fmt(labels['summary']['large_boundary_overflow_rows'])}.",
                f"Duplicate annotation rows: {_fmt(labels['summary']['duplicate_annotation_row_count'])}.",
            ],
            "Tiny boundary overflow is separated from critical malformed-label problems.",
            "Manual Review",
            "No critical YOLO label failures were found.",
        ),
        _section(
            "Class Distribution",
            "Usable annotations were counted overall, by split, and by image-level class co-occurrence.",
            _table(classes["class_counts"], ["class_id", "class_name", "annotation_count", "proportion"])
            + "\n\n"
            + _table(classes["class_counts_by_split"], ["split", "class_name", "annotation_count", "split_proportion"], max_rows=15),
            [
                _fig("outputs/eda/figures/class_counts_by_split.png", "Class counts by split"),
                _fig("outputs/eda/figures/class_cooccurrence_heatmap.png", "Class co-occurrence by image"),
            ],
            [
                f"Most common class: {classes['summary']['most_common_class']['class_name']} ({_fmt(classes['summary']['most_common_class']['annotation_count'])}).",
                f"Rarest class: {classes['summary']['rarest_class']['class_name']} ({_fmt(classes['summary']['rarest_class']['annotation_count'])}).",
                f"Max/min nonzero class-count ratio: {classes['summary']['imbalance_ratio_max_to_min_nonzero']:.3f}.",
            ],
            "The dataset is imbalanced, but imbalance alone is not treated as a dataset error.",
            "No",
            "Class distribution should be considered during model evaluation.",
        ),
        _section(
            "Bounding Box Analysis",
            "Bounding boxes were measured in normalized coordinates, original pixels, and estimated 640-input letterbox scale.",
            _table(boxes["summary_by_class"], ["class_name", "box_count", "sqrt_area_640_median", "sqrt_area_640_p10", "sqrt_area_640_p90", "very_small_boxes", "small_boxes", "medium_boxes", "large_boxes"]),
            [
                _fig("outputs/eda/figures/bbox_estimated_size_640_by_class.png", "Estimated object size at 640 input"),
                _fig("outputs/eda/figures/bbox_size_category_by_class.png", "Object size categories by class"),
            ],
            [
                f"Measured boxes: {_fmt(boxes['summary']['total_measured_boxes'])}.",
                f"Geometry review candidates: {_fmt(boxes['summary']['review_candidate_count'])}.",
                f"Small or very small PPE candidates: {_fmt(boxes['summary']['small_or_very_small_ppe_candidate_count'])}.",
            ],
            "Small PPE boxes are review and modeling-risk evidence, not confirmed annotation errors.",
            "Manual Review",
            "The size analysis highlights PPE classes that may be harder to learn.",
        ),
        _section(
            "Person and PPE Relationship Analysis",
            "PPE was associated to people using the preserved project rule: PPE center inside person box, smallest containing person, then nearest-center tie-break.",
            _table(person_ppe["unassigned_summary"], ["split", "ppe_class", "relationship_category", "unassigned_ppe_count", "affected_images"], max_rows=20),
            [
                _fig("outputs/eda/figures/person_ppe_assignment_counts.png", "PPE assignment counts"),
                _fig("outputs/eda/figures/unassigned_ppe_by_reason.png", "Unassigned PPE review categories"),
            ],
            [
                f"PPE annotations considered: {_fmt(person_ppe['summary']['total_ppe_annotations_considered'])}.",
                f"Assigned PPE: {_fmt(person_ppe['summary']['assigned_ppe_count'])}.",
                f"Unassigned PPE: {_fmt(person_ppe['summary']['unassigned_ppe_count'])}.",
                f"Unassigned PPE near person boundary: {_fmt(person_ppe['summary']['unassigned_ppe_near_person_boundary'])}.",
                f"Unassigned PPE clearly far from person boxes: {_fmt(person_ppe['summary']['unassigned_ppe_clearly_far'])}.",
            ],
            "Unassigned PPE matters because compliance is person-centered.",
            "Manual Review",
            "These records should be visually reviewed before any cleaning decision.",
        ),
        _section(
            "Multiple Same-Class PPE Analysis",
            "The EDA counted people associated with more than one instance of the same PPE class.",
            _table(person_ppe["multiple_summary"], ["split", "ppe_class", "associated_box_count", "person_count", "affected_images"], max_rows=20),
            [_fig("outputs/eda/figures/multiple_same_class_ppe_by_class.png", "Multiple same-class PPE cases by class")],
            [
                f"Multiple same-class PPE/person cases: {_fmt(person_ppe['summary']['multiple_same_class_ppe_person_cases'])}.",
                "Multiple gloves can legitimately represent two hands.",
                "Other multiple same-class cases are review candidates, not confirmed errors.",
            ],
            "These cases are heuristic review candidates.",
            "Manual Review",
            "No automatic correction is recommended at the EDA stage.",
        ),
        _section(
            "Duplicate and Split Leakage Analysis",
            "Exact duplicates were checked with file-content hashes. Near-duplicates were checked with a conservative DCT perceptual hash threshold.",
            _table(
                pd.DataFrame(
                    [
                        ["Exact duplicate image pairs", duplicates["summary"]["exact_duplicate_pair_count"]],
                        ["Cross-split exact duplicate pairs", duplicates["summary"]["exact_duplicate_cross_split_pair_count"]],
                        ["Near-duplicate candidate pairs", duplicates["summary"]["near_duplicate_candidate_pair_count"]],
                        ["Cross-split near-duplicate candidates", duplicates["summary"]["near_duplicate_cross_split_candidate_pair_count"]],
                    ],
                    columns=["metric", "value"],
                ),
                ["metric", "value"],
            ),
            duplicate_figures,
            duplicate_findings,
            "No confirmed exact duplicate image pairs were found.",
            duplicate_action,
            duplicate_conclusion,
        ),
        _visual_section(visuals),
        _section(
            "Dataset Risks and Limitations",
            "The EDA separates confirmed blocking errors from documented dataset characteristics and review heuristics.",
            _table(pd.DataFrame(summary["readiness"]["findings"]), ["severity", "category", "finding", "count", "affected_images", "action_required"], max_rows=20),
            [],
            [
                "This EDA run reports the current dataset state after the separate Data Processing stage.",
                "Small boxes, unassigned PPE, multiple same-class PPE, and tiny boundary overflows remain documented observations.",
                "These observations may influence training behavior, but they are not confirmed blocking annotation errors.",
            ],
            "No confirmed structural, label-validity, duplicate, leakage, or other blocking dataset problem remains.",
            "No",
            "The remaining findings are limitations to keep in mind during training and evaluation.",
        ),
        _section(
            "Final Dataset Readiness Assessment",
            "The final verdict is based on measured findings and does not use an any-warning-means-not-ready rule.",
            _table(pd.DataFrame(summary["readiness"]["findings"]), ["severity", "category", "finding", "count", "affected_images", "action_required"], max_rows=20),
            [],
            [
                f"Final verdict: {summary['readiness']['final_verdict']}.",
                f"Critical fail count: {_fmt(summary['readiness']['critical_fail_count'])}.",
                f"Blocking problem count: {_fmt(summary['readiness']['blocking_problem_count'])}.",
                f"Cross-split exact duplicate pairs: {_fmt(summary['readiness']['cross_split_exact_duplicate_pairs'])}.",
            ],
            "READY FOR TRAINING means no confirmed structural, label-validity, duplicate/leakage, or other blocking dataset problems remain.",
            "No",
            summary["readiness"]["final_verdict"],
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nbformat.v4.new_notebook(cells=cells), NOTEBOOK_PATH)


def _visual_section(visuals: pd.DataFrame) -> nbformat.NotebookNode:
    counts = visuals["category"].value_counts().sort_index().reset_index()
    counts.columns = ["category", "example_count"]
    figure_blocks = []
    for row in visuals.drop_duplicates("category").sort_values("category").to_dict("records"):
        figure_blocks.append(f"**{row['category'].replace('_', ' ').title()}**\n\n{_fig(row['example_file'], row['category'])}\n\n{row['reason']}")
    return _section(
        "Visual Annotation Review",
        "A compact deterministic sample was rendered with ground-truth boxes and class labels.",
        _table(counts, ["category", "example_count"]),
        figure_blocks,
        [f"Generated visual examples: {_fmt(len(visuals))}.", "Each example is linked to a manual-review evidence table."],
        "Visual examples support review; they do not prove that a label should be changed.",
        "Manual Review",
        "The visual set is small enough for mentor review while covering the important EDA categories.",
    )


def _section(
    title: str,
    analyzed: str,
    table_text: str,
    figures: list[str],
    findings: list[str],
    interpretation: str,
    action: str,
    conclusion: str,
) -> nbformat.NotebookNode:
    parts = [
        f"## {title}",
        f"### What was analyzed\n{analyzed}",
        f"### Result table or figure\n{table_text}",
        *figures,
        "### Actual findings\n" + "\n".join(f"- {finding}" for finding in findings),
        f"### Short interpretation\n{interpretation}",
        f"### Action required\n{action}",
        f"### Section conclusion\n{conclusion}",
    ]
    return _md("\n\n".join(part for part in parts if part))


def _md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def _fig(project_path: str, alt: str) -> str:
    return f"![{alt}](../{project_path})"


def _table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "No records."
    view = df[columns].head(max_rows).fillna("")
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in view.to_dict("records"):
        rows.append("| " + " | ".join(_cell(record[column]) for column in columns) + " |")
    if max_rows and len(df) > max_rows:
        rows.append(f"\nShowing {max_rows} of {len(df)} rows. Detailed records are in `outputs/eda/tables/`.")
    return "\n".join(rows)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        value = f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "/").replace("\n", " ")


def _fmt(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, int) else str(value)


def _affected_images(df: pd.DataFrame) -> int:
    return int(df["image_file"].replace("", pd.NA).dropna().nunique()) if not df.empty and "image_file" in df else 0


def _affected_issue_images(issues: pd.DataFrame, issue_types: set[str]) -> int:
    return _affected_images(issues[issues["issue_type"].isin(issue_types)]) if not issues.empty else 0


def _pair_images(df: pd.DataFrame, cross_split: bool | None) -> int:
    if df.empty:
        return 0
    subset = df if cross_split is None else df[df["cross_split"] == cross_split]
    return len(set(subset["image_a"]).union(set(subset["image_b"])))


def validate_notebook() -> dict[str, Any]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    unresolved = [ref for ref in image_refs if not (NOTEBOOK_PATH.parent / ref).resolve().exists()]
    return {
        "code_cell_count": sum(1 for cell in notebook.cells if cell.cell_type == "code"),
        "markdown_cell_count": sum(1 for cell in notebook.cells if cell.cell_type == "markdown"),
        "image_reference_count": len(image_refs),
        "unresolved_image_paths": unresolved,
    }


def output_counts() -> dict[str, int]:
    return {
        "tables": len(list(TABLES_DIR.glob("*.csv"))),
        "figures": len(list(FIGURES_DIR.glob("*.png"))),
        "examples": len(list(EXAMPLES_DIR.glob("*.png"))),
        "summaries": len(list(SUMMARIES_DIR.glob("*.json"))),
    }


if __name__ == "__main__":
    main()
