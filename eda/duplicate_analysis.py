"""Exact and perceptual duplicate analysis."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from eda.dataset_audit import FIGURES_DIR, TABLES_DIR, file_sha256, write_dataframe


PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8
PHASH_HAMMING_THRESHOLD = 4
PAIR_COLUMNS = [
    "image_a",
    "image_b",
    "split_a",
    "split_b",
    "cross_split",
    "perceptual_distance",
    "label_files_identical",
    "annotation_count_a",
    "annotation_count_b",
    "annotation_count_difference",
    "class_counts_a_json",
    "class_counts_b_json",
    "class_count_difference_json",
]


def analyze_duplicates(dataset_state: dict[str, Any], annotations: pd.DataFrame) -> dict[str, Any]:
    """Find confirmed exact duplicates and conservative perceptual candidates."""

    image_records = _image_records(dataset_state, annotations)
    exact_pairs = _exact_duplicate_pairs(image_records)
    perceptual_rows, phash_failures = _perceptual_hashes(image_records)
    near_candidates = _near_duplicate_candidates(perceptual_rows)

    write_dataframe(near_candidates, TABLES_DIR / "near_duplicate_candidates.csv")
    if not near_candidates.empty:
        _plot_near_duplicate_distances(near_candidates)

    summary = {
        "exact_duplicate_pair_count": len(exact_pairs),
        "exact_duplicate_cross_split_pair_count": int(exact_pairs["cross_split"].sum()) if not exact_pairs.empty else 0,
        "exact_duplicate_within_split_pair_count": int((~exact_pairs["cross_split"]).sum()) if not exact_pairs.empty else 0,
        "near_duplicate_method": "DCT perceptual hash: grayscale EXIF-corrected image resized to 32x32, 8x8 low-frequency DCT bits, Hamming distance.",
        "near_duplicate_hamming_threshold": PHASH_HAMMING_THRESHOLD,
        "near_duplicate_candidate_pair_count": len(near_candidates),
        "near_duplicate_cross_split_candidate_pair_count": int(near_candidates["cross_split"].sum()) if not near_candidates.empty else 0,
        "perceptual_hash_failure_count": phash_failures,
    }

    return {
        "exact_pairs": exact_pairs,
        "near_duplicate_candidates": near_candidates,
        "summary": summary,
    }


def _image_records(dataset_state: dict[str, Any], annotations: pd.DataFrame) -> pd.DataFrame:
    usable = annotations[annotations["is_usable_annotation"]]
    annotation_lookup = {}
    for image_file, group in usable.groupby("image_file", sort=True):
        annotation_lookup[image_file] = {
            "label_file": group["label_file"].iloc[0],
            "annotation_count": len(group),
            "class_counts": group["class_name"].value_counts().sort_index().astype(int).to_dict(),
        }
    label_hashes = {
        row["label_file"]: file_sha256(row["_path"])
        for row in dataset_state["label_files"].to_dict("records")
    }

    rows = []
    for image in dataset_state["image_files"].to_dict("records"):
        if not image["is_supported_extension"]:
            continue
        annotation_info = annotation_lookup.get(image["image_file"], {"label_file": "", "annotation_count": 0, "class_counts": {}})
        label_file = annotation_info["label_file"]
        rows.append(
            {
                "split": image["split"],
                "image_file": image["image_file"],
                "sha256": file_sha256(image["_path"]),
                "label_file": label_file,
                "label_sha256": label_hashes.get(label_file, ""),
                "annotation_count": int(annotation_info["annotation_count"]),
                "class_counts_json": json.dumps(annotation_info["class_counts"], sort_keys=True),
                "_path": image["_path"],
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "image_file"])


def _pair_record(a: dict[str, Any], b: dict[str, Any], distance: int) -> dict[str, Any]:
    counts_a = json.loads(a["class_counts_json"]) if a["class_counts_json"] else {}
    counts_b = json.loads(b["class_counts_json"]) if b["class_counts_json"] else {}
    diff = {key: int(counts_a.get(key, 0)) - int(counts_b.get(key, 0)) for key in sorted(set(counts_a) | set(counts_b))}

    return {
        "image_a": a["image_file"],
        "image_b": b["image_file"],
        "split_a": a["split"],
        "split_b": b["split"],
        "cross_split": a["split"] != b["split"],
        "perceptual_distance": distance,
        "label_files_identical": bool(a["label_sha256"] and a["label_sha256"] == b["label_sha256"]),
        "annotation_count_a": int(a["annotation_count"]),
        "annotation_count_b": int(b["annotation_count"]),
        "annotation_count_difference": int(a["annotation_count"]) - int(b["annotation_count"]),
        "class_counts_a_json": a["class_counts_json"],
        "class_counts_b_json": b["class_counts_json"],
        "class_count_difference_json": json.dumps(diff, sort_keys=True),
    }


def _exact_duplicate_pairs(image_records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in image_records.groupby("sha256", sort=True):
        if len(group) > 1:
            rows.extend(_pair_record(a, b, 0) for a, b in combinations(group.to_dict("records"), 2))
    return pd.DataFrame(rows)


def _perceptual_hashes(image_records: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    failures = 0

    for record in image_records.to_dict("records"):
        try:
            phash = _compute_phash(record["_path"])
            rows.append({**record, "phash_int": phash})
        except Exception:  # noqa: BLE001 - record count only; image readability was already audited.
            failures += 1

    return pd.DataFrame(rows).sort_values(["split", "image_file"]), failures


def _compute_phash(path: Path) -> int:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
    low_freq = cv2.dct(np.asarray(image, dtype=np.float32))[:PHASH_LOW_FREQUENCY_SIZE, :PHASH_LOW_FREQUENCY_SIZE]
    median = np.median(low_freq.flatten()[1:])
    value = 0
    for bit in (low_freq > median).flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _near_duplicate_candidates(perceptual_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    records = perceptual_rows.to_dict("records")

    for index, a in enumerate(records):
        for b in records[index + 1 :]:
            if a["sha256"] == b["sha256"]:
                continue
            distance = (int(a["phash_int"]) ^ int(b["phash_int"])).bit_count()
            if distance <= PHASH_HAMMING_THRESHOLD:
                rows.append(_pair_record(a, b, distance))

    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.DataFrame(rows, columns=PAIR_COLUMNS).sort_values(["cross_split", "perceptual_distance", "image_a", "image_b"], ascending=[False, True, True, True])


def _plot_near_duplicate_distances(near_candidates: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(
        near_candidates["perceptual_distance"],
        bins=range(PHASH_HAMMING_THRESHOLD + 2),
        color="#59A14F",
        edgecolor="white",
        align="left",
    )
    ax.set_xticks(range(PHASH_HAMMING_THRESHOLD + 1))
    ax.set_xlabel("DCT pHash Hamming distance")
    ax.set_ylabel("Candidate pair count")
    ax.set_title("Near-Duplicate Candidate Distances")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "near_duplicate_candidate_distance_histogram.png", dpi=150)
    plt.close(fig)
