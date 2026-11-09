"""Person-level safety compliance logic."""

from __future__ import annotations

from math import hypot
from typing import Any

from config.dataset import PERSON_CLASS_NAME, PPE_CLASS_NAMES


COMPLIANT = "COMPLIANT"
ATTENTION = "ATTENTION"
UNKNOWN = "UNKNOWN"
GLOVE_CLASS_NAME = "gloves"
GLOVE_RECOVERY_EXPANSION_RATIO = 0.15
GLOVE_RECOVERY_MIN_OVERLAP_RATIO = 0.25


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    """Return the center point of an xyxy bounding box."""

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox: list[float]) -> float:
    """Return the area of an xyxy bounding box."""

    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection_area(
    first_bbox: list[float],
    second_bbox: list[float],
) -> float:
    """Return the intersection area between two xyxy bounding boxes."""

    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox

    x1 = max(first_x1, second_x1)
    y1 = max(first_y1, second_y1)
    x2 = min(first_x2, second_x2)
    y2 = min(first_y2, second_y2)

    return bbox_area([x1, y1, x2, y2])


def expand_bbox(bbox: list[float], expansion_ratio: float) -> list[float]:
    """Return a bbox expanded by a ratio of its width and height."""

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    expand_x = width * expansion_ratio
    expand_y = height * expansion_ratio

    return [
        x1 - expand_x,
        y1 - expand_y,
        x2 + expand_x,
        y2 + expand_y,
    ]


def point_inside_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    """Return whether a point lies inside or on the edge of a bbox."""

    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def normalized_center_distance(
    point: tuple[float, float],
    bbox: list[float],
) -> float:
    """Return distance from a point to a bbox center normalized by bbox size."""

    center_x, center_y = bbox_center(bbox)
    x1, y1, x2, y2 = bbox
    diagonal = hypot(x2 - x1, y2 - y1)

    if diagonal == 0:
        return float("inf")

    return hypot(point[0] - center_x, point[1] - center_y) / diagonal


def validate_required_ppe(required_ppe: list[str]) -> list[str]:
    """Validate and de-duplicate required PPE in project order."""

    if not required_ppe:
        raise ValueError("At least one required PPE item must be selected.")

    invalid_items = [
        item
        for item in required_ppe
        if item not in PPE_CLASS_NAMES
    ]

    if invalid_items:
        raise ValueError(
            "Required PPE contains unsupported item(s): "
            f"{', '.join(invalid_items)}. "
            f"Supported PPE: {', '.join(PPE_CLASS_NAMES)}."
        )

    return [
        class_name
        for class_name in PPE_CLASS_NAMES
        if class_name in set(required_ppe)
    ]


def split_detections(
    detections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split YOLO detections into person and PPE detections."""

    persons = []
    ppe_detections = []

    for detection in detections:
        class_name = detection.get("class")

        if class_name == PERSON_CLASS_NAME:
            persons.append(detection)
        elif class_name in PPE_CLASS_NAMES:
            ppe_detections.append(detection)

    return persons, ppe_detections


def choose_person_for_ppe(
    ppe_detection: dict[str, Any],
    person_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Assign one PPE detection to the most plausible person.

    The primary rule is center containment: the PPE bbox center must fall inside
    the person bbox. If more than one person contains the center, the smallest
    containing person bbox wins. If that is tied, the person whose center is
    closest to the PPE center wins.
    """

    ppe_center = bbox_center(ppe_detection["bbox_xyxy"])
    candidates = [
        person
        for person in person_records
        if point_inside_bbox(ppe_center, person["person_bbox"])
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda person: (
            bbox_area(person["person_bbox"]),
            normalized_center_distance(ppe_center, person["person_bbox"]),
            person["person_id"],
        ),
    )


def containing_persons_for_ppe(
    ppe_detection: dict[str, Any],
    person_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return person records whose boxes contain the PPE center."""

    ppe_center = bbox_center(ppe_detection["bbox_xyxy"])

    return [
        person
        for person in person_records
        if point_inside_bbox(ppe_center, person["person_bbox"])
    ]


def choose_best_containing_person(
    ppe_detection: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the existing smallest-person and center-distance tie-break."""

    ppe_center = bbox_center(ppe_detection["bbox_xyxy"])

    return min(
        candidates,
        key=lambda person: (
            bbox_area(person["person_bbox"]),
            normalized_center_distance(ppe_center, person["person_bbox"]),
            person["person_id"],
        ),
    )


def assign_ppe_to_persons(
    persons: list[dict[str, Any]],
    ppe_detections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Associate PPE detections to detected people using bbox geometry."""

    person_records = [
        {
            "person_id": index,
            "person_bbox": person["bbox_xyxy"],
            "person_confidence": person.get("confidence"),
            "assigned_ppe": [],
        }
        for index, person in enumerate(persons, start=1)
    ]
    unassigned_ppe = []
    ambiguous_gloves = []
    exclusive_glove_counts = {
        person["person_id"]: 0
        for person in person_records
    }

    for ppe_detection in ppe_detections:
        if ppe_detection["class"] == GLOVE_CLASS_NAME:
            candidates = containing_persons_for_ppe(
                ppe_detection=ppe_detection,
                person_records=person_records,
            )

            if not candidates:
                unassigned_ppe.append(ppe_detection)
            elif len(candidates) == 1:
                owner = candidates[0]
                owner["assigned_ppe"].append(ppe_detection)
                exclusive_glove_counts[owner["person_id"]] += 1
            else:
                ambiguous_gloves.append((ppe_detection, candidates))

            continue

        owner = choose_person_for_ppe(ppe_detection, person_records)

        if owner is None:
            unassigned_ppe.append(ppe_detection)
        else:
            owner["assigned_ppe"].append(ppe_detection)

    for glove_detection, candidates in ambiguous_gloves:
        zero_glove_candidates = [
            person
            for person in candidates
            if count_assigned_ppe(person, GLOVE_CLASS_NAME) == 0
        ]
        candidates_with_exclusive_gloves = [
            person
            for person in candidates
            if exclusive_glove_counts[person["person_id"]] > 0
        ]

        if (
            len(zero_glove_candidates) == 1
            and candidates_with_exclusive_gloves
        ):
            owner = zero_glove_candidates[0]
        else:
            owner = choose_best_containing_person(
                ppe_detection=glove_detection,
                candidates=candidates,
            )

        owner["assigned_ppe"].append(glove_detection)

    return person_records, unassigned_ppe


def count_assigned_ppe(person: dict[str, Any], class_name: str) -> int:
    """Return how many assigned PPE detections match a class."""

    return sum(
        1
        for detection in person["assigned_ppe"]
        if detection["class"] == class_name
    )


def glove_recovery_score(
    glove_detection: dict[str, Any],
    missing_person: dict[str, Any],
) -> tuple[bool, float, float]:
    """
    Return whether a glove can be conservatively recovered for a person.

    A glove is eligible when its center is inside the missing person's bbox
    expanded by 15 percent, or when at least 25 percent of the glove box
    overlaps that expanded person bbox.
    """

    expanded_person_bbox = expand_bbox(
        bbox=missing_person["person_bbox"],
        expansion_ratio=GLOVE_RECOVERY_EXPANSION_RATIO,
    )
    glove_bbox = glove_detection["bbox_xyxy"]
    glove_area = bbox_area(glove_bbox)
    overlap_area = bbox_intersection_area(glove_bbox, expanded_person_bbox)
    overlap_ratio = 0.0 if glove_area == 0 else overlap_area / glove_area
    glove_center = bbox_center(glove_bbox)
    center_inside = point_inside_bbox(glove_center, expanded_person_bbox)
    eligible = (
        center_inside
        or overlap_ratio >= GLOVE_RECOVERY_MIN_OVERLAP_RATIO
    )
    center_distance = normalized_center_distance(
        point=glove_center,
        bbox=expanded_person_bbox,
    )

    return eligible, overlap_area, center_distance


def recover_missing_gloves(
    person_records: list[dict[str, Any]],
    unassigned_ppe: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Conservatively recover one glove for workers with no assigned gloves.

    This pass only considers currently unassigned gloves. Gloves already
    assigned to another worker are never reassigned.
    """

    for missing_person in person_records:
        if count_assigned_ppe(missing_person, GLOVE_CLASS_NAME) > 0:
            continue

        candidates = []

        for glove in unassigned_ppe:
            if glove["class"] != GLOVE_CLASS_NAME:
                continue

            eligible, overlap_area, center_distance = glove_recovery_score(
                glove_detection=glove,
                missing_person=missing_person,
            )

            if eligible:
                candidates.append(
                    {
                        "glove": glove,
                        "donor": None,
                        "overlap_area": overlap_area,
                        "center_distance": center_distance,
                    }
                )

        if not candidates:
            continue

        best_candidate = min(
            candidates,
            key=lambda candidate: (
                -candidate["overlap_area"],
                candidate["center_distance"],
            ),
        )
        recovered_glove = best_candidate["glove"]

        unassigned_ppe.remove(recovered_glove)
        missing_person["assigned_ppe"].append(recovered_glove)

    return person_records, unassigned_ppe


def summarize_person_compliance(
    person_records: list[dict[str, Any]],
    required_ppe: list[str],
) -> list[dict[str, Any]]:
    """Add per-person status and PPE-not-detected lists."""

    summarized_records = []

    for person in person_records:
        assigned_ppe_counts = {
            class_name: count_assigned_ppe(person, class_name)
            for class_name in PPE_CLASS_NAMES
        }
        detected_ppe = [
            class_name
            for class_name in PPE_CLASS_NAMES
            if assigned_ppe_counts[class_name] >= 1
        ]
        missing_ppe = [
            class_name
            for class_name in required_ppe
            if assigned_ppe_counts[class_name] < 1
        ]

        summarized_records.append(
            {
                **person,
                "detected_ppe": detected_ppe,
                "missing_ppe": missing_ppe,
                "ppe_not_detected": missing_ppe,
                "status": COMPLIANT if not missing_ppe else ATTENTION,
            }
        )

    return summarized_records


def build_missing_ppe_summary(
    person_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return compact per-person missing PPE details."""

    return [
        {
            "person_id": person["person_id"],
            "missing_ppe": person["missing_ppe"],
        }
        for person in person_records
        if person["missing_ppe"]
    ]


def evaluate_compliance(
    detections: list[dict[str, Any]],
    required_ppe: list[str],
) -> dict[str, Any]:
    """
    Evaluate safety compliance for each detected person.

    Compliance is person-level: every detected person must have every selected
    PPE class assigned to them. PPE ownership is inferred with a simple bbox
    center-containment heuristic.
    """

    required_ppe = validate_required_ppe(required_ppe)
    persons, ppe_detections = split_detections(detections)
    detected_classes = sorted(
        {
            detection.get("class")
            for detection in detections
            if detection.get("class")
        }
    )

    if not persons:
        return {
            "status": UNKNOWN,
            "persons_detected": 0,
            "required_ppe": required_ppe,
            "persons": [],
            "unassigned_ppe": ppe_detections,
            "missing": [],
            "ppe_not_detected": [],
            "missing_ppe_summary": [],
            "ppe_not_detected_summary": [],
            "detected": detected_classes,
            "compliant_persons": 0,
            "attention_persons": 0,
            "reason": "No person detected. Compliance could not be evaluated.",
        }

    person_records, unassigned_ppe = assign_ppe_to_persons(
        persons=persons,
        ppe_detections=ppe_detections,
    )

    if GLOVE_CLASS_NAME in required_ppe:
        person_records, unassigned_ppe = recover_missing_gloves(
            person_records=person_records,
            unassigned_ppe=unassigned_ppe,
        )

    person_records = summarize_person_compliance(
        person_records=person_records,
        required_ppe=required_ppe,
    )

    attention_persons = [
        person
        for person in person_records
        if person["status"] == ATTENTION
    ]
    missing_ppe_summary = build_missing_ppe_summary(person_records)
    missing_unique = [
        class_name
        for class_name in PPE_CLASS_NAMES
        if any(
            class_name in person["missing_ppe"]
            for person in attention_persons
        )
    ]

    if attention_persons:
        status = ATTENTION
        reason = "Required PPE was not detected for one or more workers"
    else:
        status = COMPLIANT
        reason = None

    return {
        "status": status,
        "persons_detected": len(person_records),
        "required_ppe": required_ppe,
        "persons": person_records,
        "unassigned_ppe": unassigned_ppe,
        "missing": missing_unique,
        "ppe_not_detected": missing_unique,
        "missing_ppe_summary": missing_ppe_summary,
        "ppe_not_detected_summary": missing_ppe_summary,
        "detected": detected_classes,
        "compliant_persons": len(person_records) - len(attention_persons),
        "attention_persons": len(attention_persons),
        "reason": reason,
    }
