"""
Reusable inference helpers for safety compliance evaluation.
"""

from math import ceil, floor
from pathlib import Path

from config.dataset import PERSON_CLASS_NAME
from config.inference_config import CONFIDENCE_THRESHOLD
from src.compliance import (
    assign_ppe_to_persons,
    evaluate_compliance,
    split_detections,
)
from src.path_utils import safe_is_file, windows_safe_path


GOGGLES_CLASS_NAME = "goggles"
GOGGLES_CANDIDATE_THRESHOLD = 0.40
GOGGLES_DIRECT_ACCEPT_THRESHOLD = 0.75
GOGGLES_CROP_CONFIRM_THRESHOLD = 0.40
HEAD_CROP_HEIGHT_RATIO = 0.40
HEAD_CROP_PADDING_RATIO = 0.12
GOGGLES_CROP_IOU = 0.7
GOGGLES_CROP_IMAGE_SIZE = 640
GOGGLES_CROP_DEVICE = 0
PERSON_DUPLICATE_IOU_THRESHOLD = 0.95

PPE_NMS_IOU_BY_CLASS = {
    "helmet": 0.50,
    "vest": 0.50,
    "gloves": 0.15,
    "goggles": 0.30,
}
PPE_NMS_CLASSES = tuple(PPE_NMS_IOU_BY_CLASS)


def detection_iou(
    first_detection: dict,
    second_detection: dict,
) -> float:
    """Return standard IoU for two xyxy detection boxes."""

    first_x1, first_y1, first_x2, first_y2 = first_detection["bbox_xyxy"]
    second_x1, second_y1, second_x2, second_y2 = second_detection["bbox_xyxy"]

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def suppress_duplicate_persons(detections: list[dict]) -> list[dict]:
    """Keep the highest-confidence person for near-identical person boxes."""

    person_detections = [
        detection
        for detection in detections
        if detection["class"] == PERSON_CLASS_NAME
    ]
    kept_persons = []

    for detection in sorted(
        person_detections,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        if all(
            detection_iou(detection, kept_detection)
            < PERSON_DUPLICATE_IOU_THRESHOLD
            for kept_detection in kept_persons
        ):
            kept_persons.append(detection)

    kept_person_ids = {
        id(detection)
        for detection in kept_persons
    }

    return [
        detection
        for detection in detections
        if detection["class"] != PERSON_CLASS_NAME
        or id(detection) in kept_person_ids
    ]


def nms_one_ppe_class(
    detections: list[dict],
    iou_threshold: float,
) -> list[dict]:
    """Keep highest-confidence same-class PPE boxes after IoU NMS."""

    kept_detections = []

    for detection in sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        if all(
            detection_iou(detection, kept_detection) <= iou_threshold
            for kept_detection in kept_detections
        ):
            kept_detections.append(detection)

    return kept_detections


def apply_second_stage_ppe_nms(detections: list[dict]) -> list[dict]:
    """Apply class-aware NMS to PPE detections using per-class IoU thresholds."""

    person_detections = [
        detection
        for detection in detections
        if detection["class"] == PERSON_CLASS_NAME
    ]
    other_detections = [
        detection
        for detection in detections
        if detection["class"] not in PPE_NMS_CLASSES
        and detection["class"] != PERSON_CLASS_NAME
    ]
    ppe_detections = [
        detection
        for detection in detections
        if detection["class"] in PPE_NMS_CLASSES
    ]

    kept_ppe_by_class = []

    for class_name in PPE_NMS_CLASSES:
        class_detections = [
            detection
            for detection in ppe_detections
            if detection["class"] == class_name
        ]
        kept_ppe_by_class.extend(
            nms_one_ppe_class(
                detections=class_detections,
                iou_threshold=PPE_NMS_IOU_BY_CLASS[class_name],
            )
        )

    return person_detections + kept_ppe_by_class + other_detections


def is_direct_goggles_accept(detection: dict) -> bool:
    """Return whether a goggles detection is accepted without crop verification."""

    return (
        detection["class"] == GOGGLES_CLASS_NAME
        and detection["confidence"] >= GOGGLES_DIRECT_ACCEPT_THRESHOLD
    )


def is_borderline_goggles_candidate(detection: dict) -> bool:
    """Return whether a goggles detection needs head-crop confirmation."""

    return (
        detection["class"] == GOGGLES_CLASS_NAME
        and GOGGLES_CANDIDATE_THRESHOLD
        <= detection["confidence"]
        < GOGGLES_DIRECT_ACCEPT_THRESHOLD
    )


def build_head_crop(image, person_bbox: list[float]):
    """Return an enlarged upper-person crop for eyewear verification."""

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = person_bbox
    person_width = max(0.0, x2 - x1)
    person_height = max(0.0, y2 - y1)

    if person_width == 0 or person_height == 0:
        return None

    head_y2 = y1 + person_height * HEAD_CROP_HEIGHT_RATIO
    padding_x = person_width * HEAD_CROP_PADDING_RATIO
    padding_y = person_height * HEAD_CROP_PADDING_RATIO

    crop_x1 = max(0, floor(x1 - padding_x))
    crop_y1 = max(0, floor(y1 - padding_y))
    crop_x2 = min(image_width, ceil(x2 + padding_x))
    crop_y2 = min(image_height, ceil(head_y2 + padding_y))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def crop_confirms_goggles(model, crop) -> bool:
    """Return whether crop inference confirms visible goggles."""

    if crop is None or crop.size == 0:
        return False

    crop_results = model.predict(
        source=crop,
        conf=CONFIDENCE_THRESHOLD,
        iou=GOGGLES_CROP_IOU,
        imgsz=GOGGLES_CROP_IMAGE_SIZE,
        device=GOGGLES_CROP_DEVICE,
        verbose=False,
        save=False,
    )
    crop_detections = extract_detections(crop_results[0])

    return any(
        detection["class"] == GOGGLES_CLASS_NAME
        and detection["confidence"] >= GOGGLES_CROP_CONFIRM_THRESHOLD
        for detection in crop_detections
    )


def verify_goggles_detections(
    model,
    result,
    detections: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Verify borderline goggles detections with one head crop per person."""

    stats = {
        "goggles_below_candidate_rejected": 0,
        "borderline_goggles_candidates": 0,
        "borderline_goggles_confirmed": 0,
        "borderline_goggles_rejected": 0,
        "direct_goggles_accepted": 0,
    }

    persons, ppe_detections = split_detections(detections)
    person_records, _ = assign_ppe_to_persons(
        persons=persons,
        ppe_detections=ppe_detections,
    )
    owner_by_detection_id = {
        id(detection): person
        for person in person_records
        for detection in person["assigned_ppe"]
    }
    crop_confirmation_cache: dict[int, bool] = {}
    verified_detections = []

    for detection in detections:
        if detection["class"] != GOGGLES_CLASS_NAME:
            verified_detections.append(detection)
            continue

        if detection["confidence"] < GOGGLES_CANDIDATE_THRESHOLD:
            stats["goggles_below_candidate_rejected"] += 1
            continue

        if is_direct_goggles_accept(detection):
            verified_detections.append(detection)
            stats["direct_goggles_accepted"] += 1
            continue

        if not is_borderline_goggles_candidate(detection):
            continue

        stats["borderline_goggles_candidates"] += 1
        owner = owner_by_detection_id.get(id(detection))

        if owner is None:
            stats["borderline_goggles_rejected"] += 1
            continue

        person_id = owner["person_id"]

        if person_id not in crop_confirmation_cache:
            crop_confirmation_cache[person_id] = crop_confirms_goggles(
                model=model,
                crop=build_head_crop(
                    image=result.orig_img,
                    person_bbox=owner["person_bbox"],
                ),
            )

        if crop_confirmation_cache[person_id]:
            verified_detections.append(detection)
            stats["borderline_goggles_confirmed"] += 1
        else:
            stats["borderline_goggles_rejected"] += 1

    return verified_detections, stats


def extract_detections(result):
    """
    Extract JSON-serializable detections from an Ultralytics result.

    Parameters
    ----------
    result : ultralytics.engine.results.Results
        Single-image YOLO prediction result.

    Returns
    -------
    list[dict]
        Detected objects with class, confidence, and bounding box.
    """

    detections = []
    boxes = result.boxes

    if boxes is None:
        return detections

    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    xyxy_boxes = boxes.xyxy.tolist()

    for class_id, confidence, xyxy_box in zip(
        class_ids,
        confidences,
        xyxy_boxes,
    ):
        detections.append(
            {
                "class": result.names[int(class_id)],
                "confidence": round(float(confidence), 4),
                "bbox_xyxy": [
                    round(float(value), 2)
                    for value in xyxy_box
                ],
            }
        )

    return detections


def evaluate_image(model, image_path, required_ppe):
    """
    Run detection and compliance evaluation for one image.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded YOLO model.

    image_path : str or Path
        Image to evaluate.

    required_ppe : list[str]
        Required PPE classes.

    Returns
    -------
    dict
        Evaluation data reused by visualization and reporting.
    """

    image_path = Path(image_path)

    if not safe_is_file(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(
        source=windows_safe_path(image_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]
    detections = apply_second_stage_ppe_nms(
        suppress_duplicate_persons(
            extract_detections(result)
        )
    )
    detections, goggles_verification = verify_goggles_detections(
        model=model,
        result=result,
        detections=detections,
    )
    detected_classes = [
        detection["class"]
        for detection in detections
    ]

    compliance_result = evaluate_compliance(
        detections=detections,
        required_ppe=required_ppe,
    )

    return {
        "image_path": image_path,
        "result": result,
        "detections": detections,
        "detected_classes": detected_classes,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "goggles_verification": goggles_verification,
        "compliance_result": compliance_result,
    }
