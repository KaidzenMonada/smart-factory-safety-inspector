from datetime import datetime
from pathlib import Path

import cv2

from src.path_utils import safe_filename_stem, windows_safe_path


BOX_COLORS = {
    "person": (255, 0, 0),
    "helmet": (255, 255, 0),
    "vest": (255, 255, 255),
    "gloves": (208, 224, 64),
    "goggles": (139, 0, 0),
}
STATUS_COLORS = {
    "COMPLIANT": (0, 110, 0),
    "ATTENTION": (0, 105, 190),
    "UNKNOWN": (70, 70, 70),
}
PPE_LABEL_BACKGROUNDS = {
    "helmet": (150, 140, 0),
    "vest": (55, 55, 55),
    "gloves": (120, 130, 0),
    "goggles": (105, 0, 0),
}
TEXT_FOREGROUND = (255, 255, 255)
TEXT_OUTLINE = (0, 0, 0)
PPE_BOX_THICKNESS = 2
WORKER_BOX_THICKNESS = 3
TEXT_THICKNESS = 2
TEXT_OUTLINE_THICKNESS = 4
MIN_VISUAL_FONT_SCALE = 0.50
PPE_FONT_BASE = 0.58
WORKER_FONT_BASE = 0.62
BANNER_TITLE_FONT_BASE = 0.72
BANNER_SUBTITLE_FONT_BASE = 0.55
LABEL_PADDING_X = 6
LABEL_PADDING_Y = 4
LABEL_GAP = 4
BANNER_ALPHA = 0.90


def visual_scale(image) -> float:
    """Return an adaptive visualization scale based on image size."""

    image_height, image_width = image.shape[:2]
    min_dimension = min(image_height, image_width)
    return max(0.88, min(1.35, min_dimension / 1100))


def scaled_font(image, base_scale: float) -> float:
    """Return a readable OpenCV font scale with a strict lower bound."""

    return max(MIN_VISUAL_FONT_SCALE, base_scale * visual_scale(image))


def text_thickness(image) -> int:
    """Return readable text stroke thickness for saved annotations."""

    if min(image.shape[:2]) < 1600:
        return TEXT_THICKNESS

    return TEXT_THICKNESS + 1


def outline_thickness(image) -> int:
    """Return dark outline thickness for high-contrast text."""

    return max(TEXT_OUTLINE_THICKNESS, text_thickness(image) + 2)


def clipped_bbox(bbox: list[float], image) -> tuple[int, int, int, int]:
    """Return an integer bbox clipped to image boundaries."""

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    clipped_x1 = max(0, min(image_width - 1, int(round(x1))))
    clipped_y1 = max(0, min(image_height - 1, int(round(y1))))
    clipped_x2 = max(0, min(image_width - 1, int(round(x2))))
    clipped_y2 = max(0, min(image_height - 1, int(round(y2))))

    return (
        min(clipped_x1, clipped_x2),
        min(clipped_y1, clipped_y2),
        max(clipped_x1, clipped_x2),
        max(clipped_y1, clipped_y2),
    )


def draw_outlined_text(
    image,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw white text with a dark outline for contrast."""

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_OUTLINE,
        outline_thickness(image),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_FOREGROUND,
        thickness,
        cv2.LINE_AA,
    )


def label_block_size(
    lines: list[str],
    font_scale: float,
    thickness: int,
) -> tuple[int, int, int]:
    """Return width, height, and line step for a filled text block."""

    text_sizes = [
        cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0]
        for line in lines
    ]
    max_width = max(width for width, _ in text_sizes)
    max_height = max(height for _, height in text_sizes)
    line_step = max_height + max(5, int(5 * font_scale))
    block_width = max_width + (LABEL_PADDING_X * 2)
    block_height = (
        (max_height * len(lines))
        + (line_step - max_height) * max(0, len(lines) - 1)
        + (LABEL_PADDING_Y * 2)
    )

    return block_width, block_height, line_step


def clamp_label_rect(
    image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip a label rectangle to the image boundary."""

    image_height, image_width = image.shape[:2]
    width = min(max(1, width), image_width)
    height = min(max(1, height), image_height)
    x = max(0, min(x, image_width - width))
    y = max(0, min(y, image_height - height))

    return (x, y, x + width - 1, y + height - 1)


def rectangles_overlap(
    first_rect: tuple[int, int, int, int],
    second_rect: tuple[int, int, int, int],
) -> bool:
    """Return whether two clipped label rectangles overlap."""

    return not (
        first_rect[2] < second_rect[0]
        or second_rect[2] < first_rect[0]
        or first_rect[3] < second_rect[1]
        or second_rect[3] < first_rect[1]
    )


def place_label_rect(
    image,
    x: int,
    y: int,
    width: int,
    height: int,
    used_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Place a label, moving later labels down or up to reduce overlap."""

    offset = height + LABEL_GAP
    candidate_y_values = [y]

    for step in range(1, 8):
        candidate_y_values.append(y + step * offset)

    for step in range(1, 8):
        candidate_y_values.append(y - step * offset)

    for candidate_y in candidate_y_values:
        rect = clamp_label_rect(
            image=image,
            x=x,
            y=candidate_y,
            width=width,
            height=height,
        )

        if not any(
            rectangles_overlap(rect, used_rect)
            for used_rect in used_rects
        ):
            return rect

    return clamp_label_rect(
        image=image,
        x=x,
        y=y,
        width=width,
        height=height,
    )


def draw_label_block(
    image,
    lines: list[str],
    preferred_top_left: tuple[int, int],
    background_color: tuple[int, int, int],
    font_scale: float,
    used_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Draw a compact filled label block with outlined white text."""

    thickness = text_thickness(image)
    block_width, block_height, line_step = label_block_size(
        lines=lines,
        font_scale=font_scale,
        thickness=thickness,
    )
    rect = place_label_rect(
        image=image,
        x=preferred_top_left[0],
        y=preferred_top_left[1],
        width=block_width,
        height=block_height,
        used_rects=used_rects,
    )
    x1, y1, _, _ = rect

    cv2.rectangle(
        image,
        (rect[0], rect[1]),
        (rect[2], rect[3]),
        background_color,
        -1,
    )

    text_y = y1 + LABEL_PADDING_Y + cv2.getTextSize(
        lines[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0][1]

    for line in lines:
        draw_outlined_text(
            image=image,
            text=line,
            origin=(x1 + LABEL_PADDING_X, text_y),
            font_scale=font_scale,
            thickness=thickness,
        )
        text_y += line_step

    used_rects.append(rect)
    return rect


def ppe_label_position(
    image,
    bbox: tuple[int, int, int, int],
    label: str,
    font_scale: float,
) -> tuple[int, int]:
    """Return a top-left label position for a PPE box."""

    x1, y1, x2, y2 = bbox
    width, height, _ = label_block_size(
        lines=[label],
        font_scale=font_scale,
        thickness=text_thickness(image),
    )

    if (x2 - x1) >= width and (y2 - y1) >= height:
        return x1 + 1, y1 + 1

    return x1, y1 - height - 2


def worker_label_position(
    image,
    bbox: tuple[int, int, int, int],
    lines: list[str],
    font_scale: float,
) -> tuple[int, int]:
    """Return a nearby top-left position for a worker status block."""

    x1, y1, _, _ = bbox
    _, height, _ = label_block_size(
        lines=lines,
        font_scale=font_scale,
        thickness=text_thickness(image),
    )

    if y1 - height - 2 >= 0:
        return x1, y1 - height - 2

    return x1 + 2, y1 + 2


def status_banner_rect(
    image,
    compliance_result: dict,
) -> tuple[int, int, int, int]:
    """Return a compact top-left banner rectangle sized for its text."""

    image_height, image_width = image.shape[:2]
    required_ppe = ", ".join(compliance_result.get("required_ppe", []))
    title_text = f"STATUS: {compliance_result['status']}"
    required_text = f"Required: {required_ppe}"
    title_scale = scaled_font(image, BANNER_TITLE_FONT_BASE)
    subtitle_scale = scaled_font(image, BANNER_SUBTITLE_FONT_BASE)
    thickness = text_thickness(image)
    title_width, title_height = cv2.getTextSize(
        title_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        thickness,
    )[0]
    subtitle_width, subtitle_height = cv2.getTextSize(
        required_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        subtitle_scale,
        thickness,
    )[0]
    padding_x = max(10, int(10 * visual_scale(image)))
    padding_y = max(7, int(7 * visual_scale(image)))
    text_gap = max(5, int(5 * visual_scale(image)))
    text_width = max(title_width, subtitle_width)
    text_height = title_height + subtitle_height + text_gap
    target_width = int(image_width * 0.32)
    target_height = int(image_height * 0.09)
    banner_width = max(target_width, text_width + (padding_x * 2))
    banner_height = max(target_height, text_height + (padding_y * 2))

    return clamp_label_rect(
        image=image,
        x=6,
        y=6,
        width=min(banner_width, image_width - 12),
        height=min(banner_height, image_height - 12),
    )


def draw_detection_boxes(
    image,
    detections: list[dict],
    used_rects: list[tuple[int, int, int, int]],
) -> None:
    """Draw compact filtered PPE boxes used by compliance."""

    font_scale = scaled_font(image, PPE_FONT_BASE)

    for detection in detections:
        class_name = detection["class"]

        if class_name == "person":
            continue

        color = BOX_COLORS.get(class_name, (220, 220, 220))
        background_color = PPE_LABEL_BACKGROUNDS.get(class_name, (55, 55, 55))
        bbox = clipped_bbox(detection["bbox_xyxy"], image)
        x1, y1, x2, y2 = bbox
        label = f"{class_name} {detection['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, PPE_BOX_THICKNESS)
        draw_label_block(
            image=image,
            lines=[label],
            preferred_top_left=ppe_label_position(
                image=image,
                bbox=bbox,
                label=label,
                font_scale=font_scale,
            ),
            background_color=background_color,
            font_scale=font_scale,
            used_rects=used_rects,
        )


def draw_worker_statuses(
    image,
    compliance_result: dict,
    used_rects: list[tuple[int, int, int, int]],
) -> None:
    """Draw person boxes with compact worker status labels."""

    font_scale = scaled_font(image, WORKER_FONT_BASE)

    for person in compliance_result.get("persons", []):
        bbox = clipped_bbox(person["person_bbox"], image)
        x1, y1, x2, y2 = bbox
        person_color = BOX_COLORS["person"]
        status_color = STATUS_COLORS.get(person["status"], person_color)
        confidence = person.get("person_confidence")
        confidence_text = "" if confidence is None else f" {confidence:.2f}"
        label_lines = [f"W{person['person_id']} {person['status']}{confidence_text}"]

        if person["status"] == "ATTENTION" and person.get("missing_ppe"):
            label_lines.append("Missing: " + ", ".join(person["missing_ppe"]))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            person_color,
            WORKER_BOX_THICKNESS,
        )
        draw_label_block(
            image=image,
            lines=label_lines,
            preferred_top_left=worker_label_position(
                image=image,
                bbox=bbox,
                lines=label_lines,
                font_scale=font_scale,
            ),
            background_color=status_color,
            font_scale=font_scale,
            used_rects=used_rects,
        )


def draw_status_banner(
    image,
    compliance_result: dict,
    rect: tuple[int, int, int, int],
) -> None:
    """Draw a small top-left global status banner."""

    status = compliance_result["status"]
    required_ppe = ", ".join(compliance_result.get("required_ppe", []))
    status_text = f"STATUS: {status}"
    required_text = f"Required: {required_ppe}"
    banner_color = STATUS_COLORS.get(status, (80, 80, 80))
    overlay = image.copy()
    x1, y1, x2, y2 = rect

    cv2.rectangle(overlay, (x1, y1), (x2, y2), banner_color, -1)
    cv2.addWeighted(overlay, BANNER_ALPHA, image, 1 - BANNER_ALPHA, 0, image)

    padding_x = max(10, int(10 * visual_scale(image)))
    padding_y = max(7, int(7 * visual_scale(image)))
    text_gap = max(5, int(5 * visual_scale(image)))
    title_scale = scaled_font(image, BANNER_TITLE_FONT_BASE)
    subtitle_scale = scaled_font(image, BANNER_SUBTITLE_FONT_BASE)
    thickness = text_thickness(image)
    title_height = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        thickness,
    )[0][1]
    subtitle_height = cv2.getTextSize(
        required_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        subtitle_scale,
        thickness,
    )[0][1]

    draw_outlined_text(
        image=image,
        text=status_text,
        origin=(x1 + padding_x, y1 + padding_y + title_height),
        font_scale=title_scale,
        thickness=thickness,
    )
    draw_outlined_text(
        image=image,
        text=required_text,
        origin=(
            x1 + padding_x,
            y1 + padding_y + title_height + text_gap + subtitle_height,
        ),
        font_scale=subtitle_scale,
        thickness=thickness,
    )


def build_annotated_image(
    image,
    detections: list[dict],
    compliance_result: dict,
):
    """Build the final clean annotation image from filtered detections."""

    annotated_image = image.copy()
    used_label_rects = [
        status_banner_rect(
            image=annotated_image,
            compliance_result=compliance_result,
        )
    ]

    draw_detection_boxes(
        image=annotated_image,
        detections=detections,
        used_rects=used_label_rects,
    )
    draw_worker_statuses(
        image=annotated_image,
        compliance_result=compliance_result,
        used_rects=used_label_rects,
    )
    draw_status_banner(
        image=annotated_image,
        compliance_result=compliance_result,
        rect=used_label_rects[0],
    )

    return annotated_image


def save_prediction_image(
    image,
    compliance_result,
    output_path,
    filename_prefix="prediction",
    include_microseconds=False,
    detections=None,
    output_filename=None,
):
    """
    Save the annotated prediction image.

    When detections are provided, they are the filtered detections used by
    compliance and are drawn with the project visualization style.
    """

    if detections is not None:
        image = build_annotated_image(
            image=image,
            detections=detections,
            compliance_result=compliance_result,
        )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_filename is not None:
        output_file = output_path / Path(output_filename).name
        output_suffix = output_file.suffix or ".jpg"
    else:
        if include_microseconds:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_prefix = safe_filename_stem(filename_prefix)
        output_file = output_path / f"{safe_prefix}_{timestamp}.jpg"
        output_suffix = ".jpg"

    success, encoded_image = cv2.imencode(output_suffix, image)

    if not success:
        raise ValueError("OpenCV could not encode prediction image.")

    encoded_image.tofile(windows_safe_path(output_file))

    return output_file
