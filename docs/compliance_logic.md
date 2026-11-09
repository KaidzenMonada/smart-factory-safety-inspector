# Compliance Logic

## 1. Purpose

The Smart Factory Safety Inspector uses a YOLO object detector to recognize:

- `person`
- `helmet`
- `vest`
- `gloves`
- `goggles`

Compliance is evaluated against a user-selected required PPE set. The system does not infer requirements from folder names or image paths.

## 2. End-to-End Decision Pipeline

Current inference flow:

```text
Image
  ->
YOLO inference
  ->
Detection extraction
  ->
Person duplicate suppression
  ->
PPE-specific suppression
  ->
Goggles verification
  ->
PPE-to-person association
  ->
Glove-specific assignment/recovery
  ->
Worker compliance
  ->
Image-level status
  ->
Visualization / batch result
```

The annotated output image is drawn from the filtered detections that are actually used for compliance.

## 3. Detection Stage

The detector is loaded from the final project model and is expected to use the project class schema from `config/dataset.py`.

| Setting | Current value |
|---|---:|
| General inference confidence threshold | `0.25` |
| Supported detection classes | `person`, `helmet`, `vest`, `gloves`, `goggles` |
| Selectable PPE requirements | `helmet`, `vest`, `gloves`, `goggles` |

The general confidence threshold is applied during YOLO prediction. Additional filtering is applied only by the explicit rules documented below.

## 4. Duplicate Person Suppression

After YOLO detection extraction, the current pipeline applies a person-only duplicate suppression step.

| Rule | Current value |
|---|---:|
| Duplicate person IoU threshold | `0.95` |
| Class affected | `person` only |
| Kept detection | Higher-confidence person box |

If two person detections have IoU greater than or equal to `0.95`, they are treated as near-identical detections of the same physical worker, and only the higher-confidence detection is kept.

This rule exists because duplicate person detections can create artificial workers and false `ATTENTION` decisions. PPE detections are not affected by this person-only rule.

## 5. PPE-Specific NMS

After person duplicate suppression, the pipeline applies a second-stage class-aware NMS only to PPE classes.

| PPE class | IoU threshold |
|---|---:|
| `helmet` | `0.50` |
| `vest` | `0.50` |
| `gloves` | `0.15` |
| `goggles` | `0.30` |

Within each PPE class, detections are processed from highest confidence to lowest confidence. A detection is kept only if its IoU with every already-kept detection of the same class is less than or equal to that class threshold.

This step does not suppress person detections.

## 6. Goggles Verification

Goggles receive an additional verification pass after PPE NMS and before compliance evaluation.

| Parameter | Current value |
|---|---:|
| Goggles candidate threshold | `0.40` |
| Goggles direct-accept threshold | `0.75` |
| Crop-confirm threshold | `0.40` |
| Head-crop height ratio | `0.40` |
| Head-crop padding ratio | `0.12` |
| Crop inference IoU | `0.70` |
| Crop inference image size | `640` |

Current goggles rules:

| Detection condition | Result |
|---|---|
| `goggles` confidence `< 0.40` | Rejected. |
| `goggles` confidence `>= 0.75` | Accepted directly. |
| `0.40 <= goggles confidence < 0.75` | Requires head-crop verification. |

For borderline goggles, the pipeline first identifies the owner person using the same PPE-to-person association rule. It then crops the upper portion of that person box: the top `40%` of the person height, padded by `12%` of the person width and height, clipped to image bounds. The same detector is run on that crop. The original goggles detection is kept only if crop inference detects goggles with confidence at least `0.40`.

Rejected goggles are removed before worker assignment, compliance evaluation, and visualization.

Limitation: this verification reduces some borderline false positives, but it still uses the same detector. If the detector systematically confuses facial or eye regions with goggles, crop verification cannot eliminate every such error.

## 7. PPE-to-Person Association

YOLO detects workers and PPE independently, so PPE ownership is inferred geometrically.

For normal PPE classes (`helmet`, `vest`, `goggles`):

1. Compute the PPE bounding-box center.
2. Find person boxes containing that center point.
3. If exactly one person contains it, assign the PPE to that person.
4. If multiple people contain it, choose the smallest containing person box.
5. If size is tied, choose the person whose box center is nearest to the PPE center after normalization by that person box diagonal.
6. If still tied, choose the lower `person_id`.
7. If no person contains the PPE center, leave the PPE unassigned.

Unassigned PPE remains visible in the compliance result, but it does not automatically satisfy any worker requirement.

## 8. Glove-Specific Assignment Logic

Gloves use additional logic because hands are small, often close to worker boundaries, and frequently overlap in multi-person scenes.

### Primary glove assignment

After PPE NMS, glove detections are classified by center containment:

| Glove type | Meaning | Current handling |
|---|---|---|
| Exclusive | The glove center is inside exactly one person box. | Assigned first to that person. |
| Ambiguous | The glove center is inside more than one person box. | Assigned after exclusive gloves. |
| Unassigned | The glove center is inside no person box. | Left for optional recovery. |

For ambiguous gloves:

1. Consider only the person boxes that contain the glove center.
2. If exactly one candidate worker currently has zero assigned gloves while another candidate already has at least one exclusive glove, assign the ambiguous glove to the zero-glove worker.
3. Otherwise, use the standard fallback: smallest containing person, then normalized center-distance tie-break, then `person_id`.

Already assigned exclusive gloves are never transferred from one worker to another.

### Glove recovery

Glove recovery runs only when `gloves` are part of the selected required PPE set. It runs only for workers with zero assigned gloves, and it considers only truly unassigned glove detections.

| Recovery parameter | Current value |
|---|---:|
| Person-box expansion ratio | `0.15` |
| Minimum glove overlap ratio | `0.25` |

An unassigned glove is eligible for recovery if either:

- its center is inside the worker box expanded by `15%`; or
- at least `25%` of the glove box area overlaps that expanded worker box.

If multiple unassigned gloves are eligible, the chosen glove maximizes overlap area first and then minimizes normalized center distance.

One valid assigned `gloves` detection is sufficient to satisfy the gloves requirement for a worker.

## 9. Configurable PPE Requirements

The required PPE set is selected by the user before inference. Examples include:

- `helmet`
- `helmet` + `vest`
- `helmet` + `vest` + `gloves`
- `helmet` + `vest` + `gloves` + `goggles`

Folder names such as `Compliant`, `Attention`, `required_all_ppe`, or `required_helmet_vest` do not control the requirement set. Required PPE remains manually selected through the existing workflow.

## 10. Worker-Level Status

| Status | Meaning |
|---|---|
| `COMPLIANT` | All currently required PPE items were detected and assigned to the worker. |
| `ATTENTION` | At least one currently required PPE item was not detected or assigned to the worker. |

`ATTENTION` is detection-based. It means the system did not detect the required PPE for that worker. It does not prove that the worker physically lacks the PPE.

## 11. Image-Level Status

| Status | Current implementation |
|---|---|
| `COMPLIANT` | At least one person was detected, and every detected worker is compliant. |
| `ATTENTION` | At least one detected worker is missing one or more required PPE items according to the detection result. |
| `UNKNOWN` | No person was detected, so compliance could not be evaluated. |

When no person is detected, all PPE detections are reported as unassigned and the image-level status is `UNKNOWN`.

## 12. Unassigned PPE

A PPE detection is unassigned when the association logic cannot attach it to a detected worker. Unassigned PPE can happen when:

- the PPE center is outside all person boxes;
- the related person was not detected;
- the person box is too small, shifted, or overlapping in a difficult way;
- the PPE detection itself is false or poorly localized.

Unassigned PPE does not make a worker compliant. Only PPE assigned to that worker counts toward the required PPE set.

## 13. Output Behavior

Single-image workflow:

- prompts the user to select required PPE;
- runs inference on one image;
- prints the image-level and worker-level compliance result;
- saves one annotated prediction image.

Batch workflow:

- prompts the user to select required PPE;
- processes supported images recursively under the selected input folder;
- mirrors nested input folders under `outputs/predictions/<input-folder-name>/`;
- saves annotated images using the original input filenames, overwriting prior annotated images for the same source image;
- writes `batch_compliance_results.csv`.

## 14. Important System Limitations

The compliance layer is intentionally conservative, but it still depends on detector quality and geometric association.

| Limitation | Effect |
|---|---|
| Missed PPE detections | Can create false `ATTENTION` results. |
| False person detections | Can create extra workers and false `ATTENTION` results. |
| Goggles are visually difficult | Eye and face regions can still be confused with goggles. |
| Overlapping workers | PPE ownership can be ambiguous even with glove-specific handling. |
| Detection-based semantics | `ATTENTION` means required PPE was not detected, not that physical PPE absence is proven. |

These limitations were observed during final inference testing and should be considered when interpreting application output.

## 15. Summary

The final compliance system converts YOLO object detections into worker-centric safety decisions. It filters near-duplicate person boxes, applies PPE-specific suppression, verifies borderline goggles, associates PPE to workers, handles difficult glove assignment cases, and reports conservative `COMPLIANT`, `ATTENTION`, or `UNKNOWN` statuses.

The system is a computer-vision support tool. It highlights detected safety concerns while preserving uncertainty when required PPE is not detected.
