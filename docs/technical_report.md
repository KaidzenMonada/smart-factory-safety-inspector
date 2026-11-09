# Smart Factory Safety Inspector

## Technical Report

## 1. Executive Summary

Smart Factory Safety Inspector is an end-to-end computer vision project for detecting workers and personal protective equipment, then converting those detections into worker-level safety compliance decisions. The detector recognizes five classes: `person`, `helmet`, `vest`, `gloves`, and `goggles`. The application then evaluates each detected worker against a user-selected PPE requirement set.

The project is technically broader than a standard object-detection training exercise. A custom PPE dataset was constructed and curated from multiple source pools, cleaned through automated audits, reviewed in CVAT and related manual workflows, supplemented with reviewed synthetic data, and checked for duplicate/source leakage. The current canonical dataset contains **3,865 images** and **23,686 annotations**.

Six YOLO candidate models were trained and compared under a shared configuration. YOLO26m achieved the highest validation mAP50-95 by a very small margin, but YOLO26s was selected because it preserved nearly identical validation quality with much lower model size and compute cost. The selected YOLO26s checkpoint is stored at `models/best.pt`.

On the held-out test split, YOLO26s achieved:

| Metric | Value |
|---|---:|
| Precision | 0.8846 |
| Recall | 0.7708 |
| mAP50 | 0.8371 |
| mAP50-95 | 0.5833 |

The complete compliance system was then tested on 160 final inference images across 8 PPE requirement groups. It correctly classified 151 images, for **94.38% application-level accuracy**. The most important safety-oriented result was **0 false COMPLIANT cases**: none of the expected `ATTENTION` images were incorrectly passed by the system. This is a conservative result for the final test set, but it is not a real-world safety certification.

## 2. Problem Definition

The core machine learning task is PPE object detection in worker scenes. The model must localize workers and four PPE categories:

| Class ID | Class |
|---:|---|
| 0 | person |
| 1 | helmet |
| 2 | vest |
| 3 | gloves |
| 4 | goggles |

Object detection alone is not enough for the final application. An image can contain several workers and several PPE detections. The system needs to decide which PPE belongs to which worker and whether each worker satisfies the currently required PPE set.

This makes the application worker-centric rather than image-centric. Detecting a helmet somewhere in the image does not mean every worker has a helmet. Similarly, detecting gloves or goggles does not satisfy a requirement unless those detections can be assigned to a detected worker. The final system therefore combines object detection with geometric association, glove-specific assignment handling, goggles verification, duplicate person suppression, and conservative status semantics.

The required PPE set is configurable. A user may check only helmets, helmets plus vests, gloves only, goggles only, or all PPE classes. Folder names are not used to infer requirements; required PPE is selected manually in the active single-image and batch workflows.

## 3. Dataset Construction

The dataset was built as a custom project asset. Public PPE datasets and raw image pools were useful starting points, but they were not reliable enough to merge directly. The raw material contained inconsistent class definitions, missing person annotations, missing PPE annotations, wrong classes, duplicate images, source variants, off-domain scenes, and very small PPE objects that needed visual review.

The dataset-building process included:

| Stage | Role |
|---|---|
| Raw source collection | Gathered worker and PPE scenes from multiple source pools and source-family groups. |
| Automated cleaning | Standardized labels, removed unwanted classes, checked image readability, validated YOLO labels, and audited duplicate/source risk. |
| Manual review | Used CVAT and connected review workflows to inspect images, correct annotations, and reject unreliable candidates. |
| Synthetic supplementation | Added reviewed synthetic training data for useful PPE combinations and difficult cases. |
| Consolidation | Materialized a leakage-aware YOLO dataset with deterministic split structure. |
| Final project QA | Removed confirmed duplicate/source variants and repaired confirmed annotation issues before training. |

Approximately **13,000 images** were manually reviewed in CVAT or connected manual review workflows. This should be interpreted precisely: it describes large-scale manual inspection, annotation review, correction, and filtering workflows, not a claim that every image was annotated from a blank canvas. Manual work included correcting boxes, adding missing person or PPE labels where supported, deleting false labels, checking small gloves and goggles, and rejecting images that were too ambiguous or off-task.

The Dataset Builder snapshot described in `notebooks/00_Dataset_Creation_and_Curation.ipynb` contained **3,882 images** and **23,772 annotations**. The current canonical dataset is slightly smaller because later final-project QA removed confirmed duplicate/source variants and repaired confirmed annotation problems. That distinction matters: the builder snapshot documents the construction milestone, while `data/final_dataset/` is the final dataset used for training, validation, and testing.

## 4. Final Dataset

The current canonical dataset is stored at `data/final_dataset/` and uses `data/final_dataset/data.yaml`.

| Split | Images |
|---|---:|
| Train | 3,219 |
| Validation | 323 |
| Test | 323 |
| **Total** | **3,865** |

The final annotation count is **23,686**.

| Class | Annotations |
|---|---:|
| person | 7,973 |
| helmet | 4,371 |
| vest | 3,219 |
| gloves | 4,999 |
| goggles | 3,124 |
| **Total** | **23,686** |

The final processing stage was intentionally minimal. It removed 17 confirmed same-source image copies and their labels, removed 3 confirmed false `goggles` rows, and added 7 missing `person` annotations for 4 confirmed images with PPE labels but no person labels. No split recreation, class rebalancing, broad annotation cleanup, or heuristic candidate editing was performed.

Post-processing validation confirmed:

| Check | Result |
|---|---:|
| Missing image/label pairs | 0 |
| Unsupported or unreadable images | 0 |
| Empty label files | 0 |
| Malformed or unusable annotations | 0 |
| Invalid class IDs | 0 |
| Duplicate annotation rows | 0 |
| Large boundary overflows | 0 |
| Exact duplicate image pairs | 0 |
| Perceptual near-duplicate candidates | 0 |
| Cross-split leakage candidates | 0 |

The dataset therefore entered training with no confirmed structural, label-validity, duplicate, or leakage problem.

## 5. Exploratory Data Analysis and Quality Assurance

The final EDA was implemented as source scripts under `eda/`, with the notebook `notebooks/01_Final_Dataset_EDA.ipynb` serving only as a code-free explanation of generated evidence. This separation made the EDA reproducible while keeping the notebook readable for review.

The EDA checked dataset structure, image-label pairing, image readability, dimensions, YOLO row validity, boundary geometry, class distribution, bounding-box sizes, person/PPE relationships, multiple same-class PPE per worker, exact duplicates, and perceptual near-duplicate candidates.

Important EDA findings after final cleanup included:

| Finding | Count | Interpretation |
|---|---:|---|
| Total usable annotation rows | 23,686 | Final label rows available for training. |
| Malformed or unusable rows | 0 | No critical label-validity issue remained. |
| Tiny boundary overflow rows | 854 | Review observation, not a blocking annotation error. |
| Large boundary overflow rows | 0 | No large geometry overflow remained. |
| Unassigned PPE under association rule | 133 | Review observation caused by geometry or difficult scenes. |
| Multiple same-class PPE/person cases | 2,144 | Heuristic review observation; gloves can naturally appear more than once. |
| Perceptual near-duplicate candidates | 0 | Confirmed same-source candidates were resolved. |

The EDA treated findings carefully. Small boxes, tiny floating-point boundary overflow, unassigned PPE under a strict center-point rule, and multiple same-class PPE per person were documented as limitations or review heuristics rather than automatic dataset errors. This avoided unnecessary cleaning based only on statistical or geometric suspicion.

After confirmed Data Processing, the final EDA readiness verdict was **READY FOR TRAINING**.

## 6. Training Methodology

All candidate models used the same dataset and shared training configuration from `config/training_config.py`. This was important because the goal was a fair model comparison rather than independent tuning of each candidate.

| Setting | Value |
|---|---:|
| Image size | 640 |
| Epochs | 100 |
| Early-stopping patience | 15 |
| Batch size | 16 |
| Optimizer | `auto` |
| Seed | 42 |
| Device | 0 |
| Workers | 8 |
| Cache | `False` |
| Deterministic | `True` |
| Pretrained | `True` |
| AMP | `True` |

Augmentation settings:

| Augmentation | Value |
|---|---:|
| hsv_h | 0.015 |
| hsv_s | 0.5 |
| hsv_v | 0.3 |
| degrees | 5.0 |
| translate | 0.05 |
| scale | 0.2 |
| fliplr | 0.5 |
| flipud | 0.0 |
| mosaic | 0.5 |
| close_mosaic | 10 |
| mixup | 0.0 |

Early stopping allowed a model to stop before 100 epochs if validation performance stopped improving within the configured patience window. The optimizer was left as Ultralytics `auto` for all candidates, preserving a consistent training setting across the comparison.

## 7. Model Comparison

Six trained candidates were compared on the validation split:

- YOLO11n
- YOLO11s
- YOLO11m
- YOLO26n
- YOLO26s
- YOLO26m

Saved training-run validation metrics from the final model-comparison notebook:

| Model | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 82 | 0.85241 | 0.77629 | 0.82265 | 0.56706 |
| YOLO11s | 51 | 0.86664 | 0.77243 | 0.83943 | 0.57766 |
| YOLO11m | 72 | 0.89371 | 0.77279 | 0.83550 | 0.58267 |
| YOLO26n | 78 | 0.87585 | 0.75718 | 0.81667 | 0.56716 |
| YOLO26s | 64 | 0.89178 | 0.77941 | 0.84257 | 0.58735 |
| YOLO26m | 72 | 0.90803 | 0.78246 | 0.84564 | 0.58851 |

Per-class validation mAP50-95:

| Model | person | helmet | vest | gloves | goggles |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0.73038 | 0.77911 | 0.56767 | 0.31683 | 0.43876 |
| YOLO11s | 0.72852 | 0.78983 | 0.58022 | 0.35629 | 0.43627 |
| YOLO11m | 0.74119 | 0.78650 | 0.57580 | 0.34976 | 0.45698 |
| YOLO26n | 0.72139 | 0.78005 | 0.57437 | 0.34016 | 0.41468 |
| YOLO26s | 0.72779 | 0.78875 | 0.58335 | 0.37364 | 0.46351 |
| YOLO26m | 0.73730 | 0.79110 | 0.58275 | 0.37554 | 0.45149 |

Model complexity evidence:

| Model | Checkpoint size | Parameters | GFLOPs | Training time |
|---|---:|---:|---:|---:|
| YOLO11n | 5.21 MB | 2.591M | 6.44 | 1:00:03 |
| YOLO11s | 18.28 MB | 9.430M | 21.56 | 0:43:33 |
| YOLO11m | 38.63 MB | 20.057M | 68.21 | 1:47:13 |
| YOLO26n | 5.13 MB | 2.506M | 5.78 | 1:08:55 |
| YOLO26s | 19.36 MB | 9.952M | 22.52 | 1:05:28 |
| YOLO26m | 41.98 MB | 21.781M | 74.75 | 2:04:07 |

Comparable per-model validation inference time was not preserved as a final table. Efficiency was therefore judged from the preserved checkpoint size, parameter count, GFLOPs, training time, and the selected model's final measured inference speed.

The model-comparison notebook records YOLO26m as the absolute validation mAP50-95 winner, with the final validation-only selection note showing only a `0.00116` advantage over YOLO26s. The saved training-run table above also shows the same practical conclusion: the two models were extremely close. YOLO26s was slightly stronger on vest and goggles mAP50-95, while YOLO26m was slightly stronger on person, helmet, and gloves. Because YOLO26s retained nearly identical detection quality while being less than half the checkpoint size and roughly one third of the GFLOPs of YOLO26m, YOLO26s was selected as the final model.

## 8. Final Model Evaluation

The selected YOLO26s model was evaluated once on the held-out test split after model selection.

| Item | Value |
|---|---:|
| Model | YOLO26s |
| Test images | 323 |
| Test instances | 2,523 |
| Image size | 640 |

Overall held-out test metrics:

| Metric | Value |
|---|---:|
| Precision | 0.8846 |
| Recall | 0.7708 |
| mAP50 | 0.8371 |
| mAP50-95 | 0.5833 |

Per-class test metrics:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.943 | 0.904 | 0.931 | 0.710 |
| helmet | 0.943 | 0.927 | 0.960 | 0.782 |
| vest | 0.908 | 0.819 | 0.885 | 0.604 |
| gloves | 0.792 | 0.490 | 0.610 | 0.354 |
| goggles | 0.837 | 0.714 | 0.799 | 0.466 |

The strongest classes were helmet and person. Vest was also solid. Gloves were the weakest class, especially recall, which indicates that missed glove detections remain the clearest model-level limitation. Goggles were moderate and remain difficult because eyewear is small, often partially occluded, and visually close to facial regions.

Measured YOLO26s speed on the final test evaluation was:

| Stage | Speed |
|---|---:|
| Preprocess | 1.3 ms/image |
| Inference | 4.6 ms/image |
| Postprocess | 0.2 ms/image |

The test performance was close to the validation performance used for model selection, supporting that YOLO26s generalized consistently across the project splits.

## 9. Compliance System Architecture

The compliance layer converts filtered object detections into worker-level decisions. Its behavior is documented in detail in `docs/compliance_logic.md`.

Current application flow:

| Stage | Purpose |
|---|---|
| YOLO inference | Detect person and PPE boxes using the final model. |
| Person duplicate suppression | Remove near-identical duplicate person boxes at IoU `0.95`, keeping the higher-confidence person. |
| PPE-specific NMS | Apply class-aware suppression to helmet, vest, gloves, and goggles only. |
| Goggles verification | Reject low-confidence goggles and verify borderline goggles with an upper-person crop. |
| PPE-to-person association | Assign PPE by center containment inside person boxes. |
| Glove-specific handling | Assign exclusive gloves first, resolve ambiguous gloves in overlapping workers, and recover only genuinely unassigned gloves. |
| Worker compliance | Check each detected worker against the selected required PPE set. |
| Image-level status | Combine worker statuses into `COMPLIANT`, `ATTENTION`, or `UNKNOWN`. |
| Visualization | Draw filtered detections, worker labels, and global status on the saved output image. |

Key parameters:

| Rule | Current value |
|---|---:|
| General inference confidence | 0.25 |
| Person duplicate IoU | 0.95 |
| Helmet NMS IoU | 0.50 |
| Vest NMS IoU | 0.50 |
| Gloves NMS IoU | 0.15 |
| Goggles NMS IoU | 0.30 |
| Goggles candidate threshold | 0.40 |
| Goggles direct-accept threshold | 0.75 |
| Goggles crop-confirm threshold | 0.40 |
| Glove recovery expansion ratio | 0.15 |
| Glove recovery minimum overlap ratio | 0.25 |

Status semantics:

| Status | Meaning |
|---|---|
| `COMPLIANT` | All required PPE was detected and assigned for every detected worker. |
| `ATTENTION` | At least one required PPE item was not detected or assigned for at least one detected worker. |
| `UNKNOWN` | No person was detected, so compliance could not be evaluated. |

The meaning of `ATTENTION` is intentionally conservative. It means the system did not detect the required PPE for a worker. It does not prove that the worker physically lacks that PPE.

## 10. End-to-End Application Evaluation

The final application-level test used 160 images from `data/final_inference_images/`, grouped into 8 PPE requirement folders. Each requirement group contained expected `COMPLIANT` and expected `ATTENTION` images. Required PPE was manually selected during each batch run; the system did not infer PPE requirements from folder names.

Overall results:

| Metric | Result |
|---|---:|
| Total images | 160 |
| Expected COMPLIANT | 80 |
| Expected ATTENTION | 80 |
| Predicted COMPLIANT | 71 |
| Predicted ATTENTION | 89 |
| Predicted UNKNOWN | 0 |
| Correct | 151 |
| Accuracy | 94.38% |
| COMPLIANT recall | 88.75% |
| ATTENTION recall | 100% |
| False COMPLIANT | 0 |
| False ATTENTION | 9 |

Per-requirement results:

| Requirement group | Correct | Accuracy |
|---|---:|---:|
| required_all_ppe | 18 / 20 | 90% |
| required_gloves | 20 / 20 | 100% |
| required_goggles | 20 / 20 | 100% |
| required_helmet | 19 / 20 | 95% |
| required_helmet_vest | 18 / 20 | 90% |
| required_helmet_vest_gloves | 18 / 20 | 90% |
| required_helmet_vest_goggles | 18 / 20 | 90% |
| required_vest | 20 / 20 | 100% |

The more complex requirement groups were more error-sensitive because one missed or unassigned required item can change the full image status. The key safety-oriented result is that no expected `ATTENTION` image was incorrectly classified as `COMPLIANT` in this final test set. This should be interpreted as evidence of conservative behavior on the curated final test images, not as a guarantee for uncontrolled real-world deployment.

## 11. Failure Analysis

All 9 image-level errors in the final application test were false `ATTENTION` results. In other words, the system flagged some expected compliant scenes for review, but it did not incorrectly approve an expected attention scene.

The main observed patterns were:

| Pattern | Effect |
|---|---|
| Missed or unassigned helmets | A visibly compliant worker could be marked `ATTENTION` when a helmet was not detected or not assigned. |
| Missed or unassigned goggles | Small eyewear detections remained difficult, especially in multi-worker scenes. |
| Missed or unassigned gloves | Glove recall was the weakest detector metric, and hand-level PPE is small. |
| False or small person detections | Extra artificial workers could be created, producing missing-PPE status for a non-worker region. |
| Overlapping workers | PPE ownership could become ambiguous even after glove-specific assignment improvements. |

These were treated as legitimate end-to-end system errors, not as bugs to hide through post-hoc tuning. The final inference pipeline was frozen after the documented inference-stage refinements rather than being tuned to individual final-test images.

Qualitatively, the failures show the expected relationship between detector quality and application behavior. A missed helmet can make a compliant worker appear non-compliant. A false small person can create an artificial worker missing required PPE. A small glove or goggles detection may be present visually but missed or assigned incorrectly. These issues are part of the deployed decision chain and therefore counted in application-level results.

## 12. Engineering Decisions

Several project decisions shaped the final system:

| Decision | Rationale |
|---|---|
| Build a custom curated dataset | Public PPE sources were too inconsistent to use directly. |
| Keep a five-class schema | The final application needed person, helmet, vest, gloves, and goggles only. |
| Repair only confirmed dataset problems | Avoided over-cleaning based on heuristics or statistical suspicion. |
| Preserve train/valid/test split after final cleanup | Avoided reshaping evaluation conditions late in the project. |
| Use one shared training configuration | Made the six-model comparison fair. |
| Select YOLO26s instead of only the highest raw mAP model | Balanced detection quality, checkpoint size, and compute cost. |
| Use person-centric compliance | Prevented image-level PPE presence from incorrectly satisfying every worker. |
| Keep conservative status semantics | `ATTENTION` prompts review when required PPE is not detected. |
| Avoid final-test-specific tuning | Preserved the integrity of the final application evaluation. |

These decisions reflect the project goal: not just to train a detector, but to build an inspectable computer vision system with explicit data, model, and application tradeoffs.

## 13. Limitations

The final project has several important limitations:

- Small PPE detection remains difficult.
- Gloves are the weakest class, with test recall of 0.490 and mAP50-95 of 0.354.
- Goggles can be confused with facial or eye regions.
- Overlapping workers complicate PPE assignment.
- False person detections can create extra workers and false `ATTENTION` results.
- Fixed-image testing does not equal real deployment.
- Synthetic data may not fully represent all real industrial domains.
- The application-level evaluation set contains 160 curated final inference images.
- Results do not constitute safety certification.

The system should be interpreted as a computer-vision support tool. It can highlight likely compliance concerns, but final safety responsibility cannot be delegated to the model.

## 14. Reproducibility

Verified environment:

| Component | Version |
|---|---|
| Python | 3.11.9 |
| Ultralytics | 8.4.96 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| CUDA runtime used by PyTorch | 12.1 |

Exact dependency pins are stored in `requirements.txt`. The CUDA-specific PyTorch and Torchvision builds require the appropriate PyTorch wheel index during installation, as documented in `README.md`.

Training reproducibility controls included:

| Control | Value |
|---|---:|
| Seed | 42 |
| Deterministic training | `True` |
| Shared training configuration | Yes |
| Final model path | `models/best.pt` |

The project notebooks under `notebooks/` contain the final evidence trail for dataset creation, EDA, Data Processing, training configuration, model comparison, final model evaluation, and final inference checking. GPU training can still have implementation-level nondeterminism, so the project documents the exact environment and configuration rather than claiming perfect bit-for-bit reproducibility.

## 15. Conclusion

Smart Factory Safety Inspector demonstrates a complete computer vision workflow from dataset engineering through application-level evaluation. The dataset work was a major contribution: multiple raw pools were standardized, manually reviewed, supplemented, audited for duplicate/source leakage, and cleaned only where problems were confirmed.

Six YOLO models were trained and compared under a shared configuration. YOLO26s was selected as a quality/efficiency compromise: YOLO26m had a marginally higher validation mAP50-95, but YOLO26s provided nearly identical detection quality with substantially lower model size and compute cost.

The selected YOLO26s model achieved **0.5833 mAP50-95** on the held-out test split. Person and helmet performance were strongest, while gloves and goggles remained the most challenging PPE categories. The final end-to-end compliance system achieved **94.38% application-level accuracy** on the 160-image final inference check, with **0 false COMPLIANT cases**.

The remaining weaknesses are concentrated around small PPE detection, goggles ambiguity, overlapping workers, and occasional false person detections. These are meaningful limitations, but they are also clearly documented and reflected in the final evaluation rather than hidden by late tuning.
