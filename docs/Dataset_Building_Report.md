# Dataset Building Report

This report summarizes how the PPE object-detection dataset for the **Smart Factory Safety Inspector** project was built. The goal was not to simply collect a large number of public images, but to create a reliable dataset for detecting workers and PPE in industrial scenes.

**Final reviewed dataset:** `final_dataset_reviewed_v1`  
**Final result:** **3,882 images**, **3,882 label files**, **23,772 bounding boxes**, **5 classes**, and **0 split leakage**.

This report focuses only on dataset building. Model training, model comparison, inference performance, and application evaluation are documented separately.

## 1. Why a New Dataset Was Needed

Public PPE datasets were useful as raw material, but they were not consistent enough to use directly for this project. The final system needed a five-class worker-safety dataset with reliable person/PPE annotations in realistic scenes. A direct merge would have carried forward class mismatches, annotation errors, duplicates, and unsuitable images.

| Problem Found | Why It Mattered |
|---|---|
| Different class definitions and class names | The model needed one stable schema: `person`, `helmet`, `vest`, `gloves`, `goggles`. |
| Missing person annotations | PPE compliance is worker-centered. PPE boxes are much less useful when the worker box is missing. |
| Missing PPE annotations | A missing glove, vest, or goggles label teaches the model that visible PPE is background. |
| Incorrect boxes and wrong classes | Bad ground truth can be more harmful than a smaller clean dataset. |
| Unnecessary classes | Classes outside the final task add noise and make training less focused. |
| Ordinary shoes/boots mixed with safety footwear | Source data often contained normal footwear or feet-only crops, not reliably identifiable safety boots. Boots were excluded. |
| Helmet-heavy source bias | Helmet examples were common, while vest, gloves, and goggles needed more careful selection. |
| Product/object crops and off-domain images | Some images were not useful worker scenes and could not support the target application. |
| Exact duplicates and source variants | Duplicate images can inflate dataset size and create leakage between splits. |
| Small PPE objects | Gloves and goggles are often small, partially visible, and harder to annotate reliably. |

The key decision was simple: **public datasets were valuable raw sources, but they were not sufficiently consistent or reliable for the final Smart Factory Safety Inspector task without substantial review and correction.**

## 2. Target Dataset Definition

The final dataset uses exactly five classes.

| ID | Class |
|---:|---|
| 0 | person |
| 1 | helmet |
| 2 | vest |
| 3 | gloves |
| 4 | goggles |

Important class-definition decisions:

- `person` is included because the application checks PPE at the worker level.
- `goggles` means clearly visible eyewear under the project definition, including safety goggles, safety glasses, prescription glasses, sunglasses, clear glasses, and tinted glasses when visible enough to annotate.
- Boots were excluded because the raw sources often mixed safety boots with ordinary shoes, sandals, slippers, or feet-only images. The visual evidence was not reliable enough for a clean safety-boot class.
- The dataset remains object-detection focused: each label row is a YOLO bounding box for one visible object.

## 3. Raw Dataset Collection

The Dataset Builder records preserve the source data primarily as a consolidated raw image pool and source-family groups, rather than as clean external dataset titles. For that reason, this report names the audited source families and pools instead of inventing public dataset names that are not present in the frozen records.

| Dataset / Source Area | Relevant Classes | Main Issues / Use |
|---|---|---|
| Consolidated public/raw image pool: `data/Raw_images` | Worker and PPE scenes across the target classes | Very large raw pool; useful coverage, but required duplicate removal, source-variant control, quality checks, and manual filtering. |
| Roboflow-style mixed source families | Helmet, vest, gloves, goggles, person | Many files had `.rf...` naming and mixed provenance. Useful images existed, but source-family audits were required. |
| Worker/construction families such as `hard_hat_workers`, `ppe`, `dataset1`-`dataset6`, `alet_wikimgs_construction` | Mostly worker and PPE scenes | Some families were coherent, while others were broad catch-all or mixed-source groups. |
| Off-domain and off-target families | Mostly rejected | Examples included ordinary face-mask imagery, sandals/slippers, feet-only material, and composite/multi-panel images. |
| Synthetic Data 01 and Synthetic Data 02 | Person, helmet, vest, gloves, goggles | Added controlled PPE combinations and difficult visual cases, then manually reviewed before use. |

Initial raw audit found **361,939 supported images**. It also found **37,485 exact duplicate groups** with **44,903 redundant duplicate files**, which made automatic deduplication and provenance tracking mandatory before any training dataset could be trusted.

## 4. Automated Cleaning and Standardization

Before CVAT review, the dataset was narrowed and standardized through automated checks. These scripts did not replace visual review; they created a safer candidate pool.

Main automated steps included:

- class remapping to the final five-class schema;
- removal of unwanted classes and off-target source families;
- exact duplicate detection by SHA256;
- source-variant control using source identifiers and filename provenance;
- image readability and resolution checks;
- YOLO label validation and image-label pairing checks;
- filename and path-safety validation;
- quality audits for very small, blurry, dark, bright, low-contrast, low-information, and extreme-aspect images;
- family-level review of repetitive, off-domain, shoe/footwear, and composite-image sources;
- preliminary candidate balancing so the review pool was not dominated by a few source families.

The raw pool went from **361,939 supported images** to a much smaller reviewed candidate workflow. Exact duplicates and source variants removed large amounts of repeated data: the selection audit rejected **44,903 exact duplicate files** and **146,316 source variants**.

Auto-annotation was used to accelerate review, not to create final ground truth by itself. Calibration reports showed that gloves and goggles were especially threshold-sensitive, which reinforced the need for human inspection of small PPE.

## 5. Large-Scale Manual Annotation Review in CVAT

**Approximately 13,000 images were manually reviewed in CVAT or connected manual review workflows.** They were not annotated from scratch as a blank dataset. Instead, existing or model-generated YOLO annotations were visually inspected and corrected where necessary.

Manual review included:

- correcting bounding boxes;
- adding missing person annotations;
- adding missing PPE annotations;
- deleting false annotations;
- correcting wrong classes;
- checking partially visible PPE;
- checking small gloves and goggles;
- rejecting images that could not be annotated reliably;
- verifying the visual relationship between workers and PPE.

Automated scripts could catch malformed labels, missing files, duplicate hashes, and invalid coordinates. They could not reliably decide whether a tiny glove was real, whether eyewear was visible enough, whether a worker box was missing, or whether a scene was too ambiguous for training. That is why CVAT review became a central part of the dataset-building process.

Annotation environment:

```text
Windows
  -> WSL / Ubuntu
  -> Docker
  -> CVAT services
  -> Browser-based annotation interface
```

Ubuntu through WSL provided the local Linux environment. Docker ran CVAT and its supporting services in containers, avoiding a fragile manual backend setup. CVAT was then opened in the browser. Candidate datasets were uploaded in manageable batches, visually reviewed, corrected, exported, and validated again by scripts.

Dataset review workflow:

```text
Raw candidate images
        -> Batch creation
        -> CVAT import
        -> Manual visual review
        -> Annotation correction
        -> Reject unsuitable images
        -> CVAT export
        -> Automated post-validation
```

## 6. Review Batches and Dataset Filtering

The final real-image pool came from 11 reviewed batches. Earlier Batch 01-05 were frozen after review; Batch 06-11 were later reviewed and materialized through the same production-style validation workflow.

| Batch | Candidates | Retained | Removed | Notes |
|---|---:|---:|---:|---|
| Batch 01 | 1,000 | 568 | 432 | 474 frames modified during review. |
| Batch 02 | 1,000 | 524 | 476 | 403 frames modified during review. |
| Batch 03 | 1,000 | 455 | 545 | 354 frames modified during review. |
| Batch 04 | 1,000 | 363 | 637 | 305 frames modified during review. |
| Batch 05 | 1,000 | 328 | 672 | 277 frames modified during review. |
| Batch 06 | 845 | 181 | 664 | Reviewed final batch. |
| Batch 07 | 845 | 183 | 662 | Reviewed final batch. |
| Batch 08 | 845 | 212 | 633 | Reviewed final batch. |
| Batch 09 | 845 | 205 | 640 | Reviewed final batch. |
| Batch 10 | 845 | 153 | 692 | Reviewed final batch. |
| Batch 11 | 845 | 104 | 741 | Reviewed final batch. |

This table shows the practical effect of manual quality control. Many candidates were useful as search results or model-assisted proposals, but not good enough to become final training data.

## 7. Synthetic Data

Synthetic images were added to improve coverage of useful PPE combinations and difficult scenes, especially where public data was thin or visually inconsistent. Synthetic data was not accepted automatically. It went through the same review idea: keep useful images, correct labels, and reject bad frames.

| Synthetic Source | Candidates | Retained | Removed | Final Use |
|---|---:|---:|---:|---|
| Synthetic Data 01 | 500 | 336 | 164 | Train only |
| Synthetic Data 02 | 311 | 305 | 6 | Train only |
| Total synthetic reviewed | 811 | 641 | 170 | Train only |

Synthetic records were allowed only in the training split. Validation and test remained real-only so that final metrics would reflect real reviewed images rather than generated imagery.

## 8. Final Dataset Consolidation

After all reviewed real and synthetic sources were materialized, they were consolidated through a final audited pipeline.

```text
Input audit
  -> Global exact duplicate and provenance audit
  -> Exclusion plan
  -> Merge plan and deterministic filenames
  -> Leakage-safe split plan
  -> Physical materialization
  -> Final validation
```

| Consolidation Step | Result |
|---|---:|
| Reviewed input sources | 13 |
| Input records before final exclusions | 3,917 |
| Zero-label records excluded | 35 |
| Exact duplicate groups in final candidate pool | 0 |
| Exact duplicate exclusions | 0 |
| Final retained records | 3,882 |

SHA256 hashes, source manifests, label hashes, and provenance fields were used throughout the consolidation process. This protected the final dataset from accidental file changes, duplicate leakage, stale paths, and incorrect source mappings.

## 9. Leakage-Safe Split

The split was deterministic and leakage-safe. Real reviewed records were split approximately 80/10/10. Synthetic reviewed records were assigned only to train.

| Split | Total | Real | Synthetic |
|---|---:|---:|---:|
| Train | 3,234 | 2,593 | 641 |
| Validation | 324 | 324 | 0 |
| Test | 324 | 324 | 0 |
| Total | 3,882 | 3,241 | 641 |

Quality checks confirmed:

- split-group leakage: **0**;
- image-SHA leakage: **0**;
- synthetic images in validation: **0**;
- synthetic images in test: **0**.

This makes evaluation more trustworthy: validation and test measure performance on real reviewed images only, while synthetic data can still help training.

## 10. Final Dataset Statistics

Final physical dataset size:

| Metric | Count |
|---|---:|
| Images | 3,882 |
| Label files | 3,882 |
| Bounding boxes | 23,772 |
| Classes | 5 |

Class distribution:

| Class | Annotations | Share |
|---|---:|---:|
| person | 7,994 | 33.63% |
| helmet | 4,389 | 18.46% |
| vest | 3,234 | 13.60% |
| gloves | 5,015 | 21.10% |
| goggles | 3,140 | 13.21% |
| Total | 23,772 | 100.00% |

The class mix is intentionally broader than a helmet-only safety dataset. Gloves can naturally appear multiple times per worker, so a higher glove annotation count is expected and not automatically a quality problem.

## 11. Final Quality Assurance

The final materialized dataset passed physical integrity checks after publication and metadata cleanup.

| Check | Result |
|---|---:|
| Records | 3,882 |
| Images | 3,882 |
| Labels | 3,882 |
| Physical release files | 7,765 |
| Missing files | 0 |
| Orphan files | 0 |
| Corrupt images | 0 |
| Malformed labels | 0 |
| Empty labels | 0 |
| SHA mismatches | 0 |
| Leakage issues | 0 |
| Synthetic images in validation | 0 |
| Synthetic images in test | 0 |

The final dataset layout is compatible with Ultralytics YOLO:

```text
final_dataset_reviewed_v1/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
  data.yaml
```

The `data.yaml` file uses the project class mapping and keeps validation under the Ultralytics key `val`, pointing to the physical `valid/images` directory.

## 12. Conclusion

The final PPE dataset was not created by simply merging public datasets. It was created through dataset selection, class standardization, automated cleaning, large-scale CVAT review, manual annotation correction, aggressive rejection of weak images, synthetic supplementation, duplicate and provenance checks, leakage-safe splitting, and final physical validation.

The final reviewed release contains:

- **3,882 carefully selected images**;
- **23,772 reviewed annotations**;
- **5 PPE-related classes**;
- **641 reviewed synthetic images in train only**;
- **real-only validation and test sets**;
- **0 split leakage**.

This dataset is smaller than the raw public pool, but it is much more consistent, inspectable, and aligned with the Smart Factory Safety Inspector task.
