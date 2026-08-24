# CLAUDE.MD — Solar Filament Segmentation Challenge 2026

> **Role:** You are an autonomous ML research agent working on the IEEE BigData Cup 2026 Solar Filament Segmentation Challenge hosted on Kaggle. Your objective is to develop, iterate, and submit high-quality instance segmentation solutions for solar filaments. You operate in a self-contained experiment loop: build → evaluate → submit → document → handoff.

---

## 1. COMPETITION CONTEXT

### 1.1 What is this competition?

The Solar Filament Segmentation Challenge 2026 asks participants to produce pixel-precise instance segmentation masks for solar filaments observed in H-alpha images from the GONG (Global Oscillations Network Group) network. It is hosted on Kaggle and sponsored by the NSF National Solar Observatory.

- **Kaggle URL:** `https://kaggle.com/competitions/filament-segmentation-2026`
- **Self-evaluation notebook:** `https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook`
- **Competition launched:** July 10, 2026
- **Submission deadline:** November 15, 2026
- **Winner announcement:** November 30, 2026
- **IEEE BigData Conference:** December 14–17, 2026, Phoenix, AZ

### 1.2 What are solar filaments?

Solar filaments are dense clouds of solar material suspended above the photosphere by magnetic field lines. They appear as elongated dark structures in H-alpha images of the Sun. They are scientifically important because they are at the core of Coronal Mass Ejections (CMEs), solar flares, and Solar Energetic Particle storms — events that can damage power grids, disrupt GPS, and endanger astronauts.

Key morphological features:
- **Barbs:** Thin, thread-like extensions from the main filament body. Their orientation relative to the filament spine encodes magnetic field configuration (chirality). Capturing barbs accurately is both scientifically valuable and technically challenging.
- **Spines:** The central axis running along the filament's length.
- **Variable scale:** Filaments range from small active-region filaments (<10⁴ km) to massive quiescent filaments spanning hundreds of Mm.

### 1.3 The MAGFiLO dataset

MAGFiLO (Manually Annotated GONG Filaments from H-Alpha Observations) is the ground-truth dataset. Reference: `doi:10.1038/s41597-024-03876-y`.

- **Total:** 10,244 annotated filaments from 1,593 GONG observations (2011–2022)
- **Annotation method:** >1,000 person-hours of manual annotation with double-blind review
- **Inter-annotator agreement:** Kappa = 0.66 (expert-vs-expert Dice ≈ 0.69, median 0.73)
- **Format:** COCO-style JSON with polygonal segmentations, bounding boxes, spines, chirality labels
- **Categories:** Left (id=1), Right (id=2), Unidentifiable (id=3), Ambiguous (id=4) — chirality labels. This competition focuses only on segmentation masks, not classification.

**Competition data split:**
- `train/train_images/` — training H-alpha JPEG images (2048×2048, grayscale, 8-bit)
- `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` — COCO-format training annotations
- `test/test_images/` — 180 test images for submission
- 888 files total, ~751 MB

**Critical data notes:**
- Images are grayscale. Do NOT process as RGB.
- Each image filename encodes: `YYYYMMDDHHMMSSII.jpeg` where II is the instrument code (e.g., Bh = Big Bear, Lh = Learmonth, Mh = Mauna Loa, Th = El Teide, Ch = Cerro Tololo, Uh = Udaipur).
- Multiple annotators may have independently annotated the same physical image. Different `image_id` values like `010101-20160920230134Lh` and `010402-20160920230134Lh` refer to the same observation annotated by different people. Treat them as separate training samples.
- 296 of 707 training observations carry 2–3 independent annotation batches.
- **DO NOT** use the publicly available MAGFiLO v1.0 from Harvard Dataverse for training labels beyond what the competition provides. The test set images overlap with that public release, and using those external annotations is grounds for disqualification.
- You MAY use external H-alpha images (e.g., additional GONG data without MAGFiLO labels) for self-supervised pretraining, as long as no external ground-truth annotations for test images are used.

### 1.4 Evaluation metric: Panoptic Quality (PQ)

The leaderboard metric is **Panoptic Quality** (Kirillov et al., CVPR 2019: `doi:10.1109/CVPR.2019.00963`).

```
PQ = (Σ_{(p,g)∈TP} IoU(p,g)) / (|TP| + 0.5·|FP| + 0.5·|FN|)
```

Equivalently: PQ = SQ × RQ, where:
- **SQ (Segmentation Quality)** = mean IoU over matched pairs = Σ IoU / |TP|
- **RQ (Recognition Quality)** = |TP| / (|TP| + 0.5·|FP| + 0.5·|FN|)

A predicted segment and a ground-truth segment are matched (TP) if their IoU > 0.5. Each ground-truth segment matches at most one prediction and vice versa (bipartite matching by overlap).

**What this means practically:**
- Missing a filament (FN) hurts you. An empty submission scores exactly 0.
- Predicting a spurious filament (FP) also hurts you.
- PQ rewards both finding all filaments AND segmenting them precisely.
- Unlike the old Dice-based metric (which rewarded predicting fewer filaments), PQ has no exploit from withholding predictions.

**Calibration benchmark:** Expert-vs-expert Dice ≈ 0.69. Any PQ > 0.3 is "of great value" per the organizers. The current top honest scores on the leaderboard are ~0.40–0.53.

### 1.5 Submission format

Submit a single CSV file with columns `filament_id` and `segmentation_rle`.

```
filament_id,segmentation_rle
20150125172714Mh_1,<RLE counts string>
20150125172714Mh_2,<RLE counts string>
...
```

- `filament_id`: `<image_stem>_<N>` where N is a unique suffix per image.
- `segmentation_rle`: COCO RLE counts string encoding a binary mask. Use `pycocotools` (`mask.encode`, `mask.decode`). The mask size is always 2048×2048.
- **CRITICAL:** Masks must be pixel-disjoint per image. No two predicted filaments in the same image may share a pixel. The scorer rejects overlapping predictions. If your model produces overlapping masks (e.g., Mask R-CNN), you must resolve overlaps before submission (see Section 3.6).
- Do NOT include quotation marks around the RLE string.
- Store RLE Counts only; Size is fixed at 2048×2048.

### 1.6 Final evaluation rubric (beyond leaderboard)

The competition is not won on leaderboard PQ alone. Final scoring:

**Quantitative (70%):**
- Panoptic Quality score (the leaderboard metric)
- Distribution of Dice scores across images
- Distribution of IoU scores across instances
- Distribution of one-to-many and many-to-one correspondences

**Qualitative (30%):**
- Detailed description of the entire pipeline (4-page technical report, Overleaf template provided)
- Apparent morphology of predicted segmentations on H-alpha images
- Code quality: modularity, documentation, reproducibility

This means your code should be clean, well-documented, and organized into reusable modules — not a single monolithic notebook.

### 1.7 Key constraints and rules

- AI tools are permitted for coding and documentation, but you must understand every component.
- Only H-alpha images from the test directory may be used at inference time.
- No external ground-truth metadata for training beyond what the competition provides.
- Auxiliary annotations from the provided dataset (spine, bbox, area, category_id) ARE allowed for training supervision.
- Derived targets from segmentation masks (skeletons, boundaries, distance transforms) ARE allowed.

---

## 2. DOMAIN-SPECIFIC KNOWLEDGE

### 2.1 Image characteristics

- **Modality:** H-alpha (656.3 nm) full-disk solar images
- **Resolution:** 2048×2048 pixels
- **Format:** 8-bit grayscale JPEG (converted from FITS)
- **Content:** Full solar disk against dark background. Filaments appear as dark, elongated structures against the brighter chromospheric disk.
- **Noise sources:** Ground-based telescopes → atmospheric seeing, clouds, limb darkening, scattered light, variable exposure. Image quality varies significantly across the dataset.
- **Solar disk:** The disk does not fill the entire image; there is a dark border region outside the solar limb.

### 2.2 Why filament segmentation is hard

1. **Extreme class imbalance:** Filaments occupy a tiny fraction of total pixels. The vast majority of pixels are background (solar disk or off-limb dark).
2. **Low contrast:** Filaments are dark against a noisy, textured chromospheric background. Sunspots and other dark features can look similar.
3. **Fine structures:** Barbs are often only a few pixels wide and can be discontinuous.
4. **Fragmentation vs. continuity:** A single physical filament may appear as several disconnected dark patches. The ground truth treats it as one object. Algorithms tend to over-fragment.
5. **Variable image quality:** Some images have clouds, poor seeing, or artifacts that obscure filaments.
6. **Scale variation:** Filaments range from small active-region structures to massive quiescent filaments spanning a large fraction of the disk.

### 2.3 Preprocessing pipeline (established best practices)

From the EdgeAttNet paper (arXiv:2509.02964) and competition discussion:
1. **Normalization:** Rescale pixel intensities.
2. **Solar disk masking:** Use Hough Circle Transform to detect the solar disk boundary. Mask out off-limb pixels.
3. **Radial flattening / limb-darkening correction:** The solar disk is brighter at center and dimmer at edges. Flatten this radial gradient to make filament contrast uniform across the disk.
4. **Gaussian smoothing:** σ=0.7, 3×3 kernel to reduce noise.
5. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhance local contrast to make filaments more visible.

### 2.4 Known architectures that have been tried on solar filaments

- **U-Net and variants:** The standard baseline. U-Net with ASPP (Atrous Spatial Pyramid Pooling) has been used.
- **U-Net++ (nested U-Net):** A public baseline notebook exists on the competition page.
- **EdgeAttNet:** U-Net backbone with learnable edge maps injected into self-attention at the bottleneck. Designed specifically for barb capture on MAGFiLO. (arXiv:2509.02964)
- **MORDEN (Multiscale ORiented DENdritic model):** Modern ConvNet for solar filament semantic segmentation with DenseCRF and DBSCAN post-processing. (arXiv:2607.24525)
- **Mask R-CNN:** Instance segmentation. Works but produces overlapping masks that must be resolved. A participant reported PQ=0.31 with score-ordered panoptic painting.
- **DETR + U-Net pipeline:** Detection with DETR, then instance segmentation with U-Net per detected region. (Reche & Cid 2024)
- **CGAN-based methods:** Conditional GANs for filament segmentation, reportedly better on low-quality images.

### 2.5 The semantic-to-instance gap

This competition requires **instance** segmentation (separate each filament as its own mask), not just **semantic** segmentation (binary: filament vs. background). Most prior solar filament work was semantic only. You must bridge this gap.

Common approaches:
- **Two-stage:** Semantic segmentation → connected-component analysis or watershed → individual instances. Risk: over-fragmentation.
- **Instance segmentation models:** Mask R-CNN, Cascade Mask R-CNN, SOLOv2, Mask2Former. These directly predict instances but may produce overlapping masks.
- **Panoptic segmentation models:** Mask2Former in panoptic mode, Panoptic FPN. These produce non-overlapping predictions by design.
- **Embedding-based:** Predict per-pixel embeddings, then cluster into instances.

---

## 3. ENGINEERING SPECIFICATIONS

### 3.1 Required packages

```
torch, torchvision
pycocotools          # COCO API: RLE encoding/decoding, annotation loading
torchmetrics         # For local Dice/IoU evaluation
albumentations       # Image augmentation
opencv-python        # Image processing, Hough circles, CLAHE
numpy, pandas        # Data manipulation
matplotlib           # Visualization
scikit-learn         # Train/val splitting
segmentation-models-pytorch  # Pretrained encoder-decoder architectures
detectron2           # (optional) Mask R-CNN, Cascade R-CNN
mmdet / mmengine     # (optional) alternative detection framework
kaggle               # Kaggle API for submissions
```

### 3.2 Data loading pattern

```python
from pycocotools.coco import COCO
import numpy as np
import cv2

coco = COCO('path/to/MAGFiLO_1.0_Annotations_kaggle2026_train.json')

# Get all image records
img_ids = coco.getImgIds()

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']  # e.g., "20160920230134Lh.jpeg"
    
    # Load image as grayscale
    image = cv2.imread(f'train/train_images/{file_name}', cv2.IMREAD_GRAYSCALE)
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    for ann in anns:
        # Each ann has: segmentation (polygon), bbox, area, spine, category_id, iscrowd
        mask = coco.annToMask(ann)  # Binary mask, shape (2048, 2048)
```

### 3.3 Submission encoding pattern

```python
import pycocotools.mask as mask_util
import numpy as np
import pandas as pd

rows = []
for image_stem, masks in predictions.items():
    for idx, binary_mask in enumerate(masks, start=1):
        # binary_mask: np.ndarray, dtype=uint8, shape=(2048, 2048)
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        rle_counts = rle['counts'].decode('utf-8')
        rows.append({
            'filament_id': f'{image_stem}_{idx}',
            'segmentation_rle': rle_counts
        })

df = pd.DataFrame(rows)
df.to_csv('submission.csv', index=False)
```

### 3.4 Local evaluation (PQ)

Implement PQ locally so you can evaluate without burning Kaggle submission slots.

```python
def compute_pq(pred_masks, gt_masks, iou_threshold=0.5):
    """
    pred_masks: list of binary np.arrays (one per predicted instance)
    gt_masks:   list of binary np.arrays (one per ground-truth instance)
    Returns: PQ, SQ, RQ, TP, FP, FN
    """
    matched_pred = set()
    matched_gt = set()
    iou_sum = 0.0

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            intersection = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    # Greedy matching by descending IoU
    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]
        if max_iou < iou_threshold:
            break
        i, j = max_idx
        matched_pred.add(i)
        matched_gt.add(j)
        iou_sum += max_iou
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    tp = len(matched_pred)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp

    sq = iou_sum / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return pq, sq, rq, tp, fp, fn
```

Also use the official self-evaluation notebook: `https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook`

### 3.5 Train/validation split strategy

Since the same physical image can have multiple annotator IDs, you must split by **physical image** (the filename stem without the annotator prefix), not by `image_id`. Otherwise you leak information between train and validation.

```python
# Extract unique physical images
physical_images = set()
for img_info in coco.loadImgs(coco.getImgIds()):
    # image_id is like "010101-20160920230134Lh"
    # file_name is like "20160920230134Lh.jpeg"
    physical_images.add(img_info['file_name'])

# Split physical images, then assign all annotator versions of each image to the same fold
```

### 3.6 Overlap resolution (panoptic painting)

If your model produces overlapping instance masks (e.g., Mask R-CNN), you MUST resolve them before submission. Use score-ordered panoptic painting:

```python
def paint_panoptic(candidates, min_area=50, H=2048, W=2048):
    """
    candidates: list of (score, binary_mask) tuples
    Returns: list of (score, non_overlapping_mask, area) — guaranteed disjoint.
    """
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary * (1 - claimed)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed = np.bitwise_or(claimed, remaining)
        kept.append((score, remaining, area))
    return kept
```

### 3.7 Overlap validator (run before every submission)

```python
def check_no_overlap(csv_path, H=2048, W=2048):
    import pycocotools.mask as mu
    df = pd.read_csv(csv_path)
    df['stem'] = df['filament_id'].str.rsplit('_', n=1).str[0]
    for stem, group in df.groupby('stem'):
        if len(group) < 2:
            continue
        acc = np.zeros((H, W), dtype=np.int16)
        for r in group['segmentation_rle']:
            acc += mu.decode({'size': [H, W], 'counts': r.encode('utf-8')}).astype(np.int16)
        if (acc > 1).any():
            raise ValueError(f"OVERLAP in {stem}")
    print("PASS: no overlaps")
```

---

## 4. AGENTIC WORKFLOW PROTOCOL

You operate in a loop of self-contained experiments. Every experiment lives in its own directory under `experiments/`. The loop is:

```
[EXPERIMENT N] → build & train → evaluate locally → submit to Kaggle → log results
    → write HANDOFF.md for Experiment N+1
    → (if N==1) also produce RESEARCH_BRIEF.md for the Research Agent
```

### 4.1 Directory structure

```
solar-filament-segmentation/
├── CLAUDE.MD                          ← this file (master context)
├── context/
│   └── competition-feed.txt           ← raw competition page + discussions
├── experiments/
│   ├── exp_001_baseline/
│   │   ├── README.md                  ← what this experiment does & results
│   │   ├── config.yaml                ← hyperparameters
│   │   ├── src/
│   │   │   ├── dataset.py
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   ├── postprocess.py
│   │   │   └── evaluate.py
│   │   ├── logs/
│   │   │   └── training_log.txt
│   │   ├── outputs/
│   │   │   └── submission.csv
│   │   └── RESULTS.md                 ← final metrics, what worked, what didn't
│   ├── exp_002_<strategy_name>/
│   │   ├── ...
│   │   └── HANDOFF.md                 ← context + instructions for next experiment
│   └── ...
├── shared/
│   ├── utils.py                       ← shared utilities (RLE encoding, PQ, overlap check)
│   ├── preprocessing.py               ← shared preprocessing (disk mask, CLAHE, etc.)
│   └── data_split.py                  ← canonical train/val split
├── research/
│   ├── RESEARCH_BRIEF.md              ← instructions for the Research Agent
│   ├── papers/
│   │   └── <paper_slug>.md            ← summary of each relevant paper found
│   └── SYNTHESIS.md                   ← Research Agent's compiled strategy recommendations
└── best_submission.csv                ← current best submission
```

### 4.2 Experiment lifecycle

Each experiment follows these steps, in order:

**Step 1: Read context.**
- Read this CLAUDE.MD.
- Read the HANDOFF.md from the previous experiment (if one exists).
- Read `research/SYNTHESIS.md` if it exists (the Research Agent's recommendations).

**Step 2: Define the experiment.**
- Write `README.md` in the experiment directory: what hypothesis you are testing, what method you are using, what you expect.
- Write `config.yaml` with all hyperparameters.

**Step 3: Implement.**
- Write modular code in `src/`. Reuse utilities from `shared/`.
- Every script should be runnable independently.
- Include docstrings and comments explaining WHY, not just what.

**Step 4: Train and evaluate locally.**
- Train your model.
- Evaluate on the held-out validation set using the PQ metric (and Dice/IoU for diagnostics).
- Log all metrics to `logs/`.

**Step 5: Generate submission.**
- Run inference on the test set.
- Resolve any mask overlaps (panoptic painting).
- Run the overlap validator.
- Save `submission.csv`.

**Step 6: Submit to Kaggle.**
```bash
kaggle competitions submit \
    -c filament-segmentation-2026 \
    -f outputs/submission.csv \
    -m "exp_NNN: <brief description>"
```

**Step 7: Document results.**
- Write `RESULTS.md`: local PQ, Kaggle PQ, per-image breakdowns, what worked, what failed, visualizations of good/bad predictions.

**Step 8: Write HANDOFF.md.**
- This is the most important document. It is the context that the next agent instance will read. It must contain:
  1. **Summary of what was tried** in this experiment and all previous ones.
  2. **Current best score** (local and Kaggle).
  3. **What worked and what didn't** — specific, actionable observations.
  4. **Detailed instructions for the next experiment(s)** — each strategy gets its own folder name, a description of the method, why it should help, and implementation guidance.
  5. **Open questions** that need investigation.
  6. **List of files** the next agent needs to read.

### 4.3 Experiment 1: Baseline

Your first experiment establishes a baseline. Use the following approach:

**Architecture:** `segmentation_models_pytorch` with a U-Net or U-Net++ backbone (e.g., `efficientnet-b4` encoder, pretrained on ImageNet).

**Strategy:**
1. Semantic segmentation first: train a binary segmentation model (filament vs. background).
2. Post-process with connected-component labeling to produce instance masks.
3. Filter by minimum area.
4. Encode as RLE, validate no overlaps, submit.

**Preprocessing:**
1. Load grayscale.
2. Detect solar disk via Hough Circle Transform.
3. Apply limb-darkening correction (radial intensity normalization).
4. Apply CLAHE.
5. Resize to 512×512 for training (or use patch-based training on the full 2048×2048).

**Training:**
- Loss: combination of BCE + Dice loss (to handle class imbalance).
- Optimizer: AdamW, lr=1e-4, cosine annealing.
- Augmentations: horizontal/vertical flip, rotation, elastic deformation, brightness/contrast jitter.
- Epochs: 50–100, early stopping on validation PQ.

**This baseline should produce a PQ somewhere in the 0.15–0.30 range.** The purpose is to have a working end-to-end pipeline, not to win. Everything after this is iteration.

---

## 5. RESEARCH AGENT PROTOCOL

After Experiment 1, you will produce a `research/RESEARCH_BRIEF.md` file. This file instructs a **second agent instance** (the Research Agent) to conduct a thorough literature search and produce strategy recommendations.

### 5.1 Research Agent instructions (to be included in RESEARCH_BRIEF.md)

The Research Agent's job is to search arxiv, IEEE Xplore, Google Scholar, and other venues for papers relevant to improving solar filament instance segmentation. The search should be both deep and creative — methods from seemingly unrelated domains (medical imaging, remote sensing, materials science, biological cell segmentation) can be highly applicable.

The Research Agent must:

**A. Conduct targeted searches across these categories:**

1. **Solar filament segmentation directly:**
   - Search: `solar filament segmentation`, `solar filament detection deep learning`, `H-alpha filament`, `chromospheric feature segmentation`, `MAGFiLO`
   - Look for: EdgeAttNet, MORDEN, any papers citing MAGFiLO

2. **Instance segmentation for thin/elongated structures:**
   - Search: `thin structure instance segmentation`, `curvilinear structure segmentation`, `filament segmentation` (non-solar: biological filaments, fibers, cracks), `vessel segmentation instance`, `neuron segmentation instance`, `road extraction instance`
   - Rationale: Solar filaments are thin, elongated, branching structures. Methods designed for blood vessels, neurons, roads, or cracks face identical geometric challenges.

3. **Panoptic segmentation architectures:**
   - Search: `Mask2Former`, `OneFormer`, `kMaX-DeepLab`, `panoptic segmentation SOTA 2024 2025 2026`
   - Rationale: These models produce non-overlapping instance masks by design, avoiding the overlap problem.

4. **Small object / fine structure segmentation:**
   - Search: `small object segmentation`, `fine-grained segmentation`, `barb detection`, `hair-like structure segmentation`
   - Rationale: Filament barbs are very small and thin.

5. **Noisy / low-quality image segmentation:**
   - Search: `segmentation under noise`, `robust segmentation noisy images`, `segmentation atmospheric turbulence`, `domain adaptation segmentation`
   - Rationale: GONG images are ground-based and highly variable in quality.

6. **Class-imbalanced segmentation:**
   - Search: `extreme class imbalance segmentation`, `focal loss segmentation`, `boundary-aware loss`, `Lovász loss`, `topological loss segmentation`
   - Rationale: Filament pixels are a tiny minority.

7. **Post-processing and instance separation:**
   - Search: `watershed deep learning segmentation`, `embedding clustering instance segmentation`, `affinity-based instance segmentation`, `Hough voting instance segmentation`, `DenseCRF post-processing`
   - Rationale: Converting semantic segmentation to good instances is a core challenge.

8. **Multi-scale feature extraction:**
   - Search: `feature pyramid network segmentation`, `HRNet segmentation`, `multi-scale attention segmentation`, `ASPP`, `deformable convolutions segmentation`
   - Rationale: Filaments vary enormously in scale.

9. **Self-supervised and semi-supervised methods:**
   - Search: `self-supervised pretraining segmentation`, `MAE pretraining`, `contrastive learning segmentation`, `semi-supervised segmentation`
   - Rationale: Additional unlabeled GONG H-alpha images are allowed for pretraining.

10. **Test-time augmentation and ensembling:**
    - Search: `test-time augmentation segmentation`, `model ensemble segmentation`, `snapshot ensemble`
    - Rationale: Free performance gain at inference time.

11. **Boundary refinement and edge-aware methods:**
    - Search: `boundary-aware segmentation`, `edge-guided segmentation`, `SegFix`, `BPR boundary patch refinement`
    - Rationale: The competition rewards precise filament boundaries.

12. **Domain-specific solar image techniques:**
    - Search: `solar image enhancement`, `limb darkening correction neural`, `solar disk normalization`, `H-alpha image processing`

**B. For each relevant paper, produce a summary file in `research/papers/<slug>.md` with:**

```markdown
# <Paper Title>

- **Authors:** ...
- **Year:** ...
- **Venue:** ...
- **URL:** ...

## Key idea
<1–2 paragraph summary of the core method>

## Why it matters for this competition
<How this method addresses one of the competition's specific challenges>

## Implementation sketch
<Pseudocode or high-level steps to integrate this into the pipeline>

## Expected impact
<Which metric component (SQ, RQ, or both) this would improve, and by roughly how much>

## Dependencies / complexity
<What libraries, compute, or data this requires>
```

**C. Produce `research/SYNTHESIS.md`:**

This is the Research Agent's final deliverable. It should:
1. Rank all discovered strategies by expected impact.
2. Group them into a suggested experiment sequence (each strategy = one experiment folder).
3. Identify which strategies can be combined (e.g., better backbone + better loss + better post-processing are orthogonal and can be stacked).
4. Flag any strategies that are mutually exclusive.
5. Provide a recommended experiment roadmap with priorities.

### 5.2 How to invoke the Research Agent

After Experiment 1, include the following in your HANDOFF.md:

```
## Research Agent Task

Before starting Experiment 2, spin up a Research Agent with the brief at
`research/RESEARCH_BRIEF.md`. The Research Agent should:
1. Read the brief.
2. Conduct all searches described therein.
3. Write paper summaries to `research/papers/`.
4. Write the synthesis to `research/SYNTHESIS.md`.
5. Copy all relevant context files into the experiment directory.

Once the Research Agent has completed its work, proceed with experiments
in the order recommended by SYNTHESIS.md.
```

---

## 6. HANDOFF PROTOCOL

Every HANDOFF.md must follow this template:

```markdown
# HANDOFF — Experiment N → Experiment N+1

## Project state
- **Current best PQ (local):** X.XX
- **Current best PQ (Kaggle):** X.XX
- **Experiments completed:** [list]
- **Total Kaggle submissions used:** N / limit

## What has been tried
<For each past experiment: 1–2 sentences on the method, the result, and the key takeaway>

## What worked
<Specific techniques that improved the score, with numbers>

## What didn't work
<Specific techniques that hurt or didn't help, with numbers>

## Current pipeline summary
<Brief description of the current best pipeline end-to-end>

## Next experiments to run
### Experiment N+1: <name>
- **Folder:** `exp_NNN_<name>/`
- **Method:** <detailed description>
- **Hypothesis:** <why this should help>
- **Implementation notes:** <specific guidance>
- **Expected impact:** <what metric component this targets>

### Experiment N+2: <name>
...

## Research Agent status
- [ ] Research brief written
- [ ] Research completed
- [ ] Synthesis available at `research/SYNTHESIS.md`

## Files to read
- `CLAUDE.MD` (always)
- `experiments/exp_NNN/RESULTS.md`
- `research/SYNTHESIS.md` (if available)
- `shared/utils.py`, `shared/preprocessing.py`

## Open questions
<Things that need investigation or clarification>
```

---

## 7. CODE QUALITY STANDARDS

Since the final evaluation includes code quality (30% of total), maintain these standards throughout:

1. **Modularity:** Every conceptual step is its own module. `dataset.py`, `model.py`, `train.py`, `predict.py`, `postprocess.py`, `evaluate.py` — never one monolithic script.
2. **Documentation:** Every function has a docstring. Every module has a header comment explaining its purpose.
3. **Configuration:** All hyperparameters live in `config.yaml`, never hardcoded.
4. **Reproducibility:** Set random seeds everywhere. Log the exact command used to run each experiment.
5. **Version control:** The final deliverable requires a public Git repo. Structure your code for that from the start.

---

## 8. STRATEGIC NOTES

### 8.1 What the leaderboard tells us

- Top honest PQ scores are ~0.40–0.53 (some inflated scores exist from test-set label leakage).
- A PQ > 0.3 is described as "of great value" by organizers.
- The jump from 5th place (~0.40) to top 4 (~0.53) is large, suggesting the top teams may be using significantly different approaches or have found key tricks.

### 8.2 Low-hanging fruit (try first)

1. Better preprocessing (limb darkening correction, CLAHE tuning).
2. Stronger backbone (ConvNeXt, EfficientNet-B7, Swin Transformer).
3. Better loss function (focal + Dice + boundary loss).
4. Test-time augmentation (flips, rotations, multi-scale).
5. Ensembling multiple models.

### 8.3 Higher-effort, higher-reward strategies

1. Switch from semantic-then-split to a proper panoptic model (Mask2Former).
2. Use auxiliary supervision (spine annotations → skeleton loss).
3. Self-supervised pretraining on unlabeled GONG data.
4. Multi-task learning (segmentation + chirality classification + spine prediction).
5. Boundary refinement networks.

### 8.4 Common failure modes to watch for

- **Over-fragmentation:** Splitting one filament into many pieces. This tanks RQ (increases FP and FN).
- **Under-segmentation:** Merging nearby filaments into one blob. Also tanks RQ.
- **Missing small filaments:** Some filaments are very small. Minimum-area thresholds that are too aggressive will miss them (FN).
- **Predicting sunspots as filaments:** Sunspots are also dark. Without proper filtering, they become FP.
- **Ignoring image quality variation:** Models trained mostly on clean images may fail on noisy ones.

---

## 9. QUICK REFERENCE

| Item | Value |
|---|---|
| Competition | Solar Filament Segmentation Challenge 2026 |
| Platform | Kaggle |
| Metric | Panoptic Quality (PQ) |
| Image size | 2048 × 2048, grayscale, 8-bit JPEG |
| Dataset | MAGFiLO v1.0 (COCO format) |
| Train images | ~700+ (with multiple annotator versions) |
| Test images | 180 |
| Submission | CSV: filament_id, segmentation_rle |
| Deadline | November 15, 2026 |
| Prize pool | $3,000 |
| Key packages | pycocotools, torch, segmentation-models-pytorch |
| Expert baseline PQ | ~0.69 (Dice, expert-vs-expert) |
| Current top PQ | ~0.53 |
| Target PQ | ≥ 0.35 (competitive), ≥ 0.45 (strong) |
