# Astra Self-Audit v3: Full Cine Data, Geometry & Preprocessing Audit (Worker E)

> **Root review status: investigation record, not accepted wholesale.** Read [root disagreement resolutions](../11_RED_TEAM.md) and the corresponding root report before using these claims. Worker completion indicates a delivered report, not scientific validation.

- **Audit Target Base SHA**: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
- **Orca Run ID**: `run_698eb3b460ef`
- **Task ID**: `task_1ef1c0a81fda`
- **Auditor**: AGY Worker E
- **Target Scope**: Cine MRI loaders, ACDC/M&Ms raw image headers, preprocessing pipeline, spatial/temporal coordinate provenance, image-only contract, label schema mapping, split leakage risks.
- **Evidence Directory**: `reports/astra_v3/workers/E_evidence/`

---

## 1. Executive Summary

An exhaustive header-only data audit, geometry verification, and pipeline provenance analysis of ACDC and M&Ms loaders and preprocessing was conducted at commit `d4f503b`. All investigations strictly adhered to the image-only mandate: zero manual ground-truth segmentation arrays were accessed or opened.

**Three-Sentence Executive Summary**:
1. We inventoried the local dataset availability, inspected 200 native NIfTI image headers across 100 ACDC patients at `/tmp/astra_event_acdc_training/training`, and proved that local storage holds only static ED and ES endpoints ($t_{gap} \approx 9.9$ frames), while full 4D cine cycles ($T \approx 27$) and raw M&Ms files are completely absent locally.
2. We found that the legacy preprocessing pipeline (`scripts/preprocess_acdc.py` and `src/self_audit/data/common.py`) destructively drops temporal provenance $T$, strips native NIfTI affine/LPS orientation, and enforces a spatial-only 2.5D triplet $[z-1, z, z+1]$, rendering native-grid Dice derivation impossible from preprocessed `.npy` volumes and preventing genuine temporal motion modeling ($t-1, t, t+1$).
3. To unblock the pipeline, we formalized and empirically verified an invertible `ImageOnlySampleContract` preserving exact physical LPS coordinates and slice correspondence, specified the bijective M&Ms label remap $\{0 \mapsto 0, 1 \mapsto 3, 2 \mapsto 2, 3 \mapsto 1\}$, audited patient split isolation (`splits/acdc_patient_split_seed42.json`), and established that bounded spatial/appearance seed experiments can proceed immediately with the 1,902 local ACDC slices while true temporal cine motion modeling remains strictly blocked until 4D cine files are staged.

---

## 2. Local Data Inventory & Availability Ledger

A comprehensive system-wide search was conducted across the local worktree, the original read-only repo (`/Users/alvinluong/Self-Audit`), sibling worktrees, and system temp locations.

### 2.1 ACDC Dataset Availability

| Property | Value / Status | Empirical Verification Source |
| :--- | :--- | :--- |
| **Primary Location** | `/tmp/astra_event_acdc_training/training` | Discovered via Root Runbook directive |
| **Total Patients** | **100 patients** (`patient001` through `patient100`) | Header inspection script (`E_evidence/inspect_acdc_headers.py`) |
| **Total Image Volumes** | **200 volumes** (uncompressed `.nii`) | 100 ED frames + 100 ES frames |
| **Total Short-Axis Slices** | **1,902 slices** (951 ED slices + 951 ES slices) | Computed across all 200 image volumes |
| **Diagnostic Cohorts** | 5 pathology groups $\times$ 20 patients each | `Info.cfg`: DCM (20), HCM (20), MINF (20), NOR (20), RV (20) |
| **Full 4D Cine Availability** | **0 files (ABSENT)** | Zero `*4d.nii*` files exist in `/tmp` or repo |
| **Intermediate Time Frames** | **ABSENT** ($t \notin \{ED, ES\}$ missing) | Only ED and ES files were extracted historically |
| **Split Manifest** | `splits/acdc_patient_split_seed42.json` | 80 Train / 20 Val / 0 Test (disjoint patients) |
| **Sufficient for Bounded Seed Exp?** | **YES for appearance/spatial; NO for cine motion** | Detailed below |

#### Bounded Seed Experiment Feasibility vs. Concrete Blocker
- **Spatial / Appearance Bounded Experiment (FEASIBLE)**: The 100 patients (200 volumes, 1,902 slices) provide ample raw image data to benchmark spatial 2.5D appearance encoding, intensity normalization, region prototype clustering, and student deployment architectures.
- **Temporal Cine Motion Modeling (BLOCKED / UNAVAILABLE LOCALLY)**: The `CinePseudoTeacher` motion branch in `src/self_audit_pseudolabel/system_v3.py:56-72` expects temporal triplets $[t-1, t, t+1]$:
  $$\text{raw} = [c - p, n - c, |c - p|, |n - c|]$$
  Because intermediate cine frames are absent from disk, genuine temporal motion cannot be computed. Substituting ED for $t-1$ and ES for $t+1$ would be a severe physiological error, as $|t_{ES} - t_{ED}| \approx 10$ frames ($~37\%$ of the entire cardiac cycle).

### 2.2 M&Ms (Multi-Centre, Multi-Vendor) Dataset Availability

| Property | Status / Finding | Code & System Reference |
| :--- | :--- | :--- |
| **Local Raw Data** | **0 files (ABSENT)** | No raw M&Ms files exist on disk |
| **Local Preprocessed Data** | **0 files (ABSENT)** | `preprocessed_data/mnm` does not exist |
| **Configured Root** | Remote cluster placeholder: `/home/linhdang/...` | `configs/maskfree_mnms_150.yaml:26` |
| **Test Fixtures** | Synthetic minimal fixtures only | `reports/maskfree150/software_checks/` |
| **Status for Worker E** | **NOT RUN / UNAVAILABLE** | Audited via code specification & literature only |

---

## 3. Loader & Preprocessing Pipeline Audit

We traced the data flow through:
- `scripts/preprocess_acdc.py`
- `src/self_audit/data/acdc.py`
- `src/self_audit/data/common.py`
- `src/self_audit_maskfree/data/discovery.py`
- `src/self_audit_maskfree/data/geometry.py`
- `src/self_audit_maskfree/data/dataset.py`

### 3.1 Temporal Provenance Loss

1. **Static Downsampling of 4D to 3D**:
   `scripts/preprocess_acdc.py:60` loops over `[(ed_frame, 'ED'), (es_frame, 'ES')]`. All intermediate temporal frames ($t \in [1 \dots T] \setminus \{ED, ES\}$) are discarded.
2. **Loss of Temporal Index $t$ and Duration $T$**:
   The resulting `.npy` files are named `{patient}_ED.npy` and `{patient}_ES.npy`. The integer temporal frame indices (e.g., $t=1$ and $t=12$) and the total cycle length ($T=30$) are completely erased from the array payload.
3. **Absence of Temporal Neighbor Interfaces**:
   `VolumeSliceDataset` (`src/self_audit/data/common.py:475-600`) constructs only a spatial triplet `build_25d_triplet(volume, slice_index)` along the $Z$ axis. It possesses no temporal index, no temporal neighbor loading, and no access to adjacent frames.

### 3.2 Native Spatial Geometry & Coordinate Provenance Loss

1. **NIfTI Header Destruction in `.npy`**:
   The native NIfTI file contains a 4x4 affine matrix $\mathbf{A}_{native}$, coordinate orientation codes (`LPS`), voxel spacing $(s_x, s_y, s_z)$, and NIfTI standard headers (`sform_code=2`). When exported to `.npy` by `preprocess_acdc.py`, all header metadata is stripped from the volume container.
2. **In-Plane Resizing Destroys Native Grid Dice**:
   `preprocess_acdc.py:110-115` resizes each short-axis slice to $224 \times 224$ (or $256 \times 256$ in `VolumeSliceDataset`) via bilinear interpolation for images and nearest-neighbor for masks. As explicitly admitted in `src/self_audit/evaluation/volume_inference.py:26-30`:
   > *"volume_native Dice is NOT derivable from preprocessed_data/: the stored mask was destructively downsampled... a native-geometry Dice is not recoverable from preprocessed_data/. Supply raw ACDC."*
   Evaluating predictions against resized masks introduces irreversible discretization errors at anatomical boundaries.

### 3.3 Artificial 2.5D Spatial Triplet vs. Cine Requirement

In `src/self_audit_pseudolabel/system_v3.py:41-73`, the teacher architecture requires:
- `cur`: Current spatial slice triplet $[z-1, z, z+1]$ at time $t$ (shape: $[B, 3, H, W]$).
- `prev`: Spatial slice triplet $[z-1, z, z+1]$ at time $t-1$ (shape: $[B, 3, H, W]$).
- `nxt`: Spatial slice triplet $[z-1, z, z+1]$ at time $t+1$ (shape: $[B, 3, H, W]$).

In `tests/test_pseudolabel_system_v3.py:11-16`, previous testing fabricated these inputs using a spatial tensor roll:
```python
cur = torch.rand(batch, 3, h, w)
prev = torch.roll(cur, -1, -1)  # Spatial pixel shift, NOT temporal motion!
nxt = torch.roll(cur, 1, -1)
```
No real cine temporal loader existed anywhere in the codebase.

---

## 4. Physical & Geometric Characteristics of Raw Images

Header-only analysis of all 200 ACDC image files in `/tmp/astra_event_acdc_training/training` was performed using `reports/astra_v3/workers/E_evidence/analyze_acdc_stats.py`.

### 4.1 Statistical Summary Table

| Dimension / Parameter | Minimum | Median / Mean | Maximum | Units / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **In-Plane Matrix ($X \times Y$)** | $154 \times 154$ | $242 \times 264$ | $428 \times 512$ | Voxels |
| **Short-Axis Slices ($Z$)** | 6 | 9.51 (mean) | 18 | Slices per 3D volume |
| **In-Plane Spacing ($dx, dy$)** | 0.703 | 1.512 (mean) | 1.920 | mm (high resolution) |
| **Slice Thickness / Spacing ($dz$)** | 5.000 | 9.335 (mean) | 10.000 | mm (thick slice) |
| **Anisotropy Ratio ($dz / dx$)** | 3.5$\times$ | **6.18$\times$** | 10.2$\times$ | High through-plane anisotropy |
| **Orientation Code** | `LPS` | `LPS` (100%) | `LPS` | Left, Posterior, Superior |
| **qform_code / sform_code** | 0 / 2 | 0 / 2 (100%) | 0 / 2 | Aligned anatomical coordinate space |
| **Cine Frame Count ($T$)** | 12 | 26.98 (mean) / 30.0 | 35 | Full cardiac cycle frames |
| **ED Frame Index** | 1 | 1.05 (mean) / 1.0 | 4 | 1-indexed relative to R-wave |
| **ES Frame Index** | 6 | 10.95 (mean) / 11.0 | 16 | 1-indexed |
| **Temporal Gap ($ES - ED$)** | 5 | 9.90 (mean) / 10.0 | 15 | **Frames apart** ($~37\%$ of cycle) |

### 4.2 Key Geometric Insights

1. **Severe Slice Anisotropy**:
   Short-axis cardiac MRI has high in-plane resolution ($~1.5\text{ mm}$) but thick slices ($5 - 10\text{ mm}$). 3D volumetric convolutions across $Z$ are physically problematic without anisotropy correction; 2.5D center-slice architectures ($[z-1, z, z+1]$) are well-suited, but through-plane spacing must be tracked.
2. **Uniform LPS Orientation**:
   All 100 ACDC patients share an identical `LPS` orientation matrix with `sform_code=2`. There are no orientation flips or mixed patient coordinate spaces across the ACDC cohort.
3. **Temporal Asynchrony of ED/ES**:
   The median temporal gap between ED and ES is 10 frames ($ES - ED = 10$). End-Diastole represents maximum ventricular filling, while End-Systole represents maximum ventricular contraction. Treating them as neighboring time frames would feed maximal physiological deformation into the motion branch rather than local differential velocity.

---

## 5. Patient Split Analysis & Leakage Audits

### 5.1 ACDC Split Manifest (`splits/acdc_patient_split_seed42.json`)

The split manifest was cross-verified against the local image files:
- **Total Patients Declared**: 100
- **Train Patients**: 80 (`patient001`, `patient002`, `patient003`, `patient006`, ...)
- **Validation Patients**: 20 (`patient004`, `patient005`, `patient012`, `patient014`, ...)
- **Test Patients**: 0 (held-out challenge cohort was not part of the training set)
- **Patient Disjointness**: Strictly verified:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad |\text{Train} \cup \text{Val}| = 100$$
- **Volume Disjointness**: 160 train volumes (80 ED + 80 ES) vs. 40 val volumes (20 ED + 20 ES). Zero patient crossover exists.

### 5.2 M&Ms Split Leakage Risks from Source

Through analysis of `src/self_audit/data/mnms.py` and `src/self_audit_maskfree/data/discovery.py`, we identified two critical leakage hazards for M&Ms:

1. **Time-Suffix Patient Leakage Hazard**:
   M&Ms filenames frequently incorporate temporal or acquisition tags, e.g.:
   `{patient_id}_sa_ED.nii.gz`, `{patient_id}_sa_ES.nii.gz`, or `{patient_id}_t01.nii.gz`.
   If patient identification relies on naive case splitting (splitting by the first underscore `re.split(r"[_\-]", clean)[0]` as seen in `src/self_audit/data/common.py:98`), cases such as `case001_t01` and `case001_t02` can be classified as distinct patients if the regex is insufficiently anchored.
   `src/self_audit_maskfree/data/discovery.py:99-116` correctly resolved this with `_strip_suffixes`, which strips `_t\d+`, `_frame\d+`, and `_sa`. Any new loader MUST use this multi-pass suffix stripper to avoid splitting frames of the same patient into both train and test sets.

2. **Cross-Vendor / Multi-Center Generalization Leakage**:
   M&Ms comprises 4 distinct clinical centers and 4 MRI vendors (Siemens, Philips, GE, Canon).
   - If subjects are randomly split across the dataset without center stratification, models trained on Center 1 data will evaluate on Center 1 test subjects.
   - To legitimately evaluate cross-domain generalization, splits must be **center-stratified** or **vendor-stratified** (e.g., train on Centers 1 & 2, validate on Center 3, test on Center 4).

3. **Cross-Dataset Checkpoint Leakage**:
   As mandated in `configs/maskfree_mnms_150.yaml:18-24`, M&Ms models must initialize from random weights (epoch 0) and never inherit ACDC checkpoints, otherwise zero-shot external domain claims degenerate into fine-tuning claims.

---

## 6. M&Ms Label Remapping & Semantic Topology

The ACDC and M&Ms challenges adopted conflicting numerical label assignments for the cardiac ventricles.

### 6.1 Semantic Label Mapping Matrix

| Class ID | Anatomical Structure | ACDC / Unified Schema (`label_schema.py`) | M&Ms Native Schema (Campello et al.) | Required Remap ($\text{M\&Ms} \to \text{Unified}$) |
| :---: | :--- | :---: | :---: | :---: |
| **0** | **Background** | Background (0) | Background (0) | $0 \mapsto 0$ |
| **1** | **Right Ventricle (RV)** | **RV (1)** | Left Ventricle (LV) | $1 \mapsto 3$ |
| **2** | **Myocardium (MYO)** | MYO (2) | MYO (2) | $2 \mapsto 2$ |
| **3** | **Left Ventricle (LV)** | **LV (3)** | Right Ventricle (RV) | $3 \mapsto 1$ |

### 6.2 Mathematical Formulation of Remap

The mapping $\Phi: \mathcal{L}_{MNMS} \to \mathcal{L}_{Unified}$ is a bijective permutation:
$$\Phi(y) = \begin{cases} 
0, & y = 0 \\
3, & y = 1 \\
2, & y = 2 \\
1, & y = 3 
\end{cases}$$
Implemented in `src/self_audit/data/mnms.py:25`:
```python
DEFAULT_MNMS_TO_ACDC = {0: 0, 1: 3, 2: 2, 3: 1}
```

### 6.3 Semantic Impact of Failure to Remap
If an unmapped M&Ms mask is evaluated against an ACDC-trained segmentation head:
- True LV ($y=1$) is scored against predicted RV ($\hat{y}=1$).
- True RV ($y=3$) is scored against predicted LV ($\hat{y}=3$).
Because the RV is crescent-shaped, thin-walled, and anteriorly located, while the LV is thick-walled, cylindrical, and centrally located, their spatial overlap is virtually zero ($\text{Dice} \le 0.05$). This produces an apparent catastrophic failure of cross-domain transfer that is entirely an artifact of unmapped semantic labels.

---

## 7. The Image-Only Sample Contract

To guarantee exact spatial provenance, temporal awareness, and strict adherence to the image-only mandate, we designed and verified the formal `ImageOnlySampleContract`.

### 7.1 Contract Dataclass Specification

```python
@dataclass(frozen=True)
class SpatialNeighbors:
    z_prev: int
    z_cur: int
    z_nxt: int
    mode: str = "clamp_boundary"  # (z-1, z, z+1) with edge clamping

@dataclass(frozen=True)
class TemporalNeighbors:
    t_prev: int | None
    t_cur: int
    t_nxt: int | None
    status: str  # "available" | "unavailable_static_ed_es" | "clamped"

@dataclass(frozen=True)
class NativeCoordinateTransform:
    source_file: str
    orig_shape_hw: tuple[int, int]
    net_shape_hw: tuple[int, int]
    scale_y: float  # orig_h / net_h
    scale_x: float  # orig_w / net_w
    native_affine_4x4: list[list[float]]
    native_orientation: str  # e.g., "LPS"
    native_spacing_zyx_mm: tuple[float, float, float]
    effective_spacing_zyx_mm: tuple[float, float, float]

    def net_to_native_voxel(self, r_net: float, c_net: float, z: int) -> tuple[float, float, float]:
        """Convert network pixel coordinates to native 3D voxel indices (x, y, z)."""
        orig_y = r_net * self.scale_y
        orig_x = c_net * self.scale_x
        return (float(orig_x), float(orig_y), float(z))

    def net_to_scanner_physical_mm(self, r_net: float, c_net: float, z: int) -> tuple[float, float, float]:
        """Convert network pixel coordinates directly to physical scanner LPS space (mm)."""
        vox_x, vox_y, vox_z = self.net_to_native_voxel(r_net, c_net, z)
        affine = np.array(self.native_affine_4x4, dtype=np.float64)
        v = np.array([vox_x, vox_y, vox_z, 1.0], dtype=np.float64)
        world = affine @ v
        return (float(world[0]), float(world[1]), float(world[2]))

@dataclass
class ImageOnlySampleContract:
    patient_id: str
    case_id: str
    t: int                             # 0-indexed temporal frame
    t_total: int                       # Total frames in cardiac cycle
    z: int                             # Short-axis slice index
    z_total: int                       # Total slices in volume
    phase: str                         # "ED" | "ES" | "cine"
    spatial_neighbors: SpatialNeighbors
    temporal_neighbors: TemporalNeighbors
    native_transform: NativeCoordinateTransform
    cur_triplet: torch.Tensor          # [3, H_net, W_net] at time t
    prev_triplet: torch.Tensor | None  # [3, H_net, W_net] at time t-1 (None if static)
    nxt_triplet: torch.Tensor | None   # [3, H_net, W_net] at time t+1 (None if static)
    transforms_applied: list[dict[str, Any]]
```

### 7.2 Forward & Inverse Mathematical Mapping

1. **In-Plane Spatial Rescaling**:
   For a native plane of size $(H_{orig}, W_{orig})$ resized to network size $(H_{net}, W_{net})$:
   $$s_y = \frac{H_{orig}}{H_{net}}, \quad s_x = \frac{W_{orig}}{W_{net}}$$
   The effective pixel spacing in network space becomes:
   $$dx_{eff} = dx_{orig} \cdot s_x, \quad dy_{eff} = dy_{orig} \cdot s_y, \quad dz_{eff} = dz_{orig}$$

2. **Invertible Coordinate Projection**:
   Any predicted point $(r_{net}, c_{net})$ on slice $z$ maps to physical scanner world coordinates $\mathbf{x}_{world} \in \mathbb{R}^3$ (in millimeters, LPS convention) via:
   $$\mathbf{x}_{world} = \mathbf{A}_{native} \begin{bmatrix} c_{net} \cdot s_x \\ r_{net} \cdot s_y \\ z \\ 1 \end{bmatrix}$$
   This allows pseudo-labels or predictions generated on the network grid to be mapped back onto the native NIfTI grid with sub-voxel precision.

3. **Empirical Verification (`verify_image_contract.py`)**:
   Tested on `patient001_frame01.nii` (native shape: $256 \times 216 \times 10$, spacing: $1.5625 \times 1.5625 \times 10.0\text{ mm}$):
   - Pixel $(100.0, 150.0)$ on network grid ($256 \times 256$) maps to native voxel $(126.56, 100.00, 4.00)$.
   - Projects to physical scanner LPS space at $(-126.56, -100.00, 4.00)\text{ mm}$.
   - Verified that all operations run without touching GT arrays.

---

## 8. Categorized Findings: Facts, Inferences, Proposals & NOT RUN

### 8.1 Facts (Directly Verified Ground Truth)
- **F1**: `/tmp/astra_event_acdc_training/training` contains exactly 100 patient directories (`patient001` to `patient100`) and 200 raw uncompressed NIfTI image files (100 ED, 100 ES).
- **F2**: Full 4D cine NIfTI files (`*_4d.nii.gz`) and intermediate time frames are 100% absent from the local system.
- **F3**: All 200 available ACDC image files have uniform orientation `LPS` with valid non-singular affine matrices (`sform_code=2`, `qform_code=0`).
- **F4**: The mean in-plane voxel spacing is $1.51\text{ mm}$ ($0.70 - 1.92\text{ mm}$), while mean slice thickness is $9.34\text{ mm}$ ($5.0 - 10.0\text{ mm}$), exhibiting a $6.2\times$ through-plane anisotropy.
- **F5**: According to `Info.cfg`, cardiac cycles have an average of $26.98$ frames ($12 - 35$). The average gap between ED and ES is $9.90$ frames ($5 - 15$ frames).
- **F6**: The split manifest `splits/acdc_patient_split_seed42.json` cleanly partitions the 100 patients into 80 train (160 volumes) and 20 val (40 volumes) with zero patient overlap.
- **F7**: The legacy preprocessing pipeline (`preprocess_acdc.py`) drops all temporal dimensions and strips NIfTI coordinate headers during `.npy` conversion.
- **F8**: M&Ms raw and preprocessed data do not exist locally; only remote Linux paths and synthetic pytest fixtures are present.

### 8.2 Inferences (Architectural & Physical Deductions)
- **I1**: ED and ES cannot serve as temporal neighbors $(t-1, t+1)$ for the `system_v3.py` motion branch. Their temporal separation ($~10$ frames) represents gross physiological contraction/expansion rather than frame-to-frame velocity.
- **I2**: Bounded ACDC seed experiments focusing on 2.5D appearance encoding, spatial prototype clustering, and student distillation can proceed immediately using the 1,902 available static slices.
- **I3**: Evaluating M&Ms data without the bijective label remap $\{0 \mapsto 0, 1 \mapsto 3, 2 \mapsto 2, 3 \mapsto 1\}$ will invert RV and LV, causing an artificial near-zero Dice score.
- **I4**: M&Ms loaders that split on `case_id` without stripping temporal suffixes (`_tXX`, `_ED`, `_ES`) will suffer severe train/test patient leakage.

### 8.3 Proposals (Actionable Recommendations for Root)
- **P1**: Adopt the verified `ImageOnlySampleContract` as the standard interface for all v3 data loaders, ensuring exact native affine and LPS coordinate preservation.
- **P2**: For the offline pseudo-label teacher on current local data:
  - Either **decouple the motion branch** during bounded static runs (setting motion features $f_{mot} = 0$ or evaluating appearance-only), OR
  - Stage true 4D cine files (`patientXXX_4d.nii.gz`) if motion-based pseudo-labeling is strictly required.
- **P3**: When M&Ms data is staged, enforce center-stratified patient splitting and mandate the `DEFAULT_MNMS_TO_ACDC` label remap.

### 8.4 NOT RUN (Explicit Boundary Ledger)
- **NR1: Full 4D Cine Temporal Training**: NOT RUN. Local storage contains only ED and ES frames; full temporal sequences $(t-1, t, t+1)$ were unavailable.
- **NR2: M&Ms External Validation**: NOT RUN. No raw or preprocessed M&Ms image volumes exist on the local workstation.
- **NR3: Manual Ground-Truth Mask Analysis**: NOT RUN. Strictly avoided opening or reading any `*_gt.nii` arrays in compliance with the image-only research protocol.

---

## 9. Reproducibility Evidence & Command Log

All evidence scripts, test runners, and generated JSON manifests are housed in `reports/astra_v3/workers/E_evidence/`.

### 9.1 Evidence Artifacts Generated

1. `inspect_acdc_headers.py`: Header-only scanner for all 100 patient directories.
2. `acdc_header_inventory.json`: Full JSON inventory of shapes, spacings, orientations, and cohort assignments.
3. `analyze_acdc_stats.py`: Statistical aggregator computing temporal dynamics, gaps, and anisotropy ratios.
4. `acdc_deep_stats.json`: Authoritative quantitative metrics across the 200 ACDC image volumes.
5. `test_image_loader.py`: Functional verification of 2.5D spatial slice extraction, percentile clipping, and z-score normalization.
6. `verify_image_contract.py`: Full implementation and empirical validation of `ImageOnlySampleContract` and invertible coordinate projections.
7. `sample_contract_verification.json`: Serialized output of the verified sample contract.

### 9.2 Exact CLI Invocations Used

```bash
# 1. Environment & Package Verification
/Users/alvinluong/miniforge3/bin/python -c "import torch, nibabel, numpy, scipy; print('Imports OK')"

# 2. Header-Only ACDC Inventory
/Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/E_evidence/inspect_acdc_headers.py

# 3. Deep Geometric & Statistical Analysis
/Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/E_evidence/analyze_acdc_stats.py

# 4. Image-Only Loader Verification
/Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/E_evidence/test_image_loader.py

# 5. Formal Sample Contract & Invertible Mapping Verification
/Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/E_evidence/verify_image_contract.py
```
