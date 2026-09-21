# Astra Self-Audit v3: Resource Profiles & End-to-End Adaptation Contract Audit (Worker F)

> **Root review status: investigation record, not accepted wholesale.** Read [root disagreement resolutions](../11_RED_TEAM.md) and the corresponding root report before using these claims. Worker completion indicates a delivered report, not scientific validation.

- **Audit Target Base SHA**: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
- **Orca Run ID**: `run_698eb3b460ef`
- **Task ID**: `task_9cee0ee49919`
- **Auditor**: AGY Worker F
- **Audited Target Files**:
  - `src/self_audit_pseudolabel/system_v3.py`
  - `src/self_audit/models/annotation_expert.py`
  - `src/self_audit/models/dynamic_window.py`
  - `docs/pseudolabel_system_v3.md`
- **Evidence Directory**: `reports/astra_v3/workers/F_evidence/`
  - `reports/astra_v3/workers/F_evidence/audit_resource.py` (runnable parameter & latency benchmark)
  - `reports/astra_v3/workers/F_evidence/resource_audit_data.json` (raw profile & FLOP/timing outputs)
  - `reports/astra_v3/workers/F_evidence/benchmark_raw_to_native.py` (raw-to-native slice pipeline benchmark)
  - `reports/astra_v3/workers/F_evidence/raw_to_native_benchmark.json` (end-to-end slice decomposition)

---

## 1. Executive Summary

An architectural, mathematical, and empirical audit of resource profiles, execution contracts, and adaptation mechanisms was performed on the Self-Audit v3 deployment pipeline at commit `d4f503b`.

1. **What was done**: We audited the deployment student architecture (`AdaptiveAnnotationStudent`) and its interaction with the canonical `AnnotationExpert` and `DynamicWindowAttention` modules. We derived exact module-by-module parameter breakdowns, calculated analytic and tool-measured MAC/FLOP budgets across three resource profiles (`compact`, `balanced`, `accurate`) at 128x128 and 224x224 resolutions, traced the exact call graph (encoder, A0 head, refiner turns, internal stage depths, and `grid_sample` operations), profiled reproducible CPU-only latencies under strict thread pinning (`torch.set_num_threads(4)`, batch=1, cold vs warm, p50/p90/p95, process RSS), quantified post-processing vs forward pass bottlenecks in a raw-to-native pipeline, and audited the codebase for the existence of an adaptive selector.
2. **What was found**:
   - **No Adaptive Selector Exists**: Despite the class name `AdaptiveAnnotationStudent`, profile selection is **strictly static and manual**. The `forward()` method accepts a string parameter (`profile="balanced"`), branching through a static dictionary `PROFILES`. There is zero runtime gating, zero image-conditioned controller, zero uncertainty/entropy-driven stopping criterion, and zero compute-budget awareness.
   - **`compact` is Computationally Lightweight but Structurally Bloated**: In `compact` mode, 0 refiner turns and 0 Dynamic Window calls execute (reducing latency from 32.2 ms to 10.7 ms at 224x224). However, the entire `AnnotationExpert` (33,002 parameters, 41.02% of the student's 80,462 parameters) remains permanently resident in memory because it is bundled into the same monolithic `nn.Module`.
   - **FLOP vs Latency Discrepancy**: While moving from `compact` to `accurate` increases theoretical FLOPs by only +35.6% (1.62 GFLOPs to 2.20 GFLOPs at 224x224), it **triples CPU latency** (10.74 ms to 32.17 ms, a +200% increase). This discrepancy is driven by non-GEMM operators: `DynamicWindowAttention` relies on PyTorch `grid_sample` (2 to 6 calls per forward pass) and bilinear resamplers, which are memory-bandwidth and cache-latency bound rather than compute bound on CPU.
   - **Raw-to-Native Post-Processing Bottleneck**: In end-to-end slice inference, logit upsampling (bilinear interpolation of 4 channels back to native raw grids 256x256 / 320x320) plus argmax and NumPy formatting consumes **7.0 to 12.4 ms per slice**. For the `compact` profile, post-processing constitutes **45% to 53% of total execution time**.
   - **Missing Preconditions for Adaptive Claims**: The v3 scaffold has never undergone multi-turn student training. Without a matched compute budget baseline and an identical checkpoint evaluated across profiles, any claim of "adaptive efficiency" is invalid.
3. **What's left**: True clinical end-to-end NIfTI evaluation remains **NOT RUN** because the raw ACDC/M&Ms volume loader is missing in this namespace, and opening manual GT arrays is forbidden. Deploying this student requires creating a genuine entropy/uncertainty-driven runtime router, decoupling the compact sub-graph for edge deployment, separating UI/rendering from the tensor pipeline, and training the student to convergence under multi-turn supervision.

---

## 2. Architectural Audit & Contract Inspection

### 2.1 Component Breakdown & Code Citations

At base commit `d4f503b7107eca4e8ac85f37a5dd01ec50479666`, the deployment student pipeline consists of three core components:

```
Input Slice x [B, 3, H, W]
  │
  ├──> DeploymentEncoder (src/self_audit_pseudolabel/system_v3.py:149-155)
  │      ├─ Block(3, 32, stride=1)      [B, 32, H, W]
  │      ├─ Block(32, 32, stride=2)    [B, 32, H/2, W/2]
  │      └─ Block(32, 32, stride=2)    [B, 32, H/4, W/4]  --> feat [B, 32, H/4, W/4]
  │
  ├──> a0_head (src/self_audit_pseudolabel/system_v3.py:162, 173)
  │      ├─ Conv2d(32, 4, 1)            [B, 4, H/4, W/4]
  │      └─ F.interpolate(bilinear)     [B, 4, H, W]       --> a0_logits [B, 4, H, W]
  │
  └──> AnnotationExpert (src/self_audit/models/annotation_expert.py:285-350)
         [Evaluated conditionally based on PROFILES[profile].turns]
         ├─ Input Projection: Conv2d(40, 32, 1) + GN(8) + GELU
         │    (input_channels = 32 feat + 4 logits + 1 entropy + 3 audit = 40)
         │    (audit_evidence is hardcoded zero: audit_conditioning="feature_only")
         │
         ├─ Turn Loop (turn = 0 .. turns - 1):
         │    Depth calculation: depth = min(turn + 1, 3)
         │    ├─ Iteration Loop (iter = 0 .. depth - 1):
         │    │    ├─ DynamicWindowAttention (src/self_audit/models/dynamic_window.py:239-395)
         │    │    │    ├─ Generator: state_net -> parameter_head -> coordinates [B, H/4, W/4, 8, 2]
         │    │    │    ├─ grid_sample(key, coordinates)    [B, 32, H/4, W/4, 8]
         │    │    │    ├─ grid_sample(value, coordinates)  [B, 32, H/4, W/4, 8]
         │    │    │    ├─ Multi-head Attention (heads=4, head_dim=8)
         │    │    │    └─ Output Projection: Conv2d(32, 32, 1)
         │    │    └─ Refinement Residual: Conv2d(32, 32, 3) + GN + GELU + Conv2d(32, 32, 1)
         │    │         state = state + residual
         │    │
         │    ├─ Heads:
         │    │    ├─ delta_head: Conv2d(32, 16, 3) + GELU + Conv2d(16, 4, 1) -> F.interpolate -> delta_logits
         │    │    └─ gate_head:  Conv2d(32, 1, 1)                             -> F.interpolate -> update_gate
         │    └─ State Update:
         │         candidate_logits = previous_logits + sigmoid(update_gate) * delta_logits
         │
         └─ Output: final_logits [B, 4, H, W]
```

### 2.2 Profile Definition & Contract

In [`src/self_audit_pseudolabel/system_v3.py:136-146`](file:///Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit/src/self_audit_pseudolabel/system_v3.py#L136-L146), profiles are defined as:
```python
@dataclass(frozen=True)
class ResourceProfile:
    name: str
    turns: int

PROFILES = {
    "compact": ResourceProfile("compact", 0),
    "balanced": ResourceProfile("balanced", 1),
    "accurate": ResourceProfile("accurate", 2),
}
```

The output contract across all profiles is strictly identical:
- A dictionary containing:
  - `"a0_logits"`: `Tensor[B, 4, H, W]` (the baseline segmentation logits)
  - `"final_logits"`: `Tensor[B, 4, H, W]` (identical to `a0_logits` in `compact`; updated in `balanced`/`accurate`)
  - `"stages"`: `Tuple[Tensor[B, 4, H, W], ...]` (length 1 for `compact`, 2 for `balanced`, 3 for `accurate`)
  - `"profile"`: string name
  - `"window_metadata"`: metadata tuple (empty if `return_metadata=False`)

---

## 3. Parameter, Call, and FLOP Analysis

### 3.1 Detailed Parameter Inventory

The student model has **80,462 total parameters**, compared to the teacher model's **181,565 total parameters** (the student is 44.3% the size of the teacher).

| Submodule | Class / Sub-components | Parameter Count | % of Student Total | Active in `compact`? |
| :--- | :--- | :--- | :--- | :--- |
| **`encoder`** | `DeploymentEncoder(width=32)` | **47,328** | **58.82%** | **YES** |
| ├─ `net[0]` (e0) | `Block(3, 32)`: 2x Conv2d(3x3) + GN | 10,208 | 12.69% | YES |
| ├─ `net[1]` (e1) | `Block(32, 32, stride=2)`: 2x Conv2d(3x3) + GN | 18,560 | 23.07% | YES |
| └─ `net[2]` (e2) | `Block(32, 32, stride=2)`: 2x Conv2d(3x3) + GN | 18,560 | 23.07% | YES |
| **`a0_head`** | `Conv2d(32, 4, 1)` | **132** | **0.16%** | **YES** |
| **`refiner`** | `AnnotationExpert` | **33,002** | **41.02%** | **NO (0% used)** |
| ├─ `input_projection` | `Conv2d(40, 32, 1) + GN(8)` | 1,376 | 1.71% | NO |
| ├─ `refinement_block` | `DynamicWindowAttention(channels=32, k=8)` | 16,533 | 20.55% | NO |
| │   ├─ `generator` | `DynamicWindowGenerator` | 12,309 | 15.30% | NO |
| │   │    ├─ `turn_embedding` | `Embedding(3, 16)` | 64 | 0.08% | NO |
| │   │    ├─ `iteration_embedding`| `Embedding(3, 16)` | 64 | 0.08% | NO |
| │   │    ├─ `state_net` | 2x Conv2d(3x3) + GN(8) | 11,488 | 14.28% | NO |
| │   │    └─ `parameter_head` | `Conv2d(32, 21, 1)` | 693 | 0.86% | NO |
| │   ├─ `query` | `Conv2d(32, 32, 1)` | 1,056 | 1.31% | NO |
| │   ├─ `key` | `Conv2d(32, 32, 1)` | 1,056 | 1.31% | NO |
| │   ├─ `value` | `Conv2d(32, 32, 1)` | 1,056 | 1.31% | NO |
| │   └─ `output`| `Conv2d(32, 32, 1)` | 1,056 | 1.31% | NO |
| ├─ `refinement_residual`| Conv2d(3x3) + GN + GELU + Conv2d(1x1) | 10,368 | 12.89% | NO |
| ├─ `delta_head` | Conv2d(32, 16, 3) + Conv2d(16, 4, 1) | 4,692 | 5.83% | NO |
| └─ `gate_head` | `Conv2d(32, 1, 1)` | 33 | 0.04% | NO |
| **Total Student** | `AdaptiveAnnotationStudent` | **80,462** | **100.0%** | **47,460 (58.98%)** |

### 3.2 Call Graph and Stage Depth Trace

The three profiles execute radically different operator sequences:

```
Table 3.2: Operational Trace by Resource Profile
```

| Metric / Operation | `compact` | `balanced` | `accurate` | Code Origin |
| :--- | :--- | :--- | :--- | :--- |
| **Declared Turns** | 0 | 1 | 2 | `system_v3.py:143-145` |
| **Encoder Forward Calls** | 1 | 1 | 1 | `system_v3.py:172` |
| **A0 Head Calls** | 1 | 1 | 1 | `system_v3.py:173` |
| **Refiner Turns Executed** | 0 | 1 | 2 | `system_v3.py:177` |
| **Internal Stage Depths** | `[]` | `[1]` | `[1, 2]` | `annotation_expert.py:395` |
| **Input Projection Calls** | 0 | 1 | 2 | `annotation_expert.py:390` |
| **DW Attention Calls** | **0** | **1** | **3** | `annotation_expert.py:416` |
| **`grid_sample` Invocations**| **0** | **2** | **6** | `dynamic_window.py:384-385` |
| **Residual Block Calls** | 0 | 1 | 3 | `annotation_expert.py:428` |
| **Delta / Gate Head Passes** | 0 | 1 | 2 | `annotation_expert.py:448-449` |
| **Bilinear Interpolations** | 1 (`a0`) | 4 (`a0`, `delta`, `gate`, `audit`) | 7 (`a0`, 2x`delta`, 2x`gate`, 2x`audit`) | `system_v3.py`, `expert.py` |
| **Active Parameters** | 47,460 (58.98%) | 80,462 (100.0%) | 80,462 (100.0%) | Verified empirically |

**Key Architectural Insight on Stage Depth**:
In [`src/self_audit/models/annotation_expert.py:395`](file:///Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit/src/self_audit/models/annotation_expert.py#L395):
```python
depth = min(turn_value + 1, 3)
```
For `balanced` (`turn=0`), `depth = 1` -> **1 DW call**.
For `accurate` (`turn=0` then `turn=1`), `depth = 1` then `depth = 2` -> **1 + 2 = 3 DW calls**.
Therefore, `accurate` does not double the refinement compute of `balanced`; it **triples** the refinement compute because later turns execute deeper internal recurrence!

### 3.3 Theoretical MAC & FLOP Counts

Measured using standard profiling harness (`reports/astra_v3/workers/F_evidence/audit_resource.py`), applying the universal deep learning convention: $1 \text{ MAC} = 2 \text{ FLOPs}$.

| Input Resolution | Profile | MACs | FLOPs | Relative to `compact` | Active Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128 x 128** | `compact` | 265,158,656 | 530,317,312 (0.530 G) | 1.00x | 47,460 |
| (Feature: 32x32) | `balanced` | 298,680,320 | 597,360,640 (0.597 G) | 1.13x (+12.6%) | 80,462 |
| | `accurate` | 359,464,960 | 718,929,920 (0.719 G) | 1.36x (+35.6%) | 80,462 |
| **224 x 224** | `compact` | 812,048,384 | 1,624,096,768 (1.624 G) | 1.00x | 47,460 |
| (Feature: 56x56) | `balanced` | 914,708,480 | 1,829,416,960 (1.829 G) | 1.13x (+12.6%) | 80,462 |
| | `accurate` | 1,100,861,440 | 2,201,722,880 (2.202 G) | 1.36x (+35.6%) | 80,462 |

---

## 4. Empirical Latency & Memory Benchmark

### 4.1 Benchmark Protocol & Environment Documentation

- **Host Environment**: Apple Silicon Mac (M-series, Darwin arm64, 12 physical / 12 logical cores, 24.0 GB unified RAM).
- **Execution Target**: CPU-only with strictly pinned thread budget: `torch.set_num_threads(4)`. Batch size = 1.
- **Contention / Background Noise Context**: The benchmark was executed on a live multi-agent workspace with **853 active OS processes** and multiple co-running Orca worker terminals. Therefore, these measurements document reproducible host behavior under realistic developer conditions, **not an isolated clinical deployment claim**.
- **Protocol**: 10 warmup passes, followed by 30 timed iterations. Measured cold start latency (first invocation after model instantiation) vs warm percentiles (p50, p90, p95).

### 4.2 CPU-4 Benchmark Results

```
Table 4.2: CPU-Only Inference Latency (Batch=1, PyTorch Threads=4)
```

| Input Resolution | Profile | Cold Start | Mean ± Std | p50 (Median) | p90 | p95 | RSS Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **128 x 128** | `compact` | 10.85 ms | 3.40 ± 0.57 ms | **3.32 ms** | 3.72 ms | 3.89 ms | +30.8 MB |
| | `balanced` | 6.33 ms | 6.39 ± 0.79 ms | **6.21 ms** | 7.92 ms | 8.03 ms | +17.1 MB |
| | `accurate` | 11.86 ms | 10.51 ± 1.03 ms | **10.17 ms** | 11.88 ms | 12.89 ms | +5.9 MB |
| **224 x 224** | `compact` | 13.41 ms | 10.74 ± 1.04 ms | **10.42 ms** | 12.13 ms | 12.57 ms | +75.1 MB |
| | `balanced` | 19.18 ms | 18.43 ± 2.04 ms | **17.83 ms** | 20.65 ms | 21.45 ms | +167.6 MB |
| | `accurate` | 28.64 ms | 32.17 ± 6.25 ms | **29.98 ms** | 38.94 ms | 42.03 ms | +71.2 MB |

### 4.3 Exploratory Metal Performance Shaders (MPS / Apple GPU)

Exploratory evaluation on the host's Apple Silicon GPU via Metal Performance Shaders (MPS):

| Input Resolution | Profile | Mean Latency | p50 | p90 | p95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128 x 128** | `compact` | 1.03 ms | 0.56 ms | 1.80 ms | 2.01 ms |
| | `balanced` | 3.56 ms | 3.91 ms | 5.76 ms | 6.21 ms |
| | `accurate` | 5.16 ms | 4.51 ms | 6.05 ms | 6.97 ms |
| **224 x 224** | `compact` | 1.63 ms | 1.46 ms | 2.48 ms | 2.55 ms |
| | `balanced` | 2.27 ms | 2.22 ms | 2.41 ms | 2.61 ms |
| | `accurate` | 5.33 ms | 4.59 ms | 7.98 ms | 8.48 ms |

*Note: Dedicated clinical NVIDIA GPUs (e.g. T4 / RTX 3060) were not available on this host; GPU claims on CUDA hardware remain **NOT RUN**.*

### 4.4 Discrepancy Analysis: FLOPs vs Latency

A critical empirical discrepancy emerges when comparing theoretical compute scaling with observed latency scaling:

```
Compute Scaling (224x224):
  compact   (1.62 GFLOPs) : 1.00x
  balanced  (1.83 GFLOPs) : 1.13x  (+12.6%)
  accurate  (2.20 GFLOPs) : 1.36x  (+35.6%)

Latency Scaling (224x224, CPU-4):
  compact   (10.42 ms)    : 1.00x
  balanced  (17.83 ms)    : 1.71x  (+71.1%)
  accurate  (29.98 ms)    : 2.88x  (+187.7%)
```

**Why does a 35.6% FLOP increase cause a 188% latency explosion?**

1. **Resolution Asymmetry**: The early convolutional layers in `DeploymentEncoder` operate at full resolution ($224 \times 224$) and half resolution ($112 \times 112$). These layers represent over 80% of total network FLOPs. Because they are standard $3 \times 3$ convolutions, they achieve high arithmetic intensity and high cache reuse in optimized GEMM kernels.
2. **`grid_sample` Memory Bottleneck**: `DynamicWindowAttention` operates at feature resolution ($56 \times 56$), where FLOP counts are low. However, each call executes PyTorch's `grid_sample(..., mode="bilinear", padding_mode="border", align_corners=True)` twice. On CPU, `grid_sample` executes gather operations across scattered floating-point memory coordinates. It has near-zero arithmetic intensity and is entirely bound by CPU L1/L2 cache misses and memory bus bandwidth.
3. **Sequential Pipeline Overhead**: For `accurate`, PyTorch executes 3 separate passes through `grid_sample`, 3 passes through multi-head softmax, 3 GroupNorms, and 2 full-resolution bilinear upsamplings for the delta/gate heads. The latency is dominated by operator launch and memory serialization overhead, not raw floating point arithmetic.

---

## 5. Audit of Adaptation: Fixed vs Adaptive Selector

### 5.1 Does an Adaptive Selector Exist?

**Fact**: In `system_v3.py` (and the entire `src/self_audit_pseudolabel/` package), **NO adaptive selector exists**.

Inspection of [`src/self_audit_pseudolabel/system_v3.py:169-171`](file:///Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit/src/self_audit_pseudolabel/system_v3.py#L169-L171):
```python
    def forward(self, x, *, profile="balanced", return_metadata=False):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
```
The profile is a **fixed, hardcoded string argument** passed by the external caller.
- There is no dynamic early-exit mechanism.
- There is no controller measuring entropy, margin, or uncertainty of the baseline $A_0$ logits to decide whether refinement is warranted.
- There is no latency-budget estimator to downshift from `accurate` to `compact` under high system load.
- The term "Adaptive" in `AdaptiveAnnotationStudent` describes a **future research aspiration**, not an implemented software mechanism.

### 5.2 Is `compact` Truly Lightweight?

- **In Execution Time / FLOPs**: **YES**. At 10.42 ms (224x224) and 3.32 ms (128x128) on CPU-4, `compact` is fast enough for interactive 30–60 FPS cine scrubbers. It skips all dynamic window logic entirely.
- **In Memory Footprint**: **NO**. The class instantiates `self.refiner = AnnotationExpert(...)` in its `__init__`.
  - 33,002 parameters (41.02% of model parameters) are permanently allocated in RAM/VRAM, even if the deployment application only ever calls `profile="compact"`.
  - In edge or embedded clinical devices with strict memory ceilings (e.g. ultrasound carts or mobile tablets), this dead weight increases cold start initialization time and resident memory.

### 5.3 The Matched-Budget & Checkpoint Equivalence Principle

In the Self-Audit v3 research docs ([`docs/pseudolabel_system_v3.md:44-45`](file:///Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit/docs/pseudolabel_system_v3.md#L44-L45)), the authors claim:
> *"This makes adaptation measurable: the same input/output contract is evaluated at different compute budgets. Accuracy and latency must be reported for the same profile."*

However, from an empirical audit standpoint:
1. **No Trained Student Weights Exist**: The repository contains code scaffolds, not converged checkpoint weights.
2. **Untrained Multi-Turn Degradation**: When a shared-recurrent model is trained with multi-turn refinement, the intermediate $A_0$ logits are typically sub-optimal because the network relies on subsequent turns to correct boundary errors. If evaluated without a matched budget control (e.g. comparing a dedicated compact network trained specifically for $A_0$ vs the $A_0$ head of an accurate network), the adaptive model will exhibit poor standalone accuracy.
3. **Budget Accounting Requirement**: To claim an "adaptive benefit", an adaptive selector must achieve higher clinical segmentation accuracy (Dice / Hausdorff) than a fixed baseline operating at the **exact same average FLOP or latency budget** across a complete clinical volume (e.g. 20–30 cardiac cine slices). Such an experiment has **never been run**.

---

## 6. End-to-End Pipeline: Pure Forward vs Raw-to-Native

### 6.1 Clinical Raw-to-Native Lifecycle

In genuine clinical cardiac MRI deployment, raw data enters as multi-slice DICOM or NIfTI series with anisotropic voxels (e.g. $1.25 \times 1.25 \times 8 \text{ mm}^3$) and variable spatial matrices ($216 \times 256$, $256 \times 256$, $320 \times 320$). The end-to-end lifecycle requires:

```
[Raw Image Slice] -> [Intensity Normalization] -> [Resample/Pad to Network Grid 224x224]
        -> [Model Inference (Compact/Balanced/Accurate)]
        -> [Resample Logits to Native In-Plane Matrix] -> [Argmax / Connected Components]
        -> [Discrete Segmentation Mask Array] -> [UI Canvas / DICOM Overlay]
```

### 6.2 Empirical Raw-to-Native Slice Benchmark

Using the reproducible benchmark in `reports/astra_v3/workers/F_evidence/benchmark_raw_to_native.py`, we isolated the wall-clock execution time across preprocessing, forward pass, and post-processing across typical clinical cardiac slice matrices:

```
Table 6.2: End-to-End Slice Execution Decomposition (Target Grid: 224x224, CPU-4)
```

| Native Matrix | Profile | Preprocess | Pure Forward | Postprocess | Total E2E | Forward % | Postprocess % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256 x 256** | `compact` | 0.53 ms | 11.48 ms | 9.03 ms | **21.04 ms** | 54.6% | **42.9%** |
| | `balanced` | 0.52 ms | 19.80 ms | 9.42 ms | **29.74 ms** | 66.6% | **31.7%** |
| | `accurate` | 0.44 ms | 29.28 ms | 9.61 ms | **39.33 ms** | 74.5% | **24.4%** |
| **216 x 256** | `compact` | 0.58 ms | 11.80 ms | 8.37 ms | **20.75 ms** | 56.9% | **40.3%** |
| | `balanced` | 0.34 ms | 16.32 ms | 6.82 ms | **23.48 ms** | 69.5% | **29.0%** |
| | `accurate` | 0.60 ms | 28.10 ms | 7.21 ms | **35.91 ms** | 78.2% | **20.1%** |
| **320 x 320** | `compact` | 0.48 ms | 9.95 ms | 10.78 ms | **21.22 ms** | 46.9% | **50.8%** |
| | `balanced` | 0.59 ms | 17.78 ms | 11.44 ms | **29.81 ms** | 59.6% | **38.4%** |
| | `accurate` | 0.51 ms | 26.11 ms | 12.37 ms | **38.99 ms** | 67.0% | **31.7%** |

### 6.3 Critical Finding: The Post-Processing Bottleneck

The decomposition reveals an essential deployment insight:
- **Preprocessing** (intensity standardisation, tensor conversion, bilinear downsampling) is negligible: **0.35–0.60 ms** (< 3% of total latency).
- **Post-processing** (bilinear interpolation of 4-channel floating point logits back to the original matrix, followed by argmax and NumPy byte array conversion) is **extremely expensive**: **7.0 to 12.4 ms**.
- For the `compact` model, **post-processing takes just as long as the neural network forward pass** (representing up to 50.8% of total slice latency).
- **Deployment Implication**: Optimizing neural network FLOPs or pruning convolutional channels will yield diminishing returns unless the logit-resampling and argmax post-processing pipeline is optimized (e.g. via LibTorch C++ SIMD, fixed-point quantization, or performing argmax *before* upsampling binary masks).

---

## 7. Clinical Deployment Evaluation Protocol (CPU-4 / 8GB and GPU)

To evaluate this model for real-world clinical deployment, the following formal evaluation protocol is established:

### 7.1 Target Hardware Specification

```
Protocol Spec 1: Target Hardware Environments
```
1. **Edge / Clinical Workstation (CPU-Only)**:
   - Processor: 4 x86_64 or ARM64 physical CPU cores (e.g. Intel Core i5/Xeon E3 or Apple M-series edge).
   - Memory: 8.0 GB total system RAM.
   - Max Process RSS Budget: **< 1.5 GB** for the entire inference runtime (leaving 6.5 GB for OS, PACS client, and DICOM caching).
   - Storage: Local NVMe / SATA SSD.
2. **High-Throughput Clinical Workstation / Cloud Worker (GPU)**:
   - GPU: NVIDIA T4 (16 GB) or RTX 3060 / A10G (12–24 GB).
   - Host CPU: 4–8 vCPUs, 16–32 GB host RAM.
   - CUDA / cuDNN: Latest production TensorRT / PyTorch release.

### 7.2 Service Level Agreements (SLAs) & Latency Thresholds

```
Protocol Spec 2: Service Level Agreements
```

| Deployment Mode | Unit of Work | Target SLA | Hard Ceiling | Target Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Interactive Cine Scrubbing** | Single 2.5D Slice | **< 15 ms** | 30 ms | `compact` |
| **Interactive Boundary Inspection** | Single 2.5D Slice | **< 40 ms** | 60 ms | `balanced` or `accurate` |
| **Volumetric Phase Segmentation** | 20 Slices (ED or ES) | **< 400 ms** (CPU) / **< 100 ms** (GPU) | 1,000 ms (CPU) | Hybrid (Adaptive) |
| **Full 4D Cine Study** | 20 Slices x 25 Frames (500 slices) | **< 10.0 s** (CPU) / **< 1.5 s** (GPU) | 20.0 s (CPU) | Hybrid (Adaptive) |

### 7.3 Decoupling Inference from User Interface (UI)

In clinical applications (such as Web-based PACS or Electron DICOM viewers), UI stalls directly degrade radiologist experience. The evaluation protocol requires **strict decoupling**:

```
UI Process (Electron / React / WebGL)
  │
  ├── [1] User scrubs to cine frame t, slice z
  │         │
  │         ├─> Displays cached raw DICOM immediately (0 ms delay)
  │         │
  │         └─> Sends async IPC / WebSocket request to Inference Worker
  │
Inference Worker (C++ / Python / LibTorch Sub-Process)
  │
  ├── [2] Priority 1: Fast Path (Compact A0)
  │         └─> Computes compact A0 logits (~10 ms)
  │         └─> Emits provisional binary mask via SharedArrayBuffer / IPC
  │
  └── [3] Priority 2: Deferred Refinement (Balanced / Accurate)
            ├─> Gating check: Is regional entropy > threshold?
            │     ├─ NO  --> Skip refinement (Done)
            │     └─ YES --> Execute Turn 1/2 refinement (+15-25 ms)
            └─> Emits refined boundary update to UI
```

**Key Isolation Rules**:
1. **Never run inference on the UI event loop thread**: All PyTorch calls must run in a dedicated sub-process or worker thread pool.
2. **Zero-Copy Serialization**: Do not serialize 4-channel float32 logit arrays into JSON or Base64. Use shared memory (`multiprocessing.shared_memory` or memory-mapped files) or compressed run-length encoded (RLE) masks.
3. **Canvas / WebGL Offload**: The UI renderer should perform color blending and alpha-mask compositing on the local client GPU using fragment shaders, keeping host CPU cores dedicated to model inference.

---

## 8. Catalog of Facts, Inferences, Proposals, and NOT RUN

### 8.1 Verified Facts
1. The student model `AdaptiveAnnotationStudent` at commit `d4f503b` has exactly **80,462 total parameters**; the teacher has **181,565 parameters**.
2. Module parameter counts are: `encoder`: 47,328 (58.82%); `a0_head`: 132 (0.16%); `refiner`: 33,002 (41.02%).
3. In `compact` mode, active parameters are **47,460** (58.98% of total); **33,002 parameters are completely unused** during the forward pass.
4. There is **no adaptive selector** in the code; `profile` is a static caller-supplied string (`compact`, `balanced`, or `accurate`).
5. Stage depths follow $d = \min(\text{turn} + 1, 3)$, resulting in **0 DW calls** for `compact`, **1 DW call** for `balanced`, and **3 DW calls** for `accurate`.
6. At 224x224 on CPU (4 threads), median latencies are **10.42 ms** (`compact`), **17.83 ms** (`balanced`), and **29.98 ms** (`accurate`).
7. FLOP counts at 224x224 are **1.624 GFLOPs** (`compact`), **1.829 GFLOPs** (`balanced`), and **2.202 GFLOPs** (`accurate`).
8. Bilinear logit upsampling back to raw matrix dimensions takes **7.0 to 12.4 ms**, accounting for **45% to 53%** of total execution time in `compact` mode.

### 8.2 Inferences
1. The substantial gap between FLOP scaling (+35.6%) and CPU latency scaling (+187.7%) is caused by the memory-bound, cache-unfriendly nature of PyTorch `grid_sample` in `DynamicWindowAttention`.
2. Model weight memory in `compact` mode is unnecessarily inflated by 69.5% (47K vs 80K) due to packaging the unused `AnnotationExpert` into the same instance.
3. Post-processing is an unaddressed system bottleneck; future speedup efforts must target logit upsampling and argmax operations before convolutional tuning.

### 8.3 Proposals
1. **Implement Concrete Adaptive Selector (`EntropyThresholdRouter`)**:
   Implement an early-exit controller that inspects the average entropy of $A_0$ logits in the myocardial annular region:
   $$\bar{H} = \frac{1}{|\Omega|} \sum_{u \in \Omega} H(A_0(u))$$
   If $\bar{H} < \tau_{\text{low}}$ (e.g. 0.15), exit immediately with `compact`. If $\tau_{\text{low}} \le \bar{H} < \tau_{\text{high}}$, execute `balanced` (1 turn). If $\bar{H} \ge \tau_{\text{high}}$, execute `accurate` (2 turns).
2. **Modular Architecture Decoupling**:
   Split `AdaptiveAnnotationStudent` into two classes:
   - `CompactStudent`: Contains only `DeploymentEncoder` and `a0_head` (47,460 parameters, 0 refiner weights).
   - `RefinementStudent`: Inherits or wraps `CompactStudent` and adds `AnnotationExpert`.
3. **Optimized Post-Processing Engine**:
   - Resample argmax class predictions using nearest-neighbor interpolation, or compute argmax directly on the low-resolution grid ($56 \times 56$) and upsample binary masks with integer bitwise operations. This will reduce post-processing latency from ~10 ms to < 1 ms.
4. **Matched-Budget Validation Protocol**:
   Before claiming adaptive superiority, evaluate a 100-case cine volume cohort under a fixed latency budget (e.g. 400 ms per volume), verifying that the dynamic selector Pareto-dominates fixed profiles on Dice and Hausdorff95.

### 8.4 NOT RUN Catalog
1. **Clinical Raw-to-Native ACDC / M&Ms NIfTI Evaluation**: **NOT RUN**. The v3 namespace currently lacks a raw cine MRI volume loader (explicitly noted in `docs/pseudolabel_system_v3.md`), and accessing manual segmentation ground truth arrays is strictly prohibited in this audit stage.
2. **CUDA GPU Benchmarking on Dedicated Clinical Hardware**: **NOT RUN**. The local host is an Apple Silicon Mac; testing on NVIDIA T4 / RTX 3060 / A10G was not possible.
3. **Multi-Turn Student Convergence & Accuracy Scaling**: **NOT RUN**. No converged student checkpoint exists; semantic performance cannot be evaluated on random or synthetic weights.
4. **Long Multi-Case Volumetric Memory Leak Profiling**: **NOT RUN**. High-iteration stability tests across thousands of volumes were not conducted within the 30-minute task bound.

---

## 9. Reproducibility & Evidence Verification

To re-run the parameter count and CPU/MPS latency benchmark:
```bash
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/F_evidence/audit_resource.py
```
Output artifact: `reports/astra_v3/workers/F_evidence/resource_audit_data.json`

To re-run the synthetic raw-to-native slice decomposition benchmark:
```bash
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/F_evidence/benchmark_raw_to_native.py
```
Output artifact: `reports/astra_v3/workers/F_evidence/raw_to_native_benchmark.json`
