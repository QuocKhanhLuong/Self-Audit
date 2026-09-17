# DFC Cardiac Benchmark Audit

Audit date: 2026-09-17. Scope: read-only audit; no implementation or benchmark run.

## 1. Repository identity

| Item | Verified result |
|---|---|
| Repository path | `D:\projects\DFC\pytorch-unsupervised-segmentation-tip` |
| Branch | `master` |
| HEAD | `181318ad40dbfb5c0add8580a05c30e5e3a7ad58` |
| Origin | `https://github.com/Azios1010/pytorch-unsupervised-segmentation-tip.git` |
| Git status | Clean; `master...origin/master`; no modified or untracked files |
| Official upstream HEAD | `181318ad40dbfb5c0add8580a05c30e5e3a7ad58`, verified using `git ls-remote` |
| Correspondence to official source | Same commit as official `kanezaki/pytorch-unsupervised-segmentation-tip`; origin is a fork |

Latest five commits:

| SHA | Date | Subject |
|---|---|---|
| `181318ad40dbfb5c0add8580a05c30e5e3a7ad58` | 2020-12-29 | bug fixed (for when batch_size >1) |
| `5d2ee084720ff9773f73227833e3fbac84720c55` | 2020-07-22 | removed scikit-image |
| `4dc7e27d58018103ddb1a518d0ad60a3529fef85` | 2020-07-22 | added LICENSE |
| `7bfa7c64335ae2ea50bf3df633e851c657209567` | 2020-07-22 | modified README.md |
| `f2b705ba87d82cae686ad87f890b8054884eb9c1` | 2020-07-22 | added files. |

There are no local code differences to account for before analyzing behavior.

I inspected both Python scripts completely, the README, license, repository inventory, and relevant textual search results. There are no separate helper modules, dependency lockfiles, dataset preparation scripts, evaluation scripts, or tests in this checkout. The remaining files are demonstration images and scribble annotations.

The FreeMask reference was inspected at `QuocKhanhLuong/Self-Audit@96c32b10fc7b8e09b48822e10ae9eb6cc149e253`, using its `src/self_audit_maskfree/` code and mask-free configurations. The older supervised pipeline was not used as the benchmark reference.

**Audit scope:** no repository files were changed, no implementation plan was created, and no training or benchmark was run. A read-only extraction and instantiation of `MyNet` verified parameter counts. This report is stored outside the DFC checkout.

## 2. Original DFC task

DFC produces an anonymous pixel partition of an input image through optimization of a randomly initialized CNN. Its direct executable handles one image per invocation.

The repository exposes three variants:

| Variant | Executable | Information used |
|---|---|---|
| Direct | `demo.py` | One image |
| Scribble-assisted | `demo.py --scribble` | Image and human annotations |
| Reference-image | `demo_ref.py` | Reference images, followed by target-image inference |

The original output is a color visualization of cluster assignments. It is not a cardiac semantic segmentation, calibrated probability map, or evaluated segmentation result.

The README describes these three entry points, consistent with the executable code. [README.md, line 26](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/README.md#L26)

## 3. Executable pipeline

Let `C = nChannel`, with original image height `H` and width `W`. Under ordinary PyTorch defaults, floating tensors and model parameters are `float32`.

Device `D` is the current CUDA device when `torch.cuda.is_available()` is true; otherwise CPU. The script has no explicit device-selection argument.

| Stage | Exact operation | Shape / dtype / device | Mode and gradients |
|---|---|---|---|
| Load | `cv2.imread(args.input)` | `[H,W,3]`, normally `uint8`, CPU | BGR image; no gradients |
| Convert | Transpose, cast, divide by 255, add batch dimension | `[1,3,H,W]`, `float32`, initially CPU | No input gradients |
| Transfer | Optional `.cuda()`; wrap in `Variable` | Same shape, device `D` | `requires_grad=False` |
| Initialize | `MyNet(data.size(1))` | Default three-channel network | Parameters initialized on CPU, then optionally transferred |
| Training mode | `model.train()` | All modules | BN uses current batch/spatial statistics |
| First block | Conv3×3 → ReLU → BN | `[1,C,H,W]`, float32, `D` | Differentiable with respect to parameters |
| Repeated blocks | `nConv-1` Conv3×3 → ReLU → BN blocks | `[1,C,H,W]` | Differentiable |
| Response head | Conv1×1 → BN | `[1,C,H,W]` | Differentiable; no final ReLU or softmax |
| Flatten | Select batch 0; permute; contiguous; view | `[HW,C]` | Differentiable |
| Continuity view | Reshape responses | `[H,W,C]` | Differentiable |
| Differences | Adjacent-row and adjacent-column subtraction | `[H-1,W,C]`, `[H,W-1,C]` | Differentiable |
| Self-label | `torch.max(output, 1)` indices | `[HW]`, `int64`, `D` | Integer target has no gradients |
| Count labels | Copy indices to CPU NumPy; `np.unique` | CPU integer array and Python count | No gradients |
| Similarity loss | Cross-entropy of responses against current self-labels | Scalar, float32, `D` | Gradients through response input |
| Total loss | Weighted similarity plus continuity | Scalar | Differentiable |
| Update | `loss.backward(); optimizer.step()` | Parameter and momentum updates | One update per iteration |
| Stop | Compare previously measured count with `minLabels` | Python condition | Executed after update |
| Headless final output | Fresh forward and argmax | `[HW]`, then conceptually `[H,W]` | Still train mode; graph is built, but no backward |
| Save | Random-color lookup; `cv2.imwrite` | `[H,W,3]`, uint8, CPU | Saves `output.png` |

The sequence is defined in [demo.py, line 68](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L68), with the optimization loop beginning at line 117 and final-output branch at line 153.

The script never calls `model.eval()` or `torch.no_grad()`.

## 4. Architecture

The exact default network is:

```text
[1,3,H,W]
  Conv2d(3,100, kernel=3, stride=1, padding=1, bias=True)
  ReLU
  BatchNorm2d(100)

  Conv2d(100,100, kernel=3, stride=1, padding=1, bias=True)
  ReLU
  BatchNorm2d(100)

  Conv2d(100,100, kernel=1, stride=1, padding=0, bias=True)
  BatchNorm2d(100)
→ [1,100,H,W]
```

Key findings:

- `nConv=2` means **two 3×3 feature blocks plus one 1×1 response layer**, hence three convolutions.
- Increasing `nConv` adds 3×3 Conv/ReLU/BN blocks.
- `nChannel` controls every hidden width and the final number of response channels.
- All convolutions preserve spatial resolution.
- BN follows ReLU in feature blocks.
- The final BN follows the 1×1 convolution directly.
- No pooling, upsampling, dropout, residual connections, or explicit spatial-coordinate input exists.

`MyNet(input_dim)` already supports arbitrary input-channel counts. The **loader**, not the network constructor, makes ordinary direct execution three-channel. [MyNet, line 43](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L43)

Verified parameter counts:

| Configuration | Trainable parameters |
|---|---:|
| Default, three input channels | **103,600** |
| One input channel | **101,800** |

For RGB: convolution parameters are `2,800 + 90,100 + 10,100`; BN contributes `600`. Running statistics are buffers, not trainable parameters.

Initialization uses framework defaults; the imported `torch.nn.init` is never explicitly invoked. Modern Conv2d defaults initialize weights and biases using fan-in-dependent uniform bounds. Modern BN initializes affine scale to one and bias to zero. The repository does not pin the historical framework version, so exact historical initialization parity remains a validation requirement. [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv2d.html)

## 5. Default hyperparameters

These are literal direct-mode parser defaults:

| Setting | Code default | Source line |
|---|---:|---:|
| `scribble` | `False`, enabled by `store_true` | 18 |
| `nChannel` | `100` | 20 |
| `maxIter` | `1000` | 22 |
| `minLabels` | `3` | 24 |
| `lr` | `0.1` | 26 |
| `nConv` | `2` | 28 |
| `visualize` | `1` | 30 |
| `input` | Required; **no default path** | 32–33 |
| `stepsize_sim` | `1` | 34 |
| `stepsize_con` | `1` | 36 |
| `stepsize_scr` | `0.5` | 38 |

Source: [argument definitions, line 18](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L18). The README’s `./BSD500/101027.jpg` is an example, not a parser default.

Optimizer construction is exactly:

```python
optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
```

Therefore:

| Optimizer property | Behavior |
|---|---|
| Optimizer | SGD |
| Momentum | `0.9` |
| Weight decay | Default `0` |
| Dampening | Default `0` |
| Nesterov | Default `False` |
| Scheduler | None |
| Gradient clipping | None |
| Mixed precision | None |

Source: [optimizer construction, line 114](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L114).

## 6. Pseudo-label mechanism

The target is the channel index of the maximum **final BN-normalized response**:

```python
ignore, target = torch.max(output, 1)
```

At this point `output` is `[HW,C]`, so dimension `1` is the channel dimension. [Pseudo-label generation, line 129](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L129)

Consequences:

- BN precedes argmax.
- No softmax is required to obtain the maximizing channel.
- Labels are recomputed every iteration.
- There is no explicit `target.detach()`, but integer argmax indices are already nondifferentiable.
- `.data.cpu().numpy()` supplies counting and visualization; it is not the training target’s origin.
- The network retains all `C` channels even when only a few win argmax.
- Inactive channels are not deleted. They can still receive gradients through cross-entropy and continuity.
- Cluster counts need not decrease monotonically.
- There is no guarantee that each cluster is spatially connected.

The most accurate description is **iterative self-label clustering**, or **alternating hard self-label assignment and gradient optimization**.

An EM-like analogy is possible, but this is not a formal EM implementation. It is not K-means: there are no explicit centroids or centroid-update steps.

## 7. Losses

### Similarity loss

The direct mode defines:

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

Input: `[HW,C]` response logits.  
Target: `[HW]` current argmax indices.  
Reduction: default mean over pixels.

For response vector $r_p$ and current winning channel $y_p$:

$$
L_{\mathrm{sim}}
=
-\frac{1}{HW}\sum_p
\log\frac{\exp(r_{p,y_p})}{\sum_c\exp(r_{p,c})}.
$$

This reinforces the current assignments. Targets remain fixed during that backward pass.

### Spatial continuity

The code operates on the **continuous response tensor after final BN**, not RGB intensities, probabilities, or discrete labels:

```python
HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
```

Despite the variable names, both differences are in-plane:

$$
L_{\mathrm{row}}
=
\frac{\sum_{h,w,c}|r_{h+1,w,c}-r_{h,w,c}|}{(H-1)WC}
$$

$$
L_{\mathrm{col}}
=
\frac{\sum_{h,w,c}|r_{h,w+1,c}-r_{h,w,c}|}{H(W-1)C}.
$$

Each is `L1Loss(size_average=True)` against a zero tensor. The means are calculated separately, then added.

### Total direct objective

$$
L
=
\texttt{stepsize\_sim}\,L_{\mathrm{sim}}
+
\texttt{stepsize\_con}\,(L_{\mathrm{row}}+L_{\mathrm{col}}).
$$

The “stepsize” names denote **loss weights**, not optimizer learning rates. Defaults give both weights `1`.

Sources: [loss definitions, line 98](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L98), [differences and objective, line 123](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L123).

No additional direct-mode regularizer was found. In particular, there is no centroid loss, cluster-balance loss, fixed-cardinality penalty, or weight decay.

## 8. Stopping rule

The exact ordering is:

```text
forward
→ argmax
→ count unique labels
→ compute loss
→ backward
→ optimizer step
→ if previously counted nLabels <= minLabels: break
```

`nLabels` is measured **before the optimizer update**, although the condition is checked afterward. [Stopping rule, line 129](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L129)

`maxIter=1000` allows at most 1,000 updates. There is no convergence tolerance or patience criterion.

**Does `minLabels=4` force exactly four clusters? No.**

It does not constrain the response-channel count, prevent collapse, or enforce four connected regions.

For a `5 → 3` transition:

- If the current pre-update map has five labels, that iteration does not stop under MinL4.
- If its update causes the next forward to have three labels, the next iteration observes three.
- That next iteration still performs an optimizer update before stopping.
- Visualization output uses its three-label pre-update map.
- Headless output uses another forward after that last update and can have a different count.

If the transition occurs on the last permitted update, headless output can already have three labels without a subsequent stopping check.

Final output may have fewer than `minLabels`; it can also exceed the triggering count because no monotonicity constraint exists.

## 9. Final-output semantics

| Setting | Saved partition |
|---|---|
| `visualize=1` | Last loop’s **pre-update** pseudo-label map |
| `visualize=0` | Fresh forward **after the last optimizer update** |

The distinction is explicit in [the final-output branch, line 153](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L153).

Both paths retain training mode. Headless final forwarding therefore uses current-image BN statistics and updates BN running buffers once more.

**Recommended benchmark behavior:** match official `visualize=0`: fresh post-update forward, train-mode BN, then export integer argmax labels.

Using `torch.no_grad()` around that final forward is a reasonable wrapper optimization, subject to parity testing. Switching to `model.eval()` changes BN behavior and should not be introduced as an incidental cleanup.

Store both the stopping-map count and the final-map count.

## 10. Direct/transductive behavior

The direct script constructs one model and optimizer for one image. It saves no checkpoint and contains no multi-image loop.

The supported classification is:

**per-image, transductive, test-time optimization.**

For independent invocations:

- Model weights do not persist.
- Optimizer momentum does not persist.
- BN running buffers do not persist.
- No learned state is transferred between images.
- No explicit seed is persisted or reset by the script.

If a wrapper processes multiple images in one Python process, global RNG state naturally advances unless explicitly controlled. The wrapper must instantiate fresh model, optimizer, and BN state per image.

Optimizing on a held-out image’s intensities does not constitute GT access. It is valid under a declared transductive protocol. It must not be presented as ordinary frozen-model inference, and its optimization cost belongs in test-time compute.

## 11. Scribble mode

The annotation filename is derived from the input path by replacing its extension with `_scribble.png`; it is loaded using OpenCV’s unchanged mode. [Scribble loading, line 75](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L75)

Encoding:

- `255`: unlabeled pixel.
- Every other integer: supplied class/channel target.
- Unique non-255 values determine `args.minLabels`.
- Labels are passed through directly; there is no compact remapping.

The objective becomes:

$$
L =
\lambda_{\mathrm{sim}} CE(r_U,\hat y_U)
+
\lambda_{\mathrm{scr}} CE(r_S,y_S)
+
\lambda_{\mathrm{con}} L_{\mathrm{continuity}}.
$$

Here `U` denotes unlabeled pixels and `S` denotes scribbled pixels.

Unlabeled pixels contribute self-label loss. Scribbled pixels contribute annotation loss. Continuity covers the entire image. The two cross-entropies are separate means; weighting is not automatically proportional to subset size.

Additional limitations:

- Annotation values must be valid response-channel indices.
- Empty labeled/unlabeled subsets are not explicitly guarded.
- `np.int` is a legacy compatibility problem in this branch.

This mode is **weakly supervised**, with direct supervised loss on annotated pixels. It must be excluded from the primary strict no-label FreeMask comparison.

## 12. Reference-image mode

`demo_ref.py` trains one shared CNN over sorted files under `<input>/ref/*`. The number of reference images is unrestricted by the algorithm; the supplied BBC example contains **one reference image** and 11 test images.

Training details:

| Property | Actual behavior |
|---|---|
| Architecture | Same structure as direct DFC |
| Reference preprocessing | Resize every reference to `224×224`, BGR float32 `/255` |
| Epoch-like outer loop | `maxIter=1` |
| Updates per complete reference batch | `maxUpdate=1000` |
| Batch size | `1` |
| Similarity weight | `1` |
| Continuity weight | **`5`**, unlike direct mode |
| Optimizer | SGD, lr `0.1`, momentum `0.9` |
| Early stopping | None |
| `minLabels` argument | Parsed but unused |
| Incomplete final batch | Dropped by integer division |

Weights, BN affine parameters, BN buffers, and optimizer momentum persist across reference batches. Model `state_dict()` is saved after each outer iteration. [Reference training, line 72](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo_ref.py#L72)

For target files under `<input>/test/*`:

- Original target resolution is retained.
- The same in-memory model performs one forward per target.
- No target loss, backward pass, or optimizer update occurs.
- No labels or masks are loaded.
- **The model remains in training mode.**

Thus target BN uses each target’s spatial statistics and updates running buffers. Those buffers persist across target forwards, although train-mode outputs use current statistics rather than those accumulated buffers. [Reference inference, line 135](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo_ref.py#L135)

Classification: image-only **reference-set training and transfer**, potentially few-reference or dataset-level depending on the supplied set, with target-dependent BN normalization. It is not direct per-image optimization.

Recommendation: optional secondary/P2 experiment, separately named and with reference provenance specified. Exclude it from the primary direct profile.

## 13. GT/oracle audit

The complete DFC source contains no Dice, IoU, Hungarian matching, accuracy computation, oracle permutation, best-cluster selection, or GT evaluation implementation.

| Operation | Classification |
|---|---|
| Direct image loading | IMAGE-ONLY |
| Argmax pseudo-label generation | IMAGE-ONLY |
| Similarity and continuity optimization | IMAGE-ONLY |
| Unique-label stopping | IMAGE-ONLY |
| Random visualization palette | IMAGE-ONLY |
| Reference-image training | IMAGE-ONLY, assuming reference inputs are images |
| Target BN normalization in reference mode | IMAGE-ONLY |
| Scribble loading and annotation loss | WEAKLY SUPERVISED |
| Scribble-derived `minLabels` | WEAKLY SUPERVISED |
| Proposed isolated evaluator reading masks after freeze | GT-ACCESSIBLE |
| Proposed preprocessing/selection influenced by GT | GT-ASSISTED; prohibited |
| Proposed per-case GT-based cluster permutation or best-cluster choice | ORACLE; not a primary semantic result |

**Vanilla `demo.py` does not need GT.** Its variable `target` denotes a pseudo-label, not ground truth.

The reference-image script does not implement reference supervision; it self-labels reference images.

## 14. Randomness and reproducibility

| Source | Audited behavior |
|---|---|
| Python `random` | Imported, unused |
| NumPy RNG | Random visualization colors |
| PyTorch CPU RNG | Model initialization |
| PyTorch CUDA RNG | No explicit random CUDA operation in the direct loop |
| CUDA execution | Determinism settings unspecified |
| DataLoader | None |
| OpenCV | No stochastic augmentation or randomized algorithm |
| Explicit seeds | None |

The model is initialized on CPU before `.cuda()`, so CPU RNG controls its initial weights.

Recommended policy:

```text
benchmark_seed = 42
sample_seed = stable derivation from (42, canonical sample_id)
```

Use a versioned cryptographic-hash derivation into a supported integer seed range, rather than Python’s randomized `hash()` or the sample’s manifest row number. Include dataset, patient, frame, and slice identity in the canonical ID.

Prefer this over resetting every image to seed 42 because it gives deterministic, distinct streams while remaining independent of processing order, worker assignment, or retries. Reuse each sample’s seed across matched profile comparisons where appropriate; different input-channel counts still change initialization draws.

Seed Python, NumPy, CPU PyTorch, and CUDA; specify deterministic backend settings, precision/TF32 policy, and software/hardware provenance. Keep palette randomness separate from scientific artifacts.

Bitwise reproducibility should be claimed only for the validated environment. PyTorch does not guarantee identical results across releases or platforms. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

## 15. Preprocessing and normalization

Original direct preprocessing is precisely:

```text
OpenCV default color read
→ BGR [H,W,3]
→ CHW
→ float32
→ divide by 255
→ add batch dimension
```

There is:

- No BGR-to-RGB conversion.
- No mean/std normalization.
- No resize or crop.
- No intensity clipping beyond image decoding.
- No augmentation.
- No MRI-aware loading or geometry handling.

Source: [direct preprocessing, line 68](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L68).

For MRI, recommend a documented image-only normalization aligned with the benchmark’s common normalization recipe:

1. Read physical image intensities as float32.
2. Use full-field image values; no mask or ROI.
3. Clip at fixed `0.5/99.5` image percentiles.
4. Subtract clipped mean and divide by population standard deviation, with a fixed floor.
5. Apply the frozen shared grid transformation.

This matches the **kind** of normalization used by current FreeMask; its full-input deployment path computes these statistics across the three-slice stack and performs area-based resizing. [FreeMask dataset normalization and full-input loader](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/data/dataset.py#L148)

For strictly 2D DFC, calculate statistics from the central slice only. For 2.5D, use a declared common transform across the supplied stack.

**Fairness qualification:** central-only statistics and FreeMask’s stack statistics are not byte-identical preprocessing. Freeze and disclose this input-context distinction. Do not silently describe them as identical, or import FreeMask’s withheld-observation normalization into direct DFC.

A constant slice requires a predefined denominator floor and accounting; it must not be silently dropped.

## 16. Cardiac 2D adaptation

Recommended input:

```text
central MRI slice [1,H,W]
→ batch [1,1,H,W]
→ MyNet(input_dim=1)
```

No architecture rewrite is required. The constructor already accepts one channel. Relative to RGB, only the first convolution’s input dimension and parameter count change.

The surrounding wrapper must replace assumptions involving `im.shape`, color visualization, and image loading. The response maps, BN, loss, optimizer, argmax, and stopping behavior remain applicable.

Do not set `nChannel=4` to represent cardiac classes. Its default 100 response channels are part of the clustering mechanism.

Describe this as a **single-channel MRI input adaptation of direct DFC**.

## 17. Cardiac 2.5D adaptation

The input can be:

```text
[z-1,z,z+1] → [3,H,W] → [1,3,H,W]
```

This uses the original three-channel parameter shapes. All convolutions remain Conv2d, and continuity remains adjacent-row/column response differences.

Define:

- Central slice as the prediction target.
- Neighbor order explicitly.
- Neighbors from the same patient and frame.
- Boundary replication, matching FreeMask’s neighbor reader.
- Identical field of view and grid across channels.

This is an **input-context adaptation**, not 3D DFC. It does not introduce a cross-slice loss or guarantee consistency between independently optimized central slices.

FreeMask’s reader explicitly clips neighbor indices at the volume boundaries. [FreeMask geometry reader](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/data/geometry.py)

## 18. MinL3 vs MinL4 decision

**Recommend MinL3 as primary.**

| Profile | Recommended role |
|---|---|
| `DFC-Direct-2D-Default-MinL3` | Primary |
| `DFC-Direct-2D-MinL4` | Task-cardinality sensitivity |
| `DFC-Direct-2.5D-MinL3` | Input-context sensitivity |
| `DFC-Direct-2.5D-MinL4` | Optional combined sensitivity |

Reasons:

- MinL3 is the verified official default.
- MinL4 does not enforce four clusters or four anatomical classes.
- Four semantic roles are task knowledge, but not evidence that a stopping threshold of four is preferable.
- Changing context and stopping threshold together would confound the sensitivity comparison.
- A default-preserving primary is defensible without evaluating Dice.

There is a genuine semantic mismatch: a three-cluster partition may be unable to represent BG/RV/MYO/LV separately under a cluster-to-class adapter. Report that limitation. An adapter must not secretly create missing boundaries to conceal it.

MinL4 is scientifically reasonable as a preregistered sensitivity, with no GT-based selection between profiles.

## 19. Raw output contract

DFC core should return an anonymous integer map `[H,W]`. Recommended artifact fields:

| Group | Required contents |
|---|---|
| Identity | `dataset`, `patient_id`, `frame`, `slice`, `sample_id`, `split`, manifest ID/hash |
| Partition | `raw_cluster_map`, shape, dtype, map checksum |
| Randomness | `benchmark_seed`, `sample_seed`, seed-derivation version |
| Configuration | Resolved parameters, canonical config hash, profile ID |
| Provenance | Official repository SHA, wrapper SHA, environment identifier |
| Iterations | Number of optimizer updates; forward count |
| Stopping | `stop_reason`, last pre-update active count and IDs |
| Final output | Final active count and IDs; final-forward policy; BN mode |
| Geometry | Source identity, axis order, affine/spacing validity, transform record |
| Compute | Synchronized elapsed time, peak allocated/reserved VRAM, GPU identity |
| Accounting | Success/failure status, error information, retry history |

`stop_reason` should distinguish threshold completion, iteration cap, and failure. If the threshold condition fires on the final allowed iteration, give it the same priority as the executable condition.

Preserve original channel IDs; any compact relabeling must be deterministic and documented. Use lossless integer storage. A random-color PNG is not the scientific raw artifact.

## 20. Shared adapter boundary

The proposed boundary is scientifically appropriate:

```text
image
→ DFC
→ frozen anonymous partition
→ shared cardiac_adapter_v1
→ BG/RV/MYO/LV/VOID
→ frozen semantic artifact
→ isolated GT evaluator
```

Conditions:

- The adapter is versioned, shared, and GT-independent.
- It is invariant to arbitrary permutation of cluster IDs.
- Any image or geometry inputs it needs are explicitly permitted and shared.
- It declares whether it only names/merges clusters or can also split them.
- Unresolved cases remain visible as VOID and in coverage statistics.
- Its rules are frozen before test evaluation.

Do not import FreeMask’s auditor, predictive evidence, candidate challenges, or `O_fit/O_select/O_verify` roles into DFC.

The inspected FreeMask code contains method-specific role resolution and candidate auditing. That does not establish that the proposed independent `cardiac_adapter_v1` is already implemented or validated.

## 21. Slice-to-volume assembly

Neither DFC script implements inter-slice processing.

After external semantic adaptation, maps can be ordered by manifest slice index and stacked into `[Z,H,W]` per frame.

Call this **semantic slice assembly** or **stacking**.

Requirements include explicit axis ordering, missing-slice accounting, valid inverse grid transforms, and geometry provenance. Do not infer physical geometry for containers lacking it.

Stacking alone provides no 3D consistency. Smoothing, voting, inter-slice relabeling, and connected-component filtering across slices would be additional postprocessing.

## 22. ACDC/M&Ms protocol

Use one authoritative manifest shared with FreeMask:

- Same patient identities and splits.
- `split_seed=42`.
- Same central target-slice inventory.
- Same frame policy.
- Same source field of view and output grid.
- Same geometry metadata and source fingerprints.
- Same evaluation eligibility and failure accounting.

Current FreeMask discovery enumerates all slice indices and, for multiframe acquisitions, all acquired frames. “Central slice” means the target slice of a context stack; it does not mean only mid-ventricular slices. [FreeMask discovery](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/data/discovery.py#L1231)

Its split code is patient-based, defaults to 70/15/15 where applicable, and preserves declared official cohort information according to dataset-specific rules. Reuse the frozen manifest rather than independently reconstructing a split from seed 42.

Its firewall forbids annotation sidecars such as ACDC `Info.cfg` for image-only cohort discovery. An ED/ES-only source must retain its annotation-selected cohort limitation. [FreeMask firewall](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/data/firewall.py)

DFC Track A needs no dataset-level train checkpoint or dev checkpoint selection. Split identity remains necessary for fair evaluation, accounting, and future Track B student training.

The actual frozen data manifest and its inventory size were not supplied or verified in this audit.

## 23. Compute and ≤72h feasibility

For `224×224`, default `C=100`, `nConv=2`:

| Quantity | RGB / 2.5D | One-channel 2D |
|---|---:|---:|
| Parameters | 103,600 | 101,800 |
| Convolution forward MACs | 5.153 billion | 5.063 billion |
| Forward FLOPs, counting multiply and add separately | 10.31 GFLOPs | 10.13 GFLOPs |
| Approximate forward + backward convolution work/update | Around 30 GFLOPs | Around 30 GFLOPs |
| Approximate work at 1,000 updates | Around 30 TFLOPs/image | Around 30 TFLOPs/image |

These are analytical estimates, excluding BN, activation, loss, memory movement, and launch overhead. They are not measured throughput.

A `[1,100,224,224]` float32 response occupies about **19.1 MiB**. Training holds multiple activations, differences, zero targets, gradients, and backend workspaces. Parameters and momentum are comparatively tiny.

One image should be comfortably VRAM-safe on an otherwise available:

- RTX 4070 12GB.
- RTX 4080 16GB.
- RTX 5070 Ti 16GB, with compatible software.

Expected memory is in the sub-GB-to-few-GB range rather than near 12GB, but actual peak allocation and process memory must be measured.

**Seconds per image are unmeasured.** The local machine reports an RTX 2050 4GB; no timing experiment was run.

For perspective, these are budget scenarios, not performance predictions:

| Measured mean per slice | Serial slices fitting in 72h, before overhead |
|---:|---:|
| 10 seconds | 25,920 |
| 30 seconds | 8,640 |
| 60 seconds | 4,320 |

Required preflight:

```text
50–100 fixed, image-only selected slices
→ full configured optimization
→ mean/p50/p95/max time, iteration distribution, peak VRAM
→ measured concurrency throughput
→ projection over the exact full inventory and profiles
```

Use GPU synchronization for timing. Separate startup, loading, optimization, final forward, and export. Include failures, retries, adapter/evaluator time, and an operating margin.

Independent images can run on separate GPUs or isolated workers. Do not batch them through a shared network/BN as a throughput shortcut: that changes the method.

A ≤72h run is plausible, but cannot be claimed without inventory size and measured throughput. Do not reduce core iteration limits or losses merely to satisfy the budget.

## 24. Environment

The README lists only `pytorch, opencv2, tqdm`. Both scripts also import NumPy and torchvision; torchvision is unused but still an import dependency. There is no pinned environment. [Requirements, line 22](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/README.md#L22)

Recommend a dedicated `dfc-cardiac` environment.

For 4070/4080, a candidate is:

```text
Python 3.10
PyTorch 2.5.1 + CUDA 12.1 runtime
torchvision 0.20.1, if retaining the original import
NumPy 1.26.4
pinned OpenCV, tqdm, and MRI I/O dependencies
```

The PyTorch CUDA 12.1 distribution is documented. [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)

For 5070 Ti, use a tested Blackwell-compatible build, such as a pinned PyTorch 2.7-or-later CUDA 12.8 build. CUDA 12.1 should not be the proposed common runtime for that GPU. PyTorch introduced Blackwell support with its 2.7/CUDA 12.8 release. [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)

Prefer one validated build across benchmark GPUs where practical.

Compatibility concerns:

- `np.int` breaks the scribble branch with modern NumPy.
- `size_average=True` is legacy loss syntax.
- `Variable` and `.data` are legacy idioms.
- Unused torchvision imports can cause avoidable installation mismatches.
- Default GUI visualization conflicts with headless OpenCV.
- Float precision/backend choices can change iterative argmax trajectories.

The current local Python is 3.14 with PyTorch `2.9.0+cu128`; this is an observation, not a benchmark environment endorsement.

Before scientific execution, require architecture, initialization, one-step, loss, stopping, and final-partition parity against the pinned executable reference on deterministic fixtures.

## 25. Protected core code

Keep the original files unchanged as the provenance reference. Protect these semantics in the benchmark wrapper:

| File / component | Protected behavior |
|---|---|
| `demo.py::MyNet.__init__` | Width, depth, convolution settings, biases, BN construction |
| `demo.py::MyNet.forward` | Conv → ReLU → BN ordering; final Conv → BN |
| Top-level loop, lines 119–131 | Forward, flattening, per-channel argmax, unique-label count |
| Lines 99, 105–106, 123–142 | Cross-entropy and separate mean-L1 continuity terms |
| Line 114 | SGD learning rate and momentum semantics |
| Lines 144–151 | Update before checking the pre-update label count |
| Lines 154–158 | Fresh train-mode final forward for headless output |

The one-channel input dimension is a declared adaptation already supported by the constructor.

Source anchors: [model, line 43](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L43), [optimization, line 114](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L114), [final forwarding, line 154](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/181318ad40dbfb5c0add8580a05c30e5e3a7ad58/demo.py#L154).

There is no reusable `train()` or `main()` function: training is top-level script code, and importing the module runs argument parsing and execution. A wrapper must account for that.

## 26. Files to wrap/replace

| Existing component | Problem | Replacement boundary | Scientific impact |
|---|---|---|---|
| `demo.py` lines 69–73 | BGR/8-bit assumptions; no MRI geometry | Image-only manifest loader producing `[1,C,H,W]` | Declared modality/preprocessing adaptation |
| `im.shape` usage in losses/output | Tied to color image container | Explicit `H,W` from tensor | Neutral if dimensions match |
| Lines 132–136 | GUI calls and delay | Optional external preview | Operational; preserve chosen final-forward policy |
| Lines 115, 133–134, 159–161 | Random colors; no raw labels; fixed output filename | Lossless integer partition plus metadata | Improves artifact fidelity |
| Scribble branch | Annotation access | Excluded from primary runner | Enforces no-label protocol |
| `demo_ref.py` | Shared reference training and different defaults | Separate optional profile | Different scientific setting |
| Global argparse/execution | Unsafe import and legacy CLI coupling | Isolated runner interface | Neutral if parity holds |
| `.cuda()` auto-selection | Implicit GPU choice | Explicit worker/device configuration | Record device and precision |
| No seed/provenance | Irreproducible initialization and accounting | Deterministic per-sample execution metadata | Declared reproducibility policy |
| Fixed `output.png` | Overwrites/concurrency collisions | Sample-addressed artifact paths | Operational |

The palette contains only 100 entries even when `nChannel` is increased. Thus nondefault `nChannel>100` can fail during visualization/export. Exporting raw labels avoids this non-algorithmic limitation.

## 27. Leakage/fairness risks

| Severity | Risk |
|---|---|
| CRITICAL | GT-derived crop, ROI, intensity statistics, or anatomical slice selection |
| CRITICAL | Enabling scribble supervision in the no-label primary |
| CRITICAL | GT-based permutation/best-cluster mapping presented as ordinary semantics |
| CRITICAL | Choosing `minLabels`, `maxIter`, seeds, or adapter rules using test Dice |
| CRITICAL | Allowing masks into optimization or semantic adaptation |
| HIGH | Different patient/frame/slice inventory from FreeMask |
| HIGH | Silently dropping slow, failed, empty, or anatomically difficult slices |
| HIGH | Treating arbitrary cluster IDs as BG/RV/MYO/LV |
| HIGH | Undisclosed test-time optimization or omitted compute |
| HIGH | Sharing weights, optimizer, or BN state between direct samples |
| HIGH | Switching final BN to eval mode or exporting the wrong iteration |
| HIGH | Undisclosed preprocessing or field-of-view differences |
| HIGH | Adapter silently reconstructing missing boundaries or using FreeMask selection |
| MEDIUM | Nondeterministic kernels, unpinned versions, or order-dependent seeds |
| MEDIUM | Claiming native geometry or 3D consistency without evidence |
| MEDIUM | Reporting favorable metrics only on non-VOID pixels without coverage |
| LOW | Random visualization differences, provided raw labels are authoritative |

## 28. P0/P1/P2 proposal

These are audit-level responsibility and acceptance groupings, not an implementation plan.

| Phase | DFC-specific scope |
|---|---|
| **P0 — trustworthy raw partitions** | Frozen shared manifest/data contract; dedicated environment; image-only MRI loader; reproducible independent per-image optimization; official headless semantics; raw partition export; provenance/freeze; compute preflight |
| **P1 — primary semantic benchmark** | Shared `cardiac_adapter_v1`; semantic output and VOID accounting; slice stacking; isolated GT evaluation; primary MinL3 result |
| **P2 — controlled extensions** | MinL4 sensitivity; 2.5D MinL3 sensitivity; optional combined profile; reference-image transfer variant; multiseed robustness; CommonStudent pseudo-label utility |

For CommonStudent/Track B, training pseudo-labels must follow the declared training manifest. Availability of DFC test-image partitions does not authorize their use for student training.

## 29. Tests and acceptance gates

These are proposed gates; they were not executed during this audit.

| Test | PASS criterion | FAIL criterion | Phase |
|---|---|---|---|
| Architecture parity | Same modules/settings and 103,600 RGB / 101,800 one-channel parameters | Changed depth, ordering, bias, BN, or count | P0 |
| Initialization parity | Same seed/environment produces matching initial reference state | Different initialization or RNG consumption without declaration | P0 |
| One-step optimization | Matched input/state yields matching loss, gradients, weights, momentum, BN buffers within frozen tolerances | Any unexplained discrepancy | P0 |
| Loss parity | CE and both separately averaged L1 terms match reference and hand-checkable fixtures | Wrong axes, scaling, representation, or reduction | P0 |
| Stopping parity | Correct update/check ordering for equality, initial below-threshold, `5→3`, and iteration cap | Exact-K enforcement, early check, or off-by-one update | P0 |
| Final-forward parity | Headless output matches fresh post-update train-mode argmax | Stale pre-update map or eval-mode BN | P0 |
| Determinism | Same sample/config/environment reproduces exact integer partition and iteration count | Unexplained repeat differences | P0 |
| Per-image independence | Sample result unchanged by processing order, unrelated prior samples, or worker assignment | Shared learned state or order-dependent seed | P0 |
| GT firewall | Core/adapter operate with annotation access denied; GT mutation leaves outputs unchanged | Annotation read or output dependence | P0/P1 |
| Scribble exclusion | Primary runner cannot activate or load scribbles | Hidden annotation branch reachable | P0 |
| Reference-mode exclusion | Every primary sample begins with fresh model and optimizer | Checkpoint/reference initialization | P0 |
| Cluster-ID invariance | Permuting anonymous IDs leaves semantic output unchanged | Adapter depends on numerical IDs | P1 |
| Raw artifact reproducibility | Map/config/provenance checksums repeat; volatile timing fields excluded | Scientific payload changes or provenance absent | P0 |
| Failure accounting | Every manifest sample has exactly one explicit terminal record | Missing samples or silent substitutions | P0/P1 |
| Runtime logging | Synchronized timings, update count, device, and defined VRAM metrics present | Asynchronous/partial or missing timing | P0 |
| Assembly/geometry | Round-trip fixtures preserve slice/frame order and labels | Axis swaps, missing slices, fabricated geometry | P1 |
| Freeze/evaluation isolation | Semantic artifacts immutable before evaluator accesses GT | Feedback changes predictions or configuration | P1 |

Cross-environment floating-point tolerances must be declared before validation. Exact final-partition parity on fixed fixtures is a separate gate; similar scalar losses alone do not establish it.

## 30. Recommended primary DFC benchmark architecture

Recommended architecture:

```text
Frozen shared image manifest
→ image-only central-slice loader
→ declared normalization and shared 224×224 grid
→ fresh MyNet(input_dim=1), sample-specific deterministic seed
→ official direct DFC optimization, MinL3, maxIter=1000
→ fresh train-mode post-update forward
→ anonymous integer partition + provenance
→ shared cardiac_adapter_v1
→ semantic slice assembly
→ frozen artifacts
→ isolated GT evaluator
```

Preserve the scientific distinction between methods:

| Method | Scientific role |
|---|---|
| CUTS, as specified for this comparison | Dataset-level SSL representation → PHATE/KMeans |
| DFC | Per-image optimization → iterative self-label clustering |
| FreeMask | Representation → candidate partitions → predictive evidence → challenge/selection |

FreeMask’s inspected auditor fits candidates on fitting observations and evaluates challengers using withheld selection observations. Those mechanisms belong to FreeMask’s method and should not enter DFC. [FreeMask auditor](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/auditor.py#L1)

### Explicit final answers

1. **What should be the PRIMARY DFC profile?**  
   `DFC-Direct-2D-Default-MinL3`, single-channel input, official optimization defaults, deterministic per-sample seed, official headless final-forward semantics.

2. **Should MinL3 or MinL4 be primary?**  
   **MinL3.** MinL4 is a preregistered task-cardinality sensitivity and does not force four clusters.

3. **Should 2.5D be primary or sensitivity only?**  
   **Sensitivity only**, initially paired with MinL3 to isolate input context.

4. **Is direct DFC scientifically valid as a transductive test-time baseline?**  
   **Yes**, with image-only optimization, independent per-image state, explicit disclosure, and full compute accounting.

5. **Does vanilla DFC use GT?**  
   **No.** Scribble mode uses annotations and must be excluded.

6. **What code must not be changed?**  
   Core architecture/order, BN behavior, argmax self-labeling, loss definitions/reductions, SGD semantics, update-before-stop ordering, and headless final-forward semantics.

7. **What code must be wrapped/replaced?**  
   Color-image loading, GUI, random-color-only output, legacy CLI/device assumptions, and missing reproducibility/provenance/accounting. Scribble and reference modes must remain outside the primary runner.

8. **Can the full benchmark plausibly fit within 72 hours?**  
   **Plausibly, but unverified.** VRAM is unlikely to be the limiting factor at one 224×224 image; inventory size and iterative throughput determine feasibility.

9. **What must be measured before claiming that?**  
   Exact slice/profile inventory; fixed 50–100-slice mean/p95/max runtime; iteration distribution; peak VRAM; safe concurrent throughput; startup/export/adapter/evaluation overhead; failure/retry allowance.

10. **Is DFC ready for an `IMPLEMENTATION_PLAN.md`?**  
    **Yes, at the architecture level.** A subsequent plan should resolve the frozen manifest, normalization contract, adapter specification, GPU-compatible environment, and parity/preflight gates. DFC is not yet benchmark-ready. No implementation plan or code was created in this audit.
