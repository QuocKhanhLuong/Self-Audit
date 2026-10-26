# Independent Astra review: architecture, loss, optimizer, trainer and provenance

Status: **ACCEPTED for the declared CPU/software contracts at the final pinned snapshot.** All model/trainer findings, including the late joint batched-cache/LRU resume case, are resolved. The coordinator's final post-CLI gate passes 422 tests and check_repo; its receipt and pinned files were independently inspected. Earlier provisional holds and lower test counts below are historical; the final disposition at the end is authoritative. Only this report was edited by this reviewer. Real patient, CUDA and actual DINO-asset evidence remain NOT RUN.

- Task: `task_7d023d5190f0`; dispatch: `ctx_c12c99de1f65`; report delivered through Orca only.
- Base: `486fae40c490e382d58896442215476d7d32095e`, branch `feat/native-volume-ema-pilot`.
- Read full `docs/native_3d/ASTRA_IMPLEMENTATION_PROMPT.md`, `IMPLEMENTATION_CONTRACT.md`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md`, model/fusion/loss/optimizer/config/engine/checkpoint/evaluation/legacy compatibility modules, router, evaluation/export CLIs and associated tests. Region/cache code was inspected at the integration boundary.
- Environment: `/Users/alvinluong/miniforge3/bin/python`, Python 3.11, Torch 2.14.0, CUDA unavailable. No server rented or accessed, no real TRAIN/VAL/TEST masks or external model assets loaded, no downloads or long training.
- Historical A0 is fixed-window/ema_full/L8/batch24/crop256/FP32. Current canonical model/loss/engine/window/data bytes were independently compared with HEAD and remain identical. Historical artifacts were not modified.

## Initial blocking findings (sent to original owners through coordinator)

1. **Volume Dice has the wrong spatial reduction.** `_volume_mean` reduced `(0,2,3)` on B,C,Z,H,W, leaving W as a hidden class sub-index. Float64 seed11, shape(2,4,3,4,5), random classes and mixed valid/255 support: implementation2.4610814593419357 vs independent complete-volume reference2.4547505447307865; max gradient error0.0010500481381574035. Present/absent counts were29/-23 for two volumes. Fix all five-dimensional reduction axes and independently compare values/gradients; existing microbatch test compared the same implementation to itself.
2. **All-layer zero initialization kills nonlinear controls.** `CrossZResidual(planar_control,zero_init=True)` had every affine layer zero; after three SGD steps all weight tensors remained exactly zero. Hybrid delayed and matched-additive zero-init skip paths had zero gradients for every weight. Zero only the terminal residual projection, preserve old-function initialization, then test input-dependent learning after updates. Initial default-init tests allowed zero gradients because they checked only `isfinite`.
3. **Zero pseudo weight still steps AdamW.** Toy Conv3d seed2, threshold0, pseudo_weight0: supervised L1 displacement0.04800010845 followed by pseudo displacement0.03216287866 despite zero pseudo CE and gradient. Momentum/decay changed weights and pseudo_updates incremented. Unsupported weighted objectives must skip optimizer state/decay.
4. **Scheduler warmup is one update late.** Same probe with constant LR/warmup4: both first-iteration steps used headLR0.003, whereas the scheduler factor requires0.00075. Apply the outer-step schedule before both committed updates and check actual optimizer rates, including skipped transactions.
5. **Auxiliary accumulation changes the objective.** Whole chunk pseudo+aux loss is weighted by pseudo accepted voxels (fallback1). Temporal losses normalize valid temporal voxels; regions normalize volumes. Independent seed22 Toy Conv3d, B2, volume_mean, threshold1, identity temporal weight2, valid supports24 and2: accumulation1 loss0.00860642269/gradnorm0.02638328075 vs accumulation2 loss0.00949495472/gradnorm0.05059871078, max gradient discrepancy0.02013186365. Each loss needs its own logical numerator/denominator and mixed-support gradient oracle.
6. **Region cache crosses patient/microbatch identity.** `current_hash` was computed but ignored by cache reuse. EMA manager refresh5, patientA class1 at step0 then patientB class2 at step1 with different image/spacing returned the exact same target object, hash and class1. Same-shaped microbatches within one step also collide. Require bounded identity-keyed cache with image/support/geometry/config provenance and explicit teacher refresh policy; missing physical metadata must not default to isotropic1mm.
7. **Source digest omits executed dependencies.** Checkpoint/export digest initially covered only `self_audit/volume`, excluding imported canonical model/window and router. Store and verify the complete executed dependency closure; config digest and mandatory manifest pin must also be validated.
8. **Historical compatibility is not operational yet.** Legacy wrapper verifies selected metadata then delegates to the canonical source-digest loader, which rejects historical manifests after router changes. CLI verification proceeds into fresh training, not checkpoint resume. Provide an explicit verified historical loader command/path with executable regression and no source-digest override, or fail clearly and disclose unsupported historical execution.
9. **Evaluation estimator/native export gaps.** Initial `evaluate_frozen` pools all voxels across batches before Dice, does not invert probabilities to native grids or aggregate frames/patients, and labels absent/empty support with smoothed perfect Dice. Real frozen evaluation CLI unconditionally raises; synthetic path uses manufactured background labels. A functional hash-pinned held-out path and asymmetric native-grid/patient-average fixtures are required.
10. **Model export depends on initialization assets.** Loading DINO exports calls original pretrained loading again; checkpoint state alone cannot deploy. Correction to early message: `synthetic=True` does not replace configured DINO with Tiny because the canonical loader chooses by backbone. Export must record true architecture/source provenance and offer an asset-independent architecture-only construction path or an explicit documented dependency policy verified with blocked teacher/cache/network access.
11. **Ignored/rewritten configuration factors.** Initial hybrid width/kernels/strides, CNN width-list tail/skip options, and several architecture combinations were silently ignored. `scheduler_only` rewrites to grouped AdamW and `amp_bf16` rewrites to FP16 AMP. Fixed-window parameters beyond mode were rejected by shallow validation. Enforce every accepted field or reject unsupported combinations without changing scientific intent.
12. **Registry matching gaps.** New arms initially disable fixed windows; V05 changes both depth and skip fusion relative to V04. Initial synthetic new-branch parameter counts6176 vs10742 (+73.9%; whole-model +3.48%) must be disclosed and controls tightened. Block-level MAC sums are not total native-voxel FLOPs across varying feature resolutions.
13. **CLI/run/resume gate gaps.** `--steps` overrides are applied after checking config budget, while `train_volume` infers full-run permission from requested steps; `--model-config` is ignored in the volume branch. Real resumed data/geometry hashes are not checked against checkpoint data identity. Synthetic `train_volume` initially ignores logical batch size and supplies no U, so EMA/aux arms are not actually smoke-exercised.
14. **G6 diagnostic semantics need qualification.** `set_frozen_intervention` only detaches/freezes parameters and leaves frozen inference logits unchanged. Native decoder/detail bypasses remain live around the extra skip residual, so stop-gradient affects only that added branch. An actual branch-reliance perturbation and explicit scope/gradient tests are needed if full G6 completion is claimed.

15. **Precision can silently change and overflow.** CPU `VolumeTrainer(precision="amp")` accepted but used FP32 nullcontext/scalerNone. FP16 logits(1,4,4,128,128), all class1, produced NaN in both loss modes at65536voxels. CUDA autocast CE promotion does not automatically promote manually reduced Dice; force stable accumulation and test finite gradients, while rejecting unsupported device/dtype combinations.
16. **Required diagnostics are placeholders/omitted.** Teacher-student disagreement initially logs NaN, per-class acceptance is absent, and actual update norms returned by the optimizer helper are discarded by the trainer. Auxiliary-only gradient evidence must be isolated from CE rather than relabeling the total gradient norm.

Additional evaluation repro: two one-frame patients with foreground support1 vs9 and perfect vs entirely wrong predictions return pooled class1 Dice0.18181818256, whereas the required patient mean is0.5. All-empty evaluation returns perfect smoothed Dice1 for every class and no CE; empty support must report unavailable/absent support explicitly.

17. **Resume completeness and device handling.** Initial payload loads with map_location=device, then sends its Torch RNG tensor directly to `torch.set_rng_state`; on CUDA that tensor would be on the wrong device (source review, GPU NOT RUN). Restore CPU RNG from CPU bytes. Missing RNG/trainer/scaler/region fields default silently instead of rejecting an incomplete enabled-factor checkpoint. Require strict versioned state completeness and duplicate counter/config consistency.

## Independent positive checks and limits

- Initial `rtk proxy python -m pytest -q tests/test_volume_losses.py tests/test_volume_optim.py tests/test_volume_models.py tests/test_volume_fusion.py`: **23 passed in1.54s**. This did not detect findings1–2 and is not sufficient acceptance.
- Initial `rtk proxy python scripts/check_repo.py`: **PASS**.
- Initial `rtk proxy python -m pytest -q` during edits: **262 passed,3 failed,1 warning in24.72s**. Failures: R01 CLI fixed-window mismatch, explicit zero-init test disabled window, planar-control default depth kernel mismatch. Concurrent source changes prevent interpreting this as a stable snapshot.
- Independent next-step CPU resume (Toy Conv3d, EMA regions with refresh3, cosine schedule/warmup2) matched uninterrupted student, EMA, optimizer, scheduler, region cache, sampler state and Python/NumPy/Torch RNG bit-exactly after another transaction. Counts(outer,supervised,pseudo,EMA)=(2,2,2,2). This does not repair cache identity semantics and is not GPU resume evidence.
- Fixed-window hybrid mix zero-init reproduced unchanged per-slice NativeStudent exactly and preserved native initialization plus global RNG. On activated depth weights, perturbing raw z0 changed center z3 beyond the original triplet by max0.00091731548. Off/planar recheck interrupted by in-progress constructor edit (StopIteration on off Identity terminal-projection selection); await stable owner checkpoint.

## Owner-ready rechecks

- Loss-axis correction independently verified at `self_audit/volume/losses.py` SHA256 `489552f683a35fa9665dd3aaeee486f47a94c5a7ffdf5802e8ba26073883270d`: original counterexample now matches flattened-volume oracle2.4547505447307865 with max gradient discrepancy3.469446951953614e-18 and presence6/absence0. `rtk proxy python -m pytest -q tests/test_volume_losses.py tests/test_volume_optim.py`: **15 passed in1.52s**. This resolves finding1 only; FP16 support remains pending.
- Matched-comparison followup: original ordinary pseudo step executes `model.eval()` on the strong intensity view; initial volume trainer executes in train mode on identical teacher/student inputs. Require declared same-grid weak/strong intensity handling or separately named factor plus stochastic-mode tests before claiming grouping-only V01.

## Final verification and source hashes

The final stable-snapshot receipt and source hashes are at the end of this report. CPU/manufactured checks cannot establish real patient accuracy, GPU precision/throughput, scientific superiority or novelty.

## Reproduction harness: independent mathematical reference

Run from repository root with `rtk proxy python -` and the following stdin. It requires no project edits or assets. This was executed on the initial snapshot and again on the loss-owner-ready snapshot; it exposes the original dimensional reduction defect independently of the production loss helper.

```python
import torch
from torch.nn import functional as F
from self_audit.volume.losses import supervised_volume_loss
torch.set_num_threads(1)
torch.manual_seed(11)
x = torch.randn(2,4,3,4,5,dtype=torch.double,requires_grad=True)
y = torch.randint(0,4,(2,3,4,5)); y[0,0,0,0] = 255
geom = torch.ones_like(y,dtype=torch.bool); geom[1,-1,-1,-1] = False
safe = y.masked_fill(~geom,255)
actual, metrics = supervised_volume_loss(x,y,geom,geom,mode="volume_mean",return_metrics=True)
terms = []
for i in range(2):
    q = x[i].flatten(1).T
    labels = safe[i].flatten()
    p = q.softmax(-1)[labels != 255]
    target = F.one_hot(labels[labels != 255],4).to(x)
    dice = (2*(p*target).sum(0)+1e-5)/(p.sum(0)+target.sum(0)+1e-5)
    present = target.sum(0)[1:] > 0
    terms.append(F.cross_entropy(q,labels,ignore_index=255)+(1-dice[1:][present]).mean())
reference = torch.stack(terms).mean()
ga = torch.autograd.grad(actual,x,retain_graph=True)[0]
gb = torch.autograd.grad(reference,x)[0]
print(float(actual.detach()),float(reference.detach()),float((ga-gb).abs().max()))
print(int(metrics["present_foreground"]),int(metrics["absent_foreground"]))
```

The review is now held for coordinator owner-ready signals. Do not interpret initial blockers as final post-fix findings; final acceptance is explicitly pending the rechecks below.

## Integration owner-ready recheck 1

Reported ready engine digest: `6afa081f8de12e2526fc54f340bf0c18eae6e4651710ac79850d8a03f12d0b54`. Independent scoped suite (`test_volume_config`, `engine`, `resume`, `legacy`, `integration_followup`): **20 passed in1.99s**. Original zero-weight and warmup probe now executes only one step at0.00075, pseudo_updates0 and EMA1. Temporal support24+2, logical B2, microbatch2 vs1/accumulation2 now yields identical loss0.008606422692537308, gradient norm0.026383280754089355 and gradient maxdiff0. Thus findings3–5 are numerically cleared for this CPU fixture.

Remaining defects after that ready signal:

- Auxiliary exposure counters square counts: actual temporal support26 logs676 vs580 depending on accumulation; raw counts were multiplied by their denominators. Region counters have the same structure. Effective-support reporting must remain invariant under microbatch partition.
- Legacy direct loader still verifies claimed hashes only against another manifest field, not the actual executed canonical module files. An independent temporary manufactured manifest with `expected_model={}`, `expected_canonical_modules={}`, `canonical_module_sha256={}` loads successfully. Historical canonical manifests do not contain the new canonical_module field at all. Thus this is neither a working historical-manifest path nor verified source compatibility yet; tests manufacture the missing field with fake hashes.
- CPU AMP fallback, mandatory checkpoint state/device handling, true strong-view/model-mode control and update-norm diagnostics were not yet addressed in the ready source. A subsequent live checkpoint followup began rejecting missing RNG correctly; its remaining behavior is pending a stable recheck.
- `save_every` remains accepted but unused, and `--steps` changes the scheduler horizon before construction; this is an explicit continuation into a shortened schedule, not equivalence to an uninterrupted planned-horizon run. Require periodic/interruptible checkpoint behavior or reject unsupported cadence and distinguish horizon from stop target.
- Strict config still accepts ignored `regions.detached=False`, arbitrary temporal window strings, and contradictory native-Z/resampling spacing declarations. Route all accepted controls or reject unsupported combinations.

No final full suite was rerun during these followup edits. Model/G6/export and evaluation/FP16 owners remain active, and their final files will receive separate source-ready checks.


## Owner-ready recheck 2 (CPU manufactured, 2026-10-26)

- Latest model/fusion/optimizer suite: **33 passed in3.80s**. Model source `4095051e59cc7595f9983e70f67fc4eeb05c82720ed1f6307b6e4919006d1dec`, fusion source `6437b518a08f7b3eb91833b8aaa8ce0b361d29dc99e84ee7c17fbf001fa61779`. DINO decoder tuple is now96/128/192/256, distinguished from encoder96/192/384/768; unsupported hybrid XY kernels reject. The executed Conv/Linear MAC proxy includes actual stage resolution/groups and declares omitted operations; retained stage-local fields must not be presented as total FLOPs or runtime.
- Independent trained G6 probes (seed44,3SGD updates0.05, off-depth, x1x3x3x17x19): delayed skip upstream weight gradient1.28577e-5 and inference intervention delta1.77249e-5; matched-additive gradient5.57408e-5 and delta8.80808e-5; leaving the intervention restores exact original logits. This concerns the added residual branch; native detail/window bypasses remain part of the unchanged backbone.
- Independent raw-Z probe with7raw slices, triplet input, fixed windows and33x35 spatial shape: mix/off/planar all equal unchanged NativeStudent exactly at zero initialization, including native state. Activating terminal residual weights0.025 and perturbing rawz0 changes centerz3 by4.649162292480469e-6 in mix, exactly0 in off and planar controls. This distinguishes cross-Z reach beyond the original three-channel triplet.
- Eval/loss suite: **17 passed in1.25s**. Direct FP16 and BF16 logits1x4x1x256x256, labelsall1, both legacy_pooled and volume_mean give loss1.9862946271896362 and finite input gradients(max1.5079975128173828e-5). This is a CPU arithmetic probe, not CUDA AMP execution.
- Patient weighting repaired: class1 supports1 and9, perfect and wrong predictions, respectively, now produce patient macro0.5 with absent classesNone. New defect: labels[1,1], perfectclass1 predictions, label_valid[true,false] gives valid_voxels1 but TP1/FN1/Dice2/3; unsupported target must be excluded from the batch-scoring confusion matrix. Sent through Orca for owner repair.
- Current settled integration negatives independently reproduced: temporal identity supports75+2=77 reports5929 versus5629 voxels for full versus accumulated batch; fraction0.5133333333 versus0.9747186147, although objectives agree0.011012924835085869. Raw exposure counts/fractions remain incorrect.
- Planned4-step cosine fixture, --steps2 stop, save_every1: checkpoint horizon[2], onlyfinal weights.pt/MANIFEST.json, then resume to4 vs uninterrupted4 produces max student tensor difference0.002816423773765564. Temporary payload hashes: resumed `020bcee1a8547c0e471e5af48b14be28d595a798dd88e683f11347a648ba9cf1`, uninterrupted `61d5ac95712ae70b67e8df9553791b804a6a261d8dade0b3fd1cf2b3f4172cfd`; these are manufactured counterexample artifacts, not patient weights.
- Current strict config still accepts ignored regions.detachedFalse, temporal.windownonsense and configured resampling without spacing/FOV. Legacy module map compares manifest claims only; canonical evaluation.py is also absent from the current checkpoint source closure despite being imported by native evaluation. These remain open and were re-routed after the premature owner-ready receipt.

Final acceptance remains pending the original-owner fixes and stable full suite. No reviewer source/test edits were made.


- Independent architecture-only mocked-DINO export roundtrip, original pretrained weights absent and normal pretrained loader replaced by an exception: exact logits before/after,2local nonpretrained hubcalls, decoder96/128/192/256 and encoder96/192/384/768; incorrect encoder tuple as decoder rejects. Export payload `9ce4e94af453646acf00dea403992ffa97333e275ed5e746b2d6cd44d9891f71`. A clean pinned local DINO architecture source remains required; actual DINO assets/clinical behavior were not exercised.
- Independent four-step direct trainer resume with two alternating patient identities, EMA regions(refresh5), cosine warmup2 and planned horizon4: aftercheckpoint2, resumed student/EMA/optimizer/scheduler/cache/sampler and Python/NumPy/Torch RNG all bit-exact versus uninterrupted execution. Payload `2d85903b861f19964779b3fe9de91fdbd75a27caa2d8278641c96825f62de493`. CPU AMP constructor now explicitly rejects. This clears basic cache serialization but does not clear the separate train_volume stop/horizon counterexample.


- Two final fail-closed probes routed to the active integration repair: supervised_extra main supportvalid plus extra labelsall255 raises KeyError(update_norm) aftersup1/EMA0/outer0 because the empty extra branch omits update_norm. A correctly re-sealed checkpoint with activeEMAregions and null rng.torch/region_state loads despite incomplete deterministic-resume state; presence-only validation needs type/non-null checks. These are independent manufactured negative probes, not corrupted-file bypass claims: hashes were explicitly recomputed so the test reaches semantic validation.


- **Additional accumulation/cache blocker:** RegionManager EMA refresh5 atstep0 logical[A,B] withclass1 teacher, thenstep1[A,C] withclass2 teacher while A image/support/metadata unchanged: fullB2 returns A target[0,0,1,0], per-sample microbatch1 returns cached A[0,1,0,0], maxdifference1. The cache key is the whole microbatch and trainer obtains targets inside each chunk. Therefore the repaired scalar denominators preserve only fresh/fixed-target accumulation, not the complete cached training objective. Cache each sample independently or compute/freeze full logical-batch region targets before slicing; exact repeated-A/changing-companion regression is required. Routed through Orca.


### Cached-target accumulation counterexample (independent of trainer loss code)

```python
import torch
from self_audit.volume.regions import RegionManager
cfg = dict(mode="ema",weight=.1,refresh_every=5,target_num_regions=2,min_region_size=1)
a = torch.zeros(1,3,2,4,4); b = a+.25; c = a+.5
v = torch.ones(1,2,4,4,dtype=torch.bool)
p1 = torch.zeros(1,4,2,4,4); p1[:,1] = 1
p2 = torch.zeros_like(p1); p2[:,2] = 1
ma,mb,mc = [[dict(patient=x,spacing_xyz_mm=[1.,1.,1.])] for x in "ABC"]
full,split = RegionManager(cfg),RegionManager(cfg)
full.targets(torch.cat([a,b]),v.repeat(2,1,1,1),ma+mb,
             probabilities=p1.repeat(2,1,1,1,1),step=0)
for x,m in [(a,ma),(b,mb)]:
    split.targets(x,v,m,probabilities=p1,step=0)
yf = full.targets(torch.cat([a,c]),v.repeat(2,1,1,1),ma+mc,
                  probabilities=p2.repeat(2,1,1,1,1),step=1)
ys = split.targets(a,v,ma,probabilities=p2,step=1)
print(yf.target_probabilities[0,:,0,0,0],ys.target_probabilities[0,:,0,0,0])
```

Observed before repair: full[0,0,1,0] versus split[0,1,0,0]. If repaired at the trainer boundary by full logical-batch freezing, recheck the engine objective across repeated[A,B]then[A,C], because the direct RegionManager behavior can remain batch-keyed by design.


## Configured evaluation recheck

At evaluation.py `cfaf7a4f06ff220e67d102e92bf57292f5adac4b21ea2a66dab3b3e8d015b45c`, eval/loss suite **24 passed in2.22s**. Independent coordinate-dependent logits on native(1,4,7,9) image with translated diagonal anisotropic affine(diag1.3,.9,4.2;origin12,-4,7): configured same-spacing inference matches native probabilities and support exactly(maxdiff0). Nontrivial spacing1.8,1.4,5.0/FOV8,7,15/flipTrue returns probabilities(1,4,4,7,9),93supported nativevoxels, finite values and maximum supported normalizationerror1.1920928955078125e-7. Restricted support is retained explicitly; this is not full native support or patient accuracy evidence. The partial-label scorer repro is also cleared (TP1/FN0/Dice1).

Remaining evaluation provenance limitation routed: output JSON omits effective tile/slab sizes, overlaps, spacing/FOV/flip/resampling and executed evaluation-source digest; checkpoint identity alone does not reconstruct CLI overrides. The coordinator may record/fix this before final delivery.


## Integration recheck 3

- Independently rerun config/engine/resume/legacy/integration-followup suite: **32 passed in2.38s**.
- Original temporal support77 witness now logs77 and fraction0.5133333333333333 in full and accumulated batches, with identical objective0.011012924835085869.
- C01 mainvalid/extraall255 now completes with counters(supervised,extra,EMA,outer)=(1,0,1,1), extra_update_norm0.
- Planned4/stop2 cosine fixture preserves horizon[4], emits split.step000001, and resumes to bit-identical student and EMA versus uninterrupted4. Payloads resumed`53ccf14494ad459903f78a5802748c67b8ee934f1b663c1c72f3e045b1c74953`, uninterrupted`61d5ac95712ae70b67e8df9553791b804a6a261d8dade0b3fd1cf2b3f4172cfd`; serialized container hashes differ despite exact model state, so tensor comparisons are the decisive replay evidence.
- Original canonical-v2 manufactured fixture `/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/native-volume-v2-parity-q690k5c0/handoff/final` loads exact student and EMA plus2optimizer states through the explicit historical loader. Manifest`f34eb52c219776e32383ad15c134804799e6562dd01c51e0578bd8ba053980e5`, payload`e3ad7cf5053bf03503ce60a04ed1975ca2015e00c767207a9d33778dce2ec4bb` remain unchanged; empty model/canonical maps now reject. Current canonical files are compared against a complete external historical map; router divergence is explicit, without source-digest monkeypatching.
- Current config rejects detachedFalse and arbitrary temporal windows. Remaining configured-spacing mismatch: resamplingconfigured with spacingNone still passes training config and usesnative sampling while frozen evaluation rejects it.
- New semantic checkpoint negative: correctly re-sealed payload with rng.python=() raises IndexError after changing caller model weights. RNG shape/content must be validated using isolated generators before caller mutation. Correctly constructed checkpoint with data_provenance hashesNone also saves/loads; null data identities bypass comparison when expectedNone. Non-null valid hash/type validation is required. These defects were re-routed for a narrow final repair.


## Final configuration followup

Registry/CLI suite independently **24 passed in4.59s**. V06 declares kernels(1,3,3),(1,3,3),(3,3,3),(3,3,3); V07 declares all(3,3,3); both use strides(1,1,1),(1,2,2),(1,2,2),(1,2,2), widths16/32/64/128. Synthetic8/16/24/32 width override preserves those kernel/stride fields. Both actual models independently restored output1x4x5x33x35 and produced finite nonzero decoder gradients on an odd-shape manufactured input. O02 now declares groupedAdamW with encoder2e-5/decoder3e-5/head3e-4 and cosine warmup1; optimizer and learning-rate effects are confounded by design and no superiority is established. Historical A0 remains separate.


## Strict-state and native support recheck

At checkpoint`3245000e3b894167ce86dafd6717560b40cdd4ced5829e89d6caf83cbe4fe80a`, config`23e9acf9d514d7f36665c9089bc0220a95ab393196e45cd47636f0e3aeea7673`, engine`13a7b82ee75e12ded3d7e08f292ec26f14e85034f048e72e5ee7b7da2f6fdf27`, the three semantic counterexamples now fail closed: configured mode without spacing rejects, null data pins reject before creating destination, malformed Python RNG rejects with caller model and globalTorch/Python streams unchanged. Source uses isolated Python/NumPy/Torch generators before restoration, strict non-null64hex data identities and exactboolsynthetic. CUDA RNG topology changes/cross-device exact resume are explicitly unsupported and reject; CUDA execution was not tested.

At evaluation`90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6`, eval/geometry suite **25 passed in1.65s**. Independent coordinate-dependent-logit probe compares tile99/slab3 to tile2(overlap1)/slab1 under anisotropic resampling: exactsame support, maxprobabilitydifference8.940696716308594e-8, no invalidnative voxel supported, unsupported probabilities exactly0. Effective inference protocol and source identity are now recorded by the evaluation/CLI owners. This is software evidence with partial support93/252, not patient performance.

All reviewed original findings were cleared for their declared software scope except the then-active region-cache accumulation-partition repair; the final receipt below closes that finding and records the complete gate.

## Prior stable-snapshot receipt — 410-test gate, superseded by late joint-cache finding

All findings independently reproduced by this reviewer had been resolved at this tested snapshot. A later geometry-review counterexample at the batched-cache/LRU resume boundary supersedes provisional acceptance; the final followup receipt below is authoritative. Earlier pending/failed receipts are retained as the falsification audit trail.

- Required `rtk proxy python scripts/check_repo.py`: **PASS** (canonical v2 remains semi-supervised).
- Required `rtk proxy python -m pytest -q`: **410 passed, 1 warning in33.91s**. Warning: existing Torch `torch.jit.script` deprecation in `test_decoder_jvp_transaction`.
- **150 source/config/test files unchanged before versus after the full gate.** HEAD remains `486fae40c490e382d58896442215476d7d32095e`; the shared worktree retains the other owners' uncommitted implementation. This reviewer changed only this report, made no source/test edits, and made no commit or push.
- Final region/temporal suite: **36 passed in1.90s**. Repeated[A,B]then[A,C] cache probe now returns identical class1 target for A in full/split paths, maxdifference0. Actual two-transaction region-only U objectives are identical (1.0898513203017046e-8 then8.823802488677757e-8) with model maxweightdifference0 across fullB2 versus microbatch1/accumulation2.
- Final post-fix resume: alternating A/B, EMA regions refresh5, cosine/warmup2, checkpointafter2/planned4; student/EMA/optimizer/scheduler/region cache/sampler and Python/NumPy/Torch RNG replay bit-exactly. Clocks(outer,supervised,pseudo,EMA)=(4,4,4,4). Manufactured checkpoint payload `af9b3a58c48ddb1453bce3fcb316d63fe349e18218bda922331f7178c15d85ae`.
- Zero-function, activated cross-Z versus off/planar controls, trainable G6 interventions, exact loss/gradient oracle, independent accumulation denominators, skipped updates, scheduler timing, strict resume/provenance negatives, explicit student/EMA scoring, teacher-free student export, native-grid scoring and legacy compatibility all have the source/CPU witnesses recorded above.

### Explicit evidence limits

No real patient training or accuracy experiment, CUDA execution/AMP overflow run, GPU profile, real DINO asset forward, clinical validation or superiority/novelty experiment was performed. Standalone CPU FP16/BF16 reductions and mocked-DINO architecture/export tests do not establish those outcomes. The historical loading witness uses an unmodified **manufactured canonical-v2 checkpoint**, not historical patient A0 weights. Original patient/model asset hashes and GPU timing remain NOT RUN/unavailable in this review.

Deployment DINO construction still requires its clean pinned local architecture source, while the original pretrained weight file is not reopened by the architecture-only export loader. Exact resume requires compatible RNG/device topology; unsupported topology changes fail closed. Native inference retains explicit unsupported voxels and does not compute a spacing-certified HD95 protocol. O02 changes optimizer and rates together; no optimizer-only efficacy claim is supported.

### Identity pins for the prior 410-test snapshot

The aggregate hashes below are SHA256 of UTF-8 `json.dumps(path_to_file_sha256, sort_keys=True, separators=(',', ':'))`, with repository-relative paths. The 150-file snapshot contains `train.py`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md`, the Astra prompt, and all .py/.json/.yaml/.yml files below `self_audit/`, `tests/`, `scripts/`, `configs/native_volume/`; caches are excluded by suffix. The implementation contract is pinned separately after the gate.

| Identity | SHA256 |
|---|---|
| Tested150-file snapshot | `9c6d3a3fdc9909b497c8f7aefacbcd98fac4d253621fe36d7be78293c341c476` |
| Executed volume source closure (`source_sha256()`) | `c5e83dc20c4a4a121328f48df7fec6abafc98b5a4f5958817f7a1b95240d7e3d` |
|26 native-volume configuration files | `b10806c5d32d7eb0432e041851e03b2a11fe9fee236e304beb7f4ac8d69eb7bd` |
|17 volume test files | `cf5ce31de76fa15786025ca55781c79b5e65e1f352379f94af619e6583defe12` |
|`docs/native_3d/IMPLEMENTATION_CONTRACT.md` | `43f5d26aa3b4876d6dcc3a1a670422569e4604166af6545b70c23b0627cc996a` |

| Executed/reviewed source | SHA256 |
|---|---|
| `scripts/volume_correct.py` | `1e3cd85f78346b74f363848ae80761692e9da19e5a230c6d886b84600d86688c` |
| `scripts/volume_evaluate.py` | `9a73e678ae9f8b8b0829ce7cb087d2c6f6121a4af6713f92fbc29f5fc99e5acf` |
| `scripts/volume_export.py` | `0ed301b677864d50ff949b4cc4a180b8929d1dd6145bd93aab91eb763c84a247` |
| `scripts/volume_preflight.py` | `6b0a2c79b5ac6794dd1c561b1e142d2367d2782f65737270195ccc452f8d69c3` |
| `scripts/volume_prepare.py` | `32929bb2e1fec70dbd6ff9ca2381c6312b2e0a3433a700e9854e486eeefed881` |
| `scripts/volume_profile.py` | `e137f4352e440c8df588a1a16e249f6eb640c83f8ae4b2d171c030ccb9c176e9` |
| `scripts/volume_smoke.py` | `5cb0a63295b58a8cac7658e93869974812ac6c64143a9192273762a5919b8225` |
| `self_audit/data.py` | `f6e53d46264fc170898a20ba42fe21f3dbd997e1d38e6a52de3e218e584b0260` |
| `self_audit/engine.py` | `e40757b248811b4525205320d1a756ca9da1c1fd1260f86d977c5aa0124c3f32` |
| `self_audit/evaluation.py` | `abc805c1f94e31be948720882bbeba87fdeaed430bc2cbd219d50293d66c9898` |
| `self_audit/losses.py` | `b3753b89447f3d83ef5cc40d89f288606104ef9e526cc338a1cd2f17cff93a73` |
| `self_audit/model.py` | `f0180ba618e53f613267698a043b4295c0b721bb262979178d4d520fbafa454f` |
| `self_audit/volume/__init__.py` | `43b22499363362f3a6c57ad21cf6c2cbf35d83678f9d396671b2985570a1335d` |
| `self_audit/volume/checkpoint.py` | `3245000e3b894167ce86dafd6717560b40cdd4ced5829e89d6caf83cbe4fe80a` |
| `self_audit/volume/config.py` | `23e9acf9d514d7f36665c9089bc0220a95ab393196e45cd47636f0e3aeea7673` |
| `self_audit/volume/correction.py` | `c3b8d26f5fa7b78754d55e0ec4d43e063d89faaace9bcfb3a278bfc5bd23ab97` |
| `self_audit/volume/data.py` | `a5e7448c112e9cd70cd8b9e222837cfd0aa9b0a8d23414c51ecdc8f2d1a18cc8` |
| `self_audit/volume/engine.py` | `13a7b82ee75e12ded3d7e08f292ec26f14e85034f048e72e5ee7b7da2f6fdf27` |
| `self_audit/volume/evaluation.py` | `90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6` |
| `self_audit/volume/fixtures.py` | `f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90` |
| `self_audit/volume/fusion.py` | `6437b518a08f7b3eb91833b8aaa8ce0b361d29dc99e84ee7c17fbf001fa61779` |
| `self_audit/volume/geometry.py` | `3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd` |
| `self_audit/volume/legacy_compat.py` | `47a455396033fee4048bb5f2970a443007f43e7147270d11478fbd75367e8e6e` |
| `self_audit/volume/losses.py` | `77dbd1cd86e4a74702c63b2c56955405e9c8d4a48c7f3f37630589b54bdb77cd` |
| `self_audit/volume/models.py` | `4095051e59cc7595f9983e70f67fc4eeb05c82720ed1f6307b6e4919006d1dec` |
| `self_audit/volume/optim.py` | `502b3ef1d87d7b9add16790a6d7ff93b24642309fc063148672e4e45a8c63fdb` |
| `self_audit/volume/regions.py` | `1474c539eb8ca64e92b6eaa367107a45206b1eefad55394034e14c54f200bb9c` |
| `self_audit/volume/temporal.py` | `854bf74222402446d9d984dfe1eccabec44dd152a7ef52cdd43932e48f26881d` |
| `self_audit/window.py` | `8b661ce007ef692c4575c418b90e89b59b7217f21f098c7ab19863de2e04fa72` |
| `train.py` | `49a18abff36210d351ad74b09c8c47704ffc290125fbeeeef97f115da737140f` |



## Late joint batched-cache/LRU resume followup

The geometry reviewer found an uncovered combination after the410-test gate: restoring a combined batched return into the per-sampleLRU could evict A at capacity2. Provisional acceptance was withdrawn before worker_done. Owner source `a192976e0a08247589a6b2e2ee688782cbf28b76cb53c1774ddd0fd6caa3eca6` now restores the combined last-return separately and preserves per-sample entries/order/capacity.

Independent exact witness uses `RegionConfig` mapping with max_cache_size2, EMArefresh5: step0[A,B]class1, serialize state withtorch.save/load, restore into defaultRegionManager(), thenstep1[A,C]class3. Initial/restored/split capacity=(2,2,2); finalentries2; A remainsclass1; uninterrupted-versus-resumed and resumed-versus-split target maxdiff0; allLRUkeyorders identical. Region/resume/engine suite **44 passed in2.05s**. An earlier draft probe passed only the constructor fallback argument while configdefault64 took precedence; that draft is not capacity-two evidence. The corrected config-pinned witness above is decisive.

Awaiting the coordinator final stable verification receipt after this repair and source EOF normalization; no duplicate fullpytest will be run by this reviewer.


## Coordinator stable core receipt — independently inspected

Inspected durable `verified/FINAL_VALIDATION.json`, `verified/COMMANDS.json`, `verified/pytest.stdout`, `verified/check_repo.stdout`, `verified/ROUNDTRIP.json` and `verified/evaluation.json` directly. This coordinator-executed matrix reports **418 passed,1existingTorchwarning in27.57s**, check_repoPASS and **59commands,all exit0**. It includes23preflight and23smoke invocations (V00 is HISTORICAL_REFERENCE_ONLY;22arms execute),6CPUprofile receipts, actualCLI planned4/stop2/resume4, image-onlyevaluation with0labelreads, and teacher-free export. Executed library source before/after is identical: `3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac`, matching the reviewer's current source calculation.

Receipt hashes: FINAL_VALIDATION`7a8dd5417ff3e032d9e2e668fb32a2e7d274f2a63386612b53c169a39bb8a81a`; ROUNDTRIP`b4f086710c0be8711dd22c77ea396418405f7ec84a4e8e75a30c52190a8ea9ed`; COMMANDS`d02f4d4135e26d35b631ba5e9e6a61598296a5852c7dee65eddbb05ef4100957`.

ROUNDTRIP records no payload state differences, preserved horizon4/cadence, bit-exact exported logits and no teacher/externalteacher/pretrainedweights requirement. It pins manufactured data`6932474cbab796f47ee5e98088b588158a737ddac1e0c37064d8c2bdefcc6794`, geometry`f7ef503cf11ead739f6596359f81592375d423cc44c4bb874442bac3c25cd205`, fullcheckpointmanifest`98e1991ba3fb4b05483bddb9a78c83c32f2fca16265cab34847b1e80b2382060`, resumedmanifest`c580100a56895c5e8e0c68b11a7a0393e758f7d013f2c74d8aa61000084b6b9b`, exportmanifest`ef4d96061080cf767cdc502a291f6ee057aeb0b45e1ce14dc7a268e0a299f8b0`.

Evidence distinction: the coordinator ran that matrix; this reviewer inspected and hashed its receipts. Direct reopen of the recorded `/var/folders/.../native-volume-final-htm7s0un/full/weights.pt` path failed because temporary payloads were no longer present, so the final matrix's payload/logit assertions are **receipt-derived**, not a newly rerun payload comparison by this reviewer. Separate independent numerical/cache/resume/export probes are recorded above with their own scopes/hashes. No realpatient/CUDA claims follow.

At the time of the core matrix, the coordinator had assigned C7 time-index rejection in correction CLI and requested settlement after closure/fullpostgate. The final disposition below records that completed gate. Earlier `verified_attempt1` remains explicitly INCOMPLETE_VERIFICATION_HARNESS_INPUT_ERROR (a randn probe violated the normalized input contract) and is not promoted toPASS.

## Final disposition after post-CLI gate — ACCEPTED (CPU/software only)

The coordinator released the final gate and all model/trainer findings owned by this review are closed, including the late batched/LRU resume boundary. The authoritative final gate is `POST_CLI_GATE.json`: **check_repo PASS; 422 passed, 1 existing Torch deprecation warning in 29.27s**. The coordinator executed this final full suite; this reviewer directly inspected its command/output receipt and verified every pinned file against the current filesystem. Earlier 410/411/418-test counts are retained as intermediate evidence and do not replace this final gate.

Final gate receipt SHA256: `730821e3a2aa6129c8a0dcadaa9be1bf605413f9276bfde1dc64524b5c154c7e`. Core source `3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac` is unchanged from the 59-command matrix. Since that matrix, only `scripts/volume_correct.py` and its correction test changed within the review's 151-file snapshot; the final postgate covers that strict time-index rejection. Correction/geometry ownership belongs to the separate geometry review; this report's independent final additional execution is the 44-test region/resume/engine suite plus the exact capacity 2 witness above.

Final reviewer snapshot: **151 files**, aggregate `98dd031b526ae23209ee29654a2c09f0398133dc819ddd086af14c6b2629d8a4`. This uses the same sorted-path-map JSON hash method as above, now including `docs/native_3d/IMPLEMENTATION_CONTRACT.md`. Configuration aggregate remains `b10806c5d32d7eb0432e041851e03b2a11fe9fee236e304beb7f4ac8d69eb7bd`; 17 volume-test aggregate is `7691247203efbb75568d2a34f75fe56713e60faf306a92b889cc7036b9d5b016`. These final pins supersede the earlier snapshot pins.

Temporary payload cleanup in the coordinator harness was intentional; durable training weights/data were not requested for this software pilot. The inspected receipts preserve the executed assertions and hashes; absent temporary payloads were not presented as freshly reopened evidence. All real patient, CUDA, real DINO asset, accuracy, clinical and novelty limits stated above remain **NOT RUN/unestablished**. Historical A0 and existing reports/figures were preserved by this reviewer, who changed only this owned report.

| Final source | SHA256 |
|---|---|
| `scripts/volume_correct.py` | `85439b39b33c6f0ac8d814ec8b1e11faee6223660434325a0d1f3303fb86e69f` |
| `scripts/volume_evaluate.py` | `9a73e678ae9f8b8b0829ce7cb087d2c6f6121a4af6713f92fbc29f5fc99e5acf` |
| `scripts/volume_export.py` | `0ed301b677864d50ff949b4cc4a180b8929d1dd6145bd93aab91eb763c84a247` |
| `scripts/volume_preflight.py` | `9b6098cb78ee4abfde8912caff76997d1a14a5299b375472b46862886a7503d9` |
| `scripts/volume_prepare.py` | `32929bb2e1fec70dbd6ff9ca2381c6312b2e0a3433a700e9854e486eeefed881` |
| `scripts/volume_profile.py` | `7a78f8352cfef083d09df36481673e31a234959b995b7ad704360e8117c37866` |
| `scripts/volume_smoke.py` | `f1db88a88136099fa96e00d9468bed9752154960115c131894e7d521ba1f4b81` |
| `self_audit/volume/__init__.py` | `43b22499363362f3a6c57ad21cf6c2cbf35d83678f9d396671b2985570a1335d` |
| `self_audit/volume/checkpoint.py` | `3245000e3b894167ce86dafd6717560b40cdd4ced5829e89d6caf83cbe4fe80a` |
| `self_audit/volume/config.py` | `23e9acf9d514d7f36665c9089bc0220a95ab393196e45cd47636f0e3aeea7673` |
| `self_audit/volume/correction.py` | `fdb27bad361d67786652f2d8515ba132639fb606738d99b4a9a0cc1134947f08` |
| `self_audit/volume/data.py` | `a5e7448c112e9cd70cd8b9e222837cfd0aa9b0a8d23414c51ecdc8f2d1a18cc8` |
| `self_audit/volume/engine.py` | `13a7b82ee75e12ded3d7e08f292ec26f14e85034f048e72e5ee7b7da2f6fdf27` |
| `self_audit/volume/evaluation.py` | `90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6` |
| `self_audit/volume/fixtures.py` | `f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90` |
| `self_audit/volume/fusion.py` | `6437b518a08f7b3eb91833b8aaa8ce0b361d29dc99e84ee7c17fbf001fa61779` |
| `self_audit/volume/geometry.py` | `3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd` |
| `self_audit/volume/legacy_compat.py` | `47a455396033fee4048bb5f2970a443007f43e7147270d11478fbd75367e8e6e` |
| `self_audit/volume/losses.py` | `49dd32ec8f275333f584cc8da21dcbc3b2d89519a25e06ba282f698f5f844ed0` |
| `self_audit/volume/models.py` | `4095051e59cc7595f9983e70f67fc4eeb05c82720ed1f6307b6e4919006d1dec` |
| `self_audit/volume/optim.py` | `ae52a7866d88c364240cc63db22f657518be77aa2fac4d881013b8b25475fde3` |
| `self_audit/volume/regions.py` | `a192976e0a08247589a6b2e2ee688782cbf28b76cb53c1774ddd0fd6caa3eca6` |
| `self_audit/volume/temporal.py` | `854bf74222402446d9d984dfe1eccabec44dd152a7ef52cdd43932e48f26881d` |
| `train.py` | `49a18abff36210d351ad74b09c8c47704ffc290125fbeeeef97f115da737140f` |
