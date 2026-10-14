# Executed Q/h/Z target-information audit

**VERIFIED source; DERIVED algebra; new live tensors NOT RUN HERE.** Snapshot `b937e3b746608725e43d3bc3cef8861666d152dc`; executed base `7d1c6a04064cda473d1eccfbed818392c37b1a37` plus launcher patch. [Checks](TARGET_SIGNAL_CHECKS.json) verify eleven executable hashes against recipe/student manifests and patch SHA256 `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`. **E** below denotes [executed source](../stable_guidance_world_v2_20261014/executed_source/src/self_audit_guidance/), never historical top-level `src`.

The published original-checkpoint audit remains valid historical evidence. **REPORTED:** the current SSH4070 attempt exited 255, Permission denied; configured environments were empty. Missing payloads/access block new tensors, not historical findings; latest steering prohibits further access attempts. No live reproduction/pilot occurred here. This specializes the canonical [RQ1: first failing semantic stage; RQ2: conditional safe expansion](PROBLEM_FORMULATION.md).

## Information decomposition

**VERIFIED:** E/`core.py:208–281` accumulates nonnegative votes v, total mass m, V=v/m on supported sources, and s=1−exp(−m). It creates R₀=(1−s)u+sV, u=(¼,¼,¼,¼). Names enter through wall/cavity/right-adjacency priors (`193–199,248–270`), explicitly not unique heart identifiers. Multiscale votes can repeat (`213,287`). Define sV=0 at unsupported sources.

**DERIVED, exact arithmetic:** for fixed nonnegative row-stochastic P and 0<α≤1,

```
K = α[I−(1−α)P]⁻¹ = α Σ(t≥0) (1−α)ᵗPᵗ
M = K(sV),  h = Ks,  Q = KR₀ = (1−h)u + M
Zᵢ = Mᵢ/hᵢ  only when hᵢ>0.
```

Proof: the convergent series gives K≥0 and K1=1. Consequently 0≤h≤1, Σc Mᵢc=hᵢ, and linearity gives the decomposition. Z is simplex-valued; Qᵢc−Qᵢd=hᵢ(Zᵢc−Zᵢd), preserving winners, rankings and exact ties. De-neutralization removes the specified uniform mixture; it is not temperature sharpening, a new prior, calibration or semantic validation. Positive h measures source exposure, not informative/correct support: V=u or balanced contradictory sources can yield Z=u.

E/`core.py:115–144` builds unit self-loops, symmetric spatial edges and mutual temporal matches, then row-normalizes. E/`core.py:159–167,294–300` normalizes rounding errors, solves Q and complementary support separately, casts Q to float32, and computes r=h[1−H(Q)/log4], clipped. Actual tensors require a normalization/solver/casting error budget. The simplex-only solver rejects nonunit sV; normalizing sV to pass erases its mass. One algebraically valid embedding is the five-channel simplex [sV,1−s], whose solution is [M,1−h]; its finite-precision reproduction still requires verification.

## Exact loss and gradient change

Linearity proves CE(Qᵢ,pᵢ)=(1−hᵢ)CE(u,pᵢ)+hᵢCE(Zᵢ,pᵢ). **VERIFIED:** E/`student.py:40,53–58,79–101` detaches targets/weights, interpolates logits to the bank grid, applies original permission A and computes

```
wᵢ = rᵢ A[argmax Qᵢ],  D = max(Σᵢwᵢ, 10⁻⁸)
a = 1{semantic enabled and some argmax Qᵢ ≠ BG with wᵢ>0}
LQ = (a/D) Σᵢ wᵢ CE(Qᵢ,pᵢ),  pᵢ = softmax(ℓᵢ).
```

Latest coordinator steering requires the **same certified positive-h mask S in both arms**, original full-batch denominator and activation, plus untouched full-Q diagnostics and excluded weight mass. Thus compare LQ,S and LZ,S, each summing only i∈S, while keeping original detached w,A,D,a:

```
LZ,S−LQ,S = (a/D) Σ(i∈S) wᵢ(1−hᵢ)[CE(Zᵢ,pᵢ)−CE(u,pᵢ)]
Δ∇ℓᵢ L = (awᵢ/D)(1−hᵢ)(u−Zᵢ)  for i∈S; zero otherwise.
```

Proof: substitute the mixture; the unit-target gradient is p−target. With rounded nonunit targets the exact difference is awᵢ/D[(ΣcZᵢc−ΣcQᵢc)pᵢ+Qᵢ−Zᵢ]; record row-mass errors. The masked Q arm differs from the executed full-Q objective when excluded weight is positive: disclose this, do not call it exact baseline reproduction. Native/parameter gradients follow through the interpolation adjoint/model Jacobian. These are pre-clipping differences; global clipping/AdamW are nonlinear (`104–110`). Feature loss is unchanged at identical parameters.

Preserve original activation including inactive semantic parameters' `grad=None`; exact ranking preservation does not license activation recomputation after rounding. Keep permissions, reliability, sampling, batches, context, backbone, graph and thresholds fixed. Do not recompute reliability from Z, harden distributions or rebalance classes by mass. Tiny weights cancel in a single active batch unless the denominator floor applies; they need not imply tiny updates.

## Constructive failure and conditional opportunity

**DERIVED synthetic fixture, not observed anatomy:** P=[1], α=.35, m=−log(.9), s=h=.1, V=(.01,.01,.97,.01) in BG/RV/MYO/LV order, true class RV. Then K=[1], M=.1V, Q=(.226,.226,.322,.226), Z=V. These are valid simplex distributions. Original reliability ≈0.0009426>10⁻⁸. With MYO originally permitted, the batch is active despite hard-label abstention. At p=Q the original gradient is zero; replacement gives MYO derivative .322−.97=−.648. Descent amplifies the wrong name. Perfect Z fitting passes native .7 probability/.2 margin gates with wrong MYO; Q fails. Permissions, weights and graph are unchanged. This demonstrates general-family feasibility, not realizability by the executed image readout.

Swapping RV/MYO components makes the same construction beneficial. Generally define aᵢⱼ=Kᵢⱼsⱼ/hᵢ, summing to one. If routes totaling at least 1−bᵢ have Vⱼy≥1−η for true name y, then Zᵢy≥r=(1−bᵢ)(1−η): sum good contributions and discard nonnegative bad contributions. Every competitor is ≤1−r. A same-grid student with max-coordinate error ≤d therefore has pᵢy≥r−d and margin ≥2r−1−2d. Exceeding fixed probability/margin thresholds with original permission open proves correct acceptance, while Qy=¼+h(Zy−¼) can fail. The check additionally constructs a genuinely neutral node receiving good and wrong influence with Zy>.96. Neither semantic premise is verified on real images. Bank-grid approximation cannot certify native predictions. Equal RV/MYO support remains tied; absent class evidence is not created.

## Recoverability and informative-source measures

R₀ alone does not identify support: (.4,.2,.2,.2) comes from s=.2,V=(1,0,0,0) and s=.6,V=(.5,⅙,⅙,⅙). Its centered difference gives K(R₀−u)=Q−u, not M. **With independently verified s/h**, M=K[R₀−(1−s)u]=Q−(1−h)u is algebraically recoverable, but near-cancellation erases small information and cannot recover event histories. The check loses all contrast at h=10⁻⁹ in float32. Reliability inversion divides by near-zero entropy deficit and inherits clipping/rounding; do not use it.

Recompute votes from recorded candidates or reproducible original inputs/cache, retaining event provenance. E/`io.py:116` permits only image/features/q/r0/reliability/labels; raw votes, s,h,M are absent. E/`pipeline.py:44–55` constructs the bank before saving features as float16; crucially the executed `teacher.py` adapter already returns float16 features. The serialization cast therefore does not by itself establish precision loss or a replay obstruction. Verify saved R₀/Q/reliability/labels with the exact cached inputs and code; any mismatch remains a hard reproduction failure.

At h=0, M=0 and Z is undefined. Keep full-Q diagnostics; exclude the row from both paired arms without changing original D/a, and report excluded mass. For positive h with errors |M̂c−Mc|≤εM and |ĥ−h|≤εh<h,

```
|Ẑc−Zc| ≤ (εM+εh)/(h−εh).
```

Proof: expand the quotient difference, use Mc≤h and ĥ≥h−εh. Certify tiny-h rows against a predetermined numerical budget; exclude uncertified rows symmetrically. Do not silently clip/renormalize failed reconstructions or fabricate Z. Selection of S is numerical, never reference-informed.

Report h, source diversity and positive named contrast Cᵢc=[Mᵢc−max(d≠c)Mᵢd]₊=hᵢ[Zᵢc−max(d≠c)Zᵢd]₊=[Qᵢc−max(d≠c)Qᵢd]₊ by patient/class under fixed uniform-patient and original training-weight measures. Positive channel mass alone establishes neither favorable contrast nor favorable gradient.

For event e at source j, score ρe, influence aᵢe=Kᵢⱼsⱼρe/(hᵢmⱼ). Descriptive ESS=1/Σe aᵢe². Ten duplicate equal votes give ESS ten despite one information source; group repeated candidates/scales and disclose unique events/patients. Repetition increases m and s, potentially h/confidence, without independent information. Graphs are patient-local (`core.py:116`); pixel/event ESS cannot manufacture independent patients or validate names.

## Fail-closed sequence: three diagnostics

1. **Identity/reproduction:** record source/patch, run/bank/checkpoint/input/teacher hashes, split, geometry, dtype/grid, candidates/events, s/V, graph identity and residuals. Reproduce saved R₀/Q/reliability/labels and original-checkpoint audit; compare maximum/L1 differences, simplex/mixture identities, rankings/gates and zero/tiny-h counts against declared tolerances. Unresolved mismatch means no pilot.
2. **Frozen target/gradient audit:** log M/h/Z, contrast/diversity, original permissions/w/D/activation, full and masked-Q CE/entropy/KL, excluded mass, row sums and pre-clipping differences on identical batches. Match original loss first, then compare masked Q→Z only. This supplies no semantic correctness certificate.
3. **Only after 1–2 pass:** first ablation Q→Z with the same valid-h mask, all listed controls fixed, soft source distributions retained and predetermined final checkpoint. No simultaneous sampling/context/backbone/graph/threshold changes. Empirical benefit remains untested.

Strict no-GT remains: no references in targets, names, permissions, thresholds or checkpoint choice; exposed VAL is declared development, no TEST scoring/tuning; UNKNOWN=255 is abstention. **Current live gate: NOT RUN HERE; original-artifact reproduction is a required host-side gate, no real pilot executed.** [Standalone checks](TARGET_SIGNAL_CHECKS.py) verify algebra/source identity only; no production edits, data/weight access, training or GPU execution occurred here. Local implementation/preflight belongs to other dispatched workers.
