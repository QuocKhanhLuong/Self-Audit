# Correction backbone prior-art matrix

Audit date and publication cutoff: **2026-09-11**. Repository context: Self-Audit, requested main HEAD **35aaba9335e0d8a0aa344356e20e93ae902b950d**. This is a targeted literature audit, not a code review, experiment, exhaustive systematic review, or certification of novelty. Only this report is owned by the research worker.

**Decision:** Generic recurrence, mask-conditioned feature refinement, cached image embeddings, multiple memory stores, positive/negative guidance, and confidence-filtered memory all have substantial precedents. The strongest same-image/volume threats are RP-Net, ICBNet, FocalClick, sequential interaction memory, MAIS, PRISM, and Li & Li (2026); LeCor adds session-persistent parameter adaptation. SAMURAI, SAM2Long, RPCM, DAM4SAM and newer SENTRY/StreamDAM/UAMP/Competitive Memory Readout seriously constrain claims about selective state updates and preserving reliable evidence, while operating on changing video frames. These are mechanism comparisons inferred from the primary sources below, not a finding that their systems implement the proposed accepted-transition contract.

The narrower combination to investigate is **class-specific memory of accepted local correction outcomes**, coupled to a demonstrably observable correction-state transition. No exact implementation of that complete combination was established in the inspected corpus. That is an evidence gap, not proof of universal absence. DREAM is an unresolved, unusually close error-memory threat and must remain in the comparison.

## Reading the matrix

- **Static visual features** means reusing an image representation computed once across correction interactions. Frozen weights alone do not establish this. For video, “per frame” does not mean the whole sequence shares one image.
- **Mutable hidden state** separates an independently carried latent tensor/parameters from the current mask, explicit history bank, and features recomputed from those inputs. Decoder-depth recurrence, spatial token scanning, interaction time, and video time are different axes.
- **Memory** includes explicit external banks and histories; a previous mask can itself be a state without establishing a distinct latent memory.
- **Accepted-transition semantics** asks specifically about an audited candidate transition and stored per-pixel correctness outcomes, not merely selecting the largest predicted IoU, accepting a human prompt, admitting a reliable frame, or advancing an iteration.
- **Commit/rollback state** distinguishes selecting before writing, branch pruning, output-region preservation, and restoring an entire previous state. A memory-selection equation alone does not prove transactional rollback.
- **NE** = not established in the inspected source/sections; **U** = unknown because the relevant method detail was unavailable or not inspected. Neither means proven absent from every version, supplement, or implementation.
- **Medical: NE** means no medical experiment was verified in the cited source. This does not assert that no medical adaptation exists. “Yes” reports the paper's evaluated domain, not clinical readiness.
- Sources are original papers, author manuscripts, or publisher-hosted abstracts. Full text, targeted primary-PDF extracts, and abstract-only evidence are distinguished in the source register. Author claims of efficacy were not independently reproduced.

## Same-image and same-volume correction

| Method | Static visual features | Previous mask feedback | Mutable hidden state | Memory | Error signal | Accepted-transition semantics | Commit/rollback state | Correction objective | Medical segmentation |
|---|---|---|---|---|---|---|---|---|---|
| [RP-Net: Recurrent Mask Refinement, ICCV 2021][S01] | Base support/query features reused | Soft mask explicitly conditions context relation encoder | Context features recomputed from base features and prior mask; separate carried hidden tensor NE | Support prototypes plus current mask | Segmentation supervision; FG/BG relations, not explicit correctness history | NE | Shared recurrence advances mask; audited restoration NE | Dice + cross-entropy | Yes: abdominal CT/MRI |
| [ICBNet, BIBM 2022][S02] | Encoder features are refined; caching U | Preliminary segmentation and boundary predictions fed back | Iterative feature updates established; exact state recurrence U | Distinct persistent store U | Contextual and boundary feedback | U | U | Polyp segmentation with context/boundary refinement; exact loss U | Yes: polyps |
| [Mask2Former, CVPR 2022][S03] | Reuses image feature pyramid across decoder layers | Previous layer mask restricts cross-attention | Query features evolve across decoder depth | Query state; interaction-history bank NE | Mask/class supervision; mask attention is not an error map | NE | Decoder-layer update; audited rollback NE | Universal segmentation with mask supervision | NE |
| [SAM, ICCV 2023][S04] | Image embedding computed once for repeated prompts | Dense mask prompt; iterative prompting supported | Prompt/image tokens update inside decoder; inter-call latent persistence NE | Prompt history and supplied mask; explicit session bank NE | Corrective prompts and predicted mask quality | Valid-mask/candidate selection, not accepted FIX/REGRESS | Mask ranking; complete-state rollback NE | Promptable segmentation; ambiguity-aware candidate learning | NE in inspected base-paper sections |
| [RITM, 2021][S05] | Previous mask enters feedforward network; reusable mask-independent backbone cache NE | Explicit previous-mask input, including external masks | Separate carried latent state NE | Current mask and clicks | User/simulated corrective clicks | NE | Iterative replacement; rollback NE | Iterative mask-guided interactive training | NE |
| [FocalClick, CVPR 2022][S06] | Image/previous-mask crops processed during correction | Explicit, also supports pre-existing masks | Separate persistent latent state NE | Current mask and click guidance | New click, disagreement components, predicted boundary | Local edit selection; accepted correctness-transition labels NE | Progressive Merge preserves old output outside selected connected disagreement region | Local refinement, boundary fusion, segmentation/boundary losses | NE |
| [PseudoClick, ECCV 2022][S07] | Image/click encoder; cache across revisions U | Iterative prediction feeds correction loop | Separate persistent latent state NE | Human and pseudo-click encodings | Explicit predicted current-mask FP and FN maps | Current error type, not accepted change outcome | Generates positive/negative corrective clicks; rollback NE | Segmentation plus error-map learning | Yes: cross-domain BraTS 2D and ssTEM evaluation |
| [MAIS, 2025][S08] | 3D image encoded once before interactions | Previous mask embedded as dense prompt | Memory-conditioned features change; separate carried recurrent tensor NE | FIFO of sparse click and dense mask embeddings | User/simulated residual-error clicks | Memory added after interactions; accepted correctness gate NE | Queue update/eviction, no full-state restoration established | Interactive volumetric segmentation | Yes: CT/MRI |
| [Sequential-memory interactive framework, 2024][S09] | Cache U | Previous output participates in next system state | Features learned from ordered state sequence; latent implementation U | Explicit consecutive system-state/user-feedback sequence | User/virtual-user corrections | Human interaction loop; automatic correctness acceptance U | Detailed state restoration U | Learn multiclass refinement from sequential interactions | Yes: pelvis MRI, liver/pancreas CT |
| [QAMN / Volumetric Memory Network, 2021/2023][S10] | Slice encodings; caching U | Corrected slice masks guide propagation | Exact latent mechanism U | Volumetric propagation memory | Estimated quality selects next slice for human correction | Quality-based interaction scheduling; memory admission gate U | Bidirectional propagation; rollback U | Reduce interactive 3D annotation burden | Yes |
| [PRISM, MICCAI 2024][S11] | Hybrid image encoder; once-per-session caching U in inspected source | Selected binary mask enters shallow refiner and next round | Separate session latent state NE | Cumulative positive/negative prompt maps | Residual-error prompts, confidence and boundary supervision | Highest-confidence mask chosen; accepted FIX/REGRESS NE | Candidate choice then corrective refinement; rollback NE | Segmentation, boundary and confidence learning | Yes: 3D tumors |
| [Li & Li: Correction-aware 3D tumor segmentation, 2026][S12] | Once-per-volume embedding | Selected mask enters refinement with image and history maps | Learned gated residual features; separate recurrent tensor NE | Latest positive/negative instructions plus cumulative revision indicator | FN/FP prompt sampling; gate is not calibrated error probability | Prompt-label revisions, not accepted correctness outcomes | Latest-write prompt semantics; full-state rollback NE | Dice/CE, boundary, quality and refinement losses | Yes: colon/kidney tumors |
| [LeCor, 2026 preprint][S13] | Encoder adapters change during case-specific updates | Memory masks provide consistency targets | Session-persistent LoRA parameters in encoder/decoder | Tracker memory plus learned case adapters | Click supervision and memory-based consistency | Session update; audited candidate acceptance NE | Memory held fixed during gradient step; adapter reset per case, not rejection rollback | Meta-learn correction that transfers to unclicked slices | Yes: lung CT; slices treated as a sequence |

## Recurrence, contour, boundary and error-refinement foundations

| Method | Static visual features | Previous mask feedback | Mutable hidden state | Memory | Error signal | Accepted-transition semantics | Commit/rollback state | Correction objective | Medical segmentation |
|---|---|---|---|---|---|---|---|---|---|
| [Auto-context, Tu & Bai, TPAMI 2010][S14] | Image appearance reused conceptually; cache U | Previous posterior maps provide context | Successive classifier outputs; independent persistent hidden tensor NE | Posterior context between cascade stages | Supervised labeling error | NE | Cascade advancement; rollback NE | Learn contextual labeling from image and earlier predictions | Yes: brain structures among applications |
| [Iterative Error Feedback, CVPR 2016][S15] | Same image; prediction rendering changes network input | Previous pose estimate, not a segmentation mask | Output estimate updated; separate hidden state NE | Current structured output | Supervised bounded correction target | NE | Add predicted correction; rollback NE | Iteratively correct pose coordinates | No: pose paper, conceptual precedent only |
| [Feedback Networks, CVPR 2017][S16] | Same input; features evolve recurrently | Segmentation-mask feedback NE | Recurrent representation over inference iterations | Recurrent hidden representation | Supervised task outputs | NE | Recurrent advancement; rollback NE | Iterative/coarse-to-fine recognition | NE |
| [ConvGRU visual memory, ICCV 2017][S17] | New appearance/motion features per video frame | Explicit prior-mask input NE | ConvGRU hidden representation | Learned recurrent visual memory | Video segmentation supervision | NE | Recurrent state update; rollback NE | Integrate motion and appearance for video masks | NE; changing frames |
| [R2U-Net, 2018/2019][S18] | Same image within forward pass | Prior final mask as correction input NE | Recurrent convolution activations inside residual blocks | Within-pass feature accumulation | Segmentation supervision | NE | Unrolled block recurrence, not an audited interaction transaction | Medical segmentation representation learning | Yes |
| [Deep Snake, CVPR 2020][S19] | Image features sampled along evolving contour; cache detail U | Previous contour vertices rather than raster mask | Contour coordinates evolve | Current polygon/contour | Supervised contour displacement | NE | Iterative deformation; restoration NE | Refine instance boundaries | NE |
| [PraNet, MICCAI 2020][S20] | Multiscale encoder features feed refinement stages | Coarse prediction guides reverse attention | Derived feature maps updated across stages; separate session latent NE | Intermediate predictions | Foreground-complement attention and boundary supervision | NE | Staged refinement; rollback NE | Coarse-to-fine polyp segmentation | Yes |
| [PraNet-V2, 2025][S21] | Backbone features refined with dual attention | Prior stage segmentation guidance | Within-network feature refinement | Intermediate foreground/background predictions | Dual-supervised reverse attention | FG/BG roles, not FIX/REGRESS events | Full-state commit/rollback NE | Medical segmentation with complementary foreground/background learning | Yes |
| [ErrorNet, 2019/2020][S22] | Initial image segmentation plus shape/error refinement; cache U | Initial segmentation supplied to correction mechanism | Distinct persistent inference hidden state NE | Learned shape prior; accepted-history store NE | Synthetic realistic segmentation errors and learned error representation | NE | Correct initial output; rollback NE | Learn to repair segmentation errors using generated corruptions | Yes: retinal vessels |
| [U-Mamba][S23] / [VM-UNet][S24], 2024 | Within-pass image encoding | Previous final prediction input NE | State-space scan state across tokens/locations | Implicit sequence state, not established correction-history bank | Segmentation supervision | NE | Spatial/sequence-state update; correction rollback NE | Long-range representation for medical segmentation | Yes |
| [Positive–negative memory banks for spot signs, 2025][S40] | Training representation details U | Previous correction-mask input U | Case-persistent hidden state U | Dynamic positive/negative contrastive training banks | Positive/negative sample distinction | Accepted correction semantics U | Bank update details U; rollback U | Contrastive learning for tiny-object segmentation | Yes: multiphase CT angiography |

## Video memory, quality selection and newer close work

These methods must be compared for architectural mechanisms without treating video frame progression as repeated correction of one fixed image.

| Method | Static visual features | Previous mask feedback | Mutable hidden state | Memory | Error signal | Accepted-transition semantics | Commit/rollback state | Correction objective | Medical segmentation |
|---|---|---|---|---|---|---|---|---|---|
| [STM, ICCV 2019][S25] | Different frame encodings | Previous masks encoded with memory frames | Memory-conditioned current features; distinct recurrent hidden state NE | External space-time key/value memory | Segmentation supervision; quality gate NE | NE | Add encoded frame/mask memories; full-state rollback NE | Propagate object masks through video | NE |
| [STCN, NeurIPS 2021][S26] | Image-derived affinities per frame | Object masks remain in memory values | Memory readout updates derived representation | Key/value memory with improved coverage | Segmentation supervision; affinity similarity | NE | Memory management, not correctness transaction | Efficient video segmentation and memory coverage | NE |
| [XMem, ECCV 2022][S27] | New image features per frame | Predicted masks support memory encoding | Sensory GRU state | Sensory, working and consolidated long-term prototype stores | Segmentation supervision; usage-based consolidation | NE | Update/consolidate stores; rejected-candidate rollback NE | Long-video segmentation within bounded memory | NE |
| [XMem++, ICCV 2023][S28] | New image features per frame | User-annotated and predicted masks | Inherits XMem state | Permanent annotated-frame memory plus transient stores | Human annotations; frame-selection assistance | Permanent reference selection, not per-pixel accepted outcomes | Preserve annotation anchors; all-state restoration NE | Video annotation from multiple reference frames | NE |
| [RPCM, AAAI 2022][S29] | Frame features subsequently modulated | Preceding/reference labels condition propagation/correction | Buffered embeddings; separate propagation/correction modulators | Reliable object patch pool and embedding buffers | Entropy-threshold reliability map | Reliability-filtered patch admission; FIX/REGRESS NE | Correction follows propagation to avoid overwriting its effect; rollback NE | Segmentation and suppression of propagated errors | NE |
| [SAM 2, 2024][S30] | Unconditioned embedding once per frame; reused during interaction | Masks form dense prompts and encoded memories | Derived tokens change; default distinct GRU is not used | Recent/prompted frame FIFO plus object pointers | Prompts, IoU prediction, object-presence prediction | Highest-IoU candidate propagated; audited local transitions NE | Encode selected predictions; whole-state rollback NE | Promptable image/video segmentation | NE in inspected base-paper sections |
| [SAMURAI, 2024][S31] | SAM2 base frame embeddings | Predicted masks enter history | Kalman motion state; conditioned features | History filtered by mask/object/motion scores | Predicted quality, presence, KF consistency | Explicit eligibility thresholds; local correctness outcomes NE | Filters usable memory, chooses mask and updates motion; all-state rollback NE | Training-free robust video tracking | NE |
| [SAM2Long, 2024 / ICCV 2025][S32] | Encoder once per frame across branches | Candidate masks encoded on their own paths | Path-specific state and scores | Constrained tree of memory banks | Predicted IoU and occlusion scores | Top paths survive; frames pass quality/presence filter | Branch retention/pruning, not proof of transactional restoration | Robust long-video segmentation without extra training | NE |
| [DAM4SAM, CVPR 2025][S33] | SAM2 frame embeddings | Masks identify reliable anchor frames | Memory-conditioned features | Recent Appearance Memory + Distractor Resolving Memory | Presence, distractor detection, tracking reliability | Conditional anchor/frame admission; FIX/REGRESS NE | Separate bank rules, fixed initial anchor; all-state rollback NE | Resist distractors and improve re-detection | NE |
| [MoSAM, 2025][S34] | Frame features; motion-conditioned decoding | Historical masks/features | Memory-conditioned current representation | Selected reliable frames and filtered pixel features | Temporal IoU quality, spatial confidence, motion prompts | Frame/pixel filtering; accepted semantic transition NE | Filters memory contributions; all-state rollback NE | Motion-aware video mask propagation | NE |
| [SENTRY, 2026 preprint][S35] | SAM2 frame embedding plus auxiliary candidate cues | Candidate/previous masks and reverse tracklets | Kalman/temporal state; derived features | Target history plus unselected neighbor tracklets | Reverse temporal validation and neighbor association | Candidate selection before memory write; FIX/REGRESS NE | Selected candidate or KF fallback is written under baseline schedule; full rollback NE | Avoid target drift and memory contamination | NE |
| [StreamDAM, 2026 preprint][S36] | Per-frame visual encoding | Masks subject to presence-conditioned handling | GRU presence-state estimate | Presence-gated memory with adaptive stride | Predicted target presence, supervised with BCE | Admission by presence threshold; correctness outcome NE | Hysteretic write scheduling and re-detection; full rollback NE | Streaming video segmentation under absence/reappearance | NE |
| [UAMP, PLOS One 2026][S37] | Current and prior features fused | Historical predictions/memory condition current frame | Recurrent long-term state plus motion state | Long-term linear-attention memory; selected short-term bank | Appearance/motion uncertainty and hybrid mask/object/motion scores | Score-based selection, not accepted local outcome | Adaptive retain/update; all-state rollback NE | Consistent video segmentation; CE/Dice and uncertainty loss | NE |
| [Competitive Memory Readout, 2026 report][S38] | SAM3 frame pipeline retained | Target masks define foreground tokens | Readout-derived features; added persistent hidden tensor NE | Target and explicit same-class competitor tokens | Relative target/competitor evidence; suppression strength | Competition gate; accepted correction semantics NE | Suppress then restore attention evidence, not state rollback | Reduce identity drift while recovering weak target evidence | NE |
| [DREAM, indexed anonymous 2026 submission][S39] | U | U | Latent-memory mechanism claimed; exact recurrence U | Latent semantic memory + recurring-error memory claimed | Error-pattern memory; labels and online availability U | U | U | Continual LiDAR semantic segmentation, per indexed abstract | No: autonomous-driving LiDAR; detailed methods U |

## Strongest threats and verified mechanism anchors

The following are the practical comparison priorities. Section/equation numbers refer to the linked versions; they are not reconstructed from method names.

1. **Same-image features already change in response to masks.** RP-Net §3.2.2 and §3.2.4, Eq. (5), recomputes its context relation encoding from fixed query features and the preceding soft mask before prototype comparison. Its final loss is Eq. (6). Mask2Former §3.2.1 restricts attention using the previous decoder mask while query features evolve. ICBNet's publisher abstract explicitly describes iterative context/boundary feedback refining encoder features, but its full state and loss specification could not be verified. Thus, “dynamic features instead of a static backbone” is too broad a contribution. [RP-Net][S01], [Mask2Former][S03], [ICBNet][S02]

2. **Preserving existing predictions is an explicit earlier design goal.** FocalClick §3.1, Progressive Merge, finds a connected region in the disagreement of old/new binary masks containing the latest click and updates there while retaining the old mask elsewhere. Its boundary fusion is Eq. (1). This is a direct output-preservation baseline; it does not require identifying accepted correctness transitions. A repair/preserve claim must explain why semantic obligations differ from this spatial edit restriction. [FocalClick][S06]

3. **Same-volume correction memory is a primary comparison, not merely a video analogy.** MAIS §2.1.1–2.1.4 encodes each volume once, stores sparse/dense prompt embeddings in a FIFO after interactions, and conditions decoding on memory. The sequential-memory framework's publisher abstract already models ordered system states in multiclass medical refinement. Quality-aware volumetric memory selects the next slice to correct; do not misreport that scheduling function as a demonstrated memory-write quality gate. [MAIS][S08], [Sequential memory][S09], [VMN][S10]

4. **Positive/negative/revision channels and error-gated refinement are already published.** PRISM §2 combines mask confidence, cumulative prompt maps and a shallow corrective refiner. Li & Li §3.2, Eq. (3), uses a learned residual gate explicitly distinguished from calibrated voxel-error probability; §3.5, Eqs. (7)–(10), maintains latest positive/negative instructions and a cumulative conflict/revision indicator. Its revision map records changed instructions, not whether an accepted model edit became correct. Adding a history channel or a second bank cannot establish novelty. [PRISM][S11], [Li & Li][S12]

5. **Session-persistent adaptation is now a near-cutoff threat.** LeCor §3.3, Eq. (2), updates case adapters using click and consistency losses; Eq. (3) meta-trains improvement on unclicked slices. Algorithm 1 holds tracker memory fixed during each adaptation step and resets adapters per case. This makes parameters part of session state and can change encoder representations. Its slice-sequence protocol and gradient adaptation differ from a fixed-weight accepted-transition refiner. [LeCor][S13]

6. **Quality-controlled correction memory predates SAM2 extensions.** RPCM's “Object proxy for Correction Path,” Algorithm 1, and “Reliability filter” use a maintained object-patch pool selected using entropy; the filter follows Eq. (5). Buffered propagation and correction embeddings serve different roles, with correction applied after propagation. This threatens generic reliable-correction memory and ordering-to-preserve-corrections claims. Its reliability estimate is not a ground-truth correctness event. [RPCM][S29]

7. **SAMURAI and SAM2Long are substantive selection precedents.** SAMURAI §4.2, Eq. (9), requires mask-quality, object-presence and Kalman-consistency thresholds for memory eligibility; §4.1 also chooses masks using motion agreement. SAM2Long §3.2 maintains each path's bank and cumulative log predicted-IoU score, expands candidates and retains the top paths; §3.3 admits frames satisfying predicted IoU above threshold and positive occlusion score. These mechanisms establish selective memory and isolated hypothesis histories, but not same-image FIX/REGRESS labels or restoration of every recurrent variable. [SAMURAI][S31], [SAM2Long][S32]

8. **Two roles and negative evidence are close but semantically different.** DAM4SAM §3.2.1–3.2.2 separates recent appearance from distractor-resolving anchors with different update rules. Competitive Memory Readout §3.2, Eqs. (1)–(3), compares target and same-class competitor evidence; §3.3, Eq. (4), restores part of overly suppressed target evidence. These are explicit distractor/target roles, not local edits classified against before/after correctness. [DAM4SAM][S33], [Competitive Memory Readout][S38]

9. **Inspect actual write timing before calling any method rollback.** SENTRY §3.2 and Algorithm 1 reverse-check candidates against target and neighbor tracklets before selecting a write; unselected tracklets can remain as negative neighbors. If all candidates fail its reliability threshold, it uses a Kalman-translated prior-mask fallback and still follows the baseline memory schedule. StreamDAM §III-D–E, Eqs. (3)–(6), instead uses a GRU presence estimate to govern admission and hysteretic stride. Neither inspected specification establishes an all-state rollback transaction. [SENTRY][S35], [StreamDAM][S36]

10. **Error memory is not safely claimable as new.** DREAM's indexed primary PDF explicitly identifies latent and error memories for continual LiDAR segmentation. Direct PDF retrieval returned HTTP 403 and the web reader returned a browser-verification page. The indexed document is labeled an anonymous CVPR 2026 submission/review copy; accepted main-conference or workshop status, identified authorship, error construction, write rules, test-time label access, and FIX/REGRESS semantics remain unverified. It is a provisional threat, not an accepted-paper attribution or proof of equivalence. [DREAM indexed primary PDF][S39]

## What a narrower semantic claim would have to mean

The following is an **audit comparison contract**, not a claim extracted from a prior paper or a verification of current repository behavior.

Let m be the currently committed multiclass mask, c a candidate, y the reference used to define training/evaluation labels, and a the candidate acceptance decision. Define at pixel v:

- FIX(v) = [m(v) differs from y(v)] and [c(v) equals y(v)].
- REGRESS(v) = [m(v) equals y(v)] and [c(v) differs from y(v)].
- Wrong-to-wrong class changes are neither event. Unchanged pixels are not automatically correct.
- On an accepted FIX, a preservation obligation needs the **destination class c(v)**, not just a positive bit.
- On an accepted REGRESS, a repair obligation needs the **pre-transition class m(v)**, not just a negative bit.
- A globally accepted candidate can contain local regressions. An aggregate quality gate must not silently relabel every changed pixel as repaired.
- Accumulation/overwrite, conflicts, expiration, and whether an obligation survives subsequent edits must be explicit. A count or feature bank without this provenance is not equivalent to class-specific obligations.

This differs from four established signals: current-mask FP/FN in PseudoClick; positive/negative user instructions in PRISM and Li & Li; foreground/background feature roles in RP-Net and PraNet-V2; and reliable/unreliable frame or object evidence in video memory. They remain essential controls because they could explain much of an observed benefit. [PseudoClick][S07], [PRISM][S11], [Li & Li][S12], [RP-Net][S01], [PraNet-V2][S21], [SAMURAI][S31]

Ground truth in the definitions does not make it available at inference. Any deployed learned transition/error signal needs a specified causal estimator or actual human evidence. A result using oracle local labels establishes only an oracle experiment. The present audit has not tested any estimator, medical performance, or implementation.

A convincing distinction also needs a **same-mask intervention**: hold the image, current mask, current prompts, and randomness fixed, then vary only admissible latent/history state derived from earlier accepted transitions. Measure whether the next proposal or acceptance changes and whether it respects the stored class obligations. Compare with previous-mask-only recurrence, ordinary cumulative prompts, generic FIFO memory, and equal-capacity history features. If histories can never differ at the same current mask under the protocol, state sufficiency must be analyzed before proposing that intervention.

### Rollback observability under rejection followed by HALT

Conditional on the coordinator's stated protocol—rejection always returns the previous mask and immediately HALTs—restoring only a candidate hidden state has **no observable effect on the returned segmentation**. Both “leave candidate latent state unused” and “restore previous latent state” return the same mask when no future computation reads the state. This follows from the execution contract, not from an experimental result.

A rollback contribution becomes behaviorally testable only when rejected state could otherwise influence a subsequent proposal, another surviving branch, resumed interaction, exported reusable state, or another defined observable. If any such continuation is allowed, the committed state should be specified together: mask, latent tensor, memories, quality/controller history, and any adaptive parameters. Restoring one tensor does not establish restoration of that entire state. Branch isolation in SAM2Long and selection-before-write in SENTRY are relevant comparisons, but neither should be renamed “full rollback” without evidence. [SAM2Long][S32], [SENTRY][S35]

## Compact primary-source register

F = relevant full-text method sections read; P = targeted primary-paper text/extracts read, not a complete methods audit; A = publisher/arXiv abstract or metadata only. These grades concern this audit's access, not the paper's quality. “Preprint” avoids inferring a venue from formatting, a secondary index, or an author social post.

| ID | Primary paper/version and bibliographic status | Evidence actually inspected |
|---|---|---|
| [S01] | Tang et al., Recurrent Mask Refinement for Few-Shot Medical Image Segmentation, ICCV 2021; arXiv 2108.00622 | F: CVF PDF pp. 3–5, §3.2.1–3.2.4, Eqs. (1), (2), (5), (6) |
| [S02] | ICBNet: Iterative Context-Boundary Feedback Network for Polyp Segmentation, BIBM 2022; DOI 10.1109/BIBM55620.2022.9995022 | A: indexed publisher abstract; direct page challenged; full method not obtained |
| [S03] | Cheng et al., Masked-Attention Mask Transformer for Universal Image Segmentation, CVPR 2022 | F: CVF PDF §3.2.1–3.3 |
| [S04] | Kirillov et al., Segment Anything, ICCV 2023; arXiv 2304.02643 | P: CVF main-paper task/model sections; supplement not separately audited |
| [S05] | Sofiiuk et al., Reviving Iterative Training with Mask Guidance for Interactive Segmentation, arXiv 2102.06583 (2021) | A: original manuscript abstract; exact cache implementation not checked |
| [S06] | Chen et al., FocalClick: Towards Practical Interactive Image Segmentation, CVPR 2022 | F: CVF PDF pp. 3–5, §3.1 and Eqs. (1)–(2), Progressive Merge |
| [S07] | Liu et al., PseudoClick: Interactive Image Segmentation with Click Imitation, ECCV 2022 | F: ECVA PDF §3.1–3.4; medical cross-domain evaluation §4.3 |
| [S08] | Orbes-Arteaga et al., MAIS: Memory-Attention for Interactive Segmentation, arXiv 2505.07511v1, 12 May 2025 | F: §2.1.1–2.1.4 and §2.3 |
| [S09] | A deep learning-based interactive medical image segmentation framework with sequential memory, CMPB 245:108038 (2024); DOI 10.1016/j.cmpb.2024.108038 | A: publisher abstract and highlights; full state equations U |
| [S10] | Zhou et al., Quality-Aware Memory Network for Interactive Volumetric Image Segmentation, arXiv 2106.10686; Volumetric Memory Network for Interactive Medical Image Segmentation, Medical Image Analysis 83:102599 (2023) | A: primary manuscript/publisher abstracts; DOI 10.1016/j.media.2022.102599 |
| [S11] | Li et al., PRISM: A Promptable and Robust Interactive Segmentation Model with Visual Prompts, MICCAI 2024 | F: original manuscript full text, §2 iterative/confidence/corrective learning |
| [S12] | Hao Li and Haoxuan Li, Correction-aware interactive 3D tumor segmentation with sparse and revisable prompts, The Visual Computer (29 June 2026), DOI 10.1007/s00371-026-04560-5 | F: publisher §3.2–3.6, Eqs. (2)–(10); publication date checked |
| [S13] | Luo et al., LeCor: Learning to Be Corrected by Meta-Learned Test-Time Training for Interactive 3D Lung-Tumour Segmentation, arXiv 2609.09477v1, 8 September 2026, preprint | F: §3.2–3.3, Eqs. (2)–(3), Algorithm 1; date within cutoff |
| [S14] | Tu and Bai, Auto-Context and Its Application to High-Level Vision Tasks and 3D Brain Image Segmentation, TPAMI 2010; DOI 10.1109/TPAMI.2009.186 | A/P: original-paper abstract and primary author manuscript; exact equations not audited |
| [S15] | Carreira et al., Human Pose Estimation with Iterative Error Feedback, CVPR 2016 | P: primary PDF formulation of rendered output and bounded iterative corrections |
| [S16] | Zamir et al., Feedback Networks, CVPR 2017 | A/P: original manuscript abstract and recurrent-representation description |
| [S17] | Tokmakov et al., Learning Video Object Segmentation with Visual Memory, ICCV 2017 | P: primary PDF §4.1–4.2 recurrent visual memory/BPTT |
| [S18] | Alom et al., Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation, arXiv 1802.06955; journal full text (2019) | P: original manuscript plus publisher/PMC §3.1 |
| [S19] | Peng et al., Deep Snake for Real-Time Instance Segmentation, CVPR 2020 | A: CVF paper abstract; exact feature caching U |
| [S20] | Fan et al., PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI 2020 / arXiv 2006.11392 | A: primary manuscript abstract; exact stage loss detail not asserted |
| [S21] | Hu et al., PraNet-V2: Dual-Supervised Reverse Attention for Medical Image Segmentation, arXiv 2504.10986v2 (19 January 2026; first posted April 2025) | A: primary manuscript abstract; inference history persistence U |
| [S22] | Tajbakhsh et al., ErrorNet: Learning Error Representations from Limited Data to Improve Generalization of Medical Image Segmentation, arXiv 1910.04814 (2019), ISBI 2020 version | A: primary abstract; correction-state implementation not audited |
| [S23] | Ma et al., U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation, arXiv 2401.04722v1 (2024) | P: original HTML architecture/state-space description |
| [S24] | Ruan et al., VM-UNet: Vision Mamba UNet for Medical Image Segmentation, arXiv 2402.02491v1 (2024) | P: original HTML VSS/encoder-decoder description |
| [S25] | Oh et al., Video Object Segmentation using Space-Time Memory Networks, ICCV 2019 | P: primary paper external memory/read-write description |
| [S26] | Cheng et al., Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation, NeurIPS 2021 | P: proceedings PDF affinity and memory-value description |
| [S27] | Cheng and Schwing, XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model, ECCV 2022 | P: primary PDF sensory/working/long-term stores and GRU description |
| [S28] | Bekuzarov et al., XMem++: Production-level Video Segmentation From Few Annotated Frames, ICCV 2023 | P: primary PDF §3.2 permanent annotation memory |
| [S29] | Xu et al., Reliable Propagation-Correction Modulation for Video Object Segmentation, AAAI 2022 | F/P: primary PDF modulator, reliable object pool, Algorithm 1 and reliability filter following Eq. (5) |
| [S30] | Ravi et al., SAM 2: Segment Anything in Images and Videos, arXiv 2408.00714v1 (2024) | F: §4, §8.2.3, Appendix C.1; default direct memory storage distinguished from tested GRU |
| [S31] | Yang et al., SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory, arXiv 2411.11922v1 (2024) | F: §4.1–4.2, Eqs. (4)–(9) |
| [S32] | Ding et al., SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree, arXiv 2410.16268v1 (2024); ICCV 2025 | F: §3.1–3.3; CVF proceedings record independently located |
| [S33] | Videnovic et al., A Distractor-Aware Memory for Visual Object Tracking with SAM2, CVPR 2025 | F/P: primary PDF §3.2.1–3.2.2 and §5.1; no claims based on later extension |
| [S34] | Yang et al., MoSAM: Motion-Guided Segment Anything Model with Spatial-Temporal Memory Selection, arXiv 2505.00739v1 (30 April 2025) | F/P: §3.2 and §3.3.1–3.3.2 temporal/spatial reliability |
| [S35] | Alansari et al., SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking, arXiv 2606.24449v1 (23 June 2026), preprint | F: §3.2 and Algorithm 1, including fallback write behavior |
| [S36] | Xiang Chen, StreamDAM: Presence-Aware Memory for Real-Time Streaming Video Object Segmentation, arXiv 2608.03912v1 (4 August 2026), preprint | F: §III-D–E, Eqs. (3)–(6) |
| [S37] | UAMP: Consistent video object segmentation with uncertainty-aware memory propagation, PLOS One (7 July 2026), DOI 10.1371/journal.pone.0353156 | F/P: publisher §3, Algorithm 1; Eqs. (4)–(6), (11), (15)–(18) referenced; some equation glyphs absent in HTML extraction |
| [S38] | Gao et al., Competitive Memory Readout for Robust Video Object Segmentation, arXiv 2608.22064v1 (22 August 2026), challenge technical report | F: §3.1–3.3, Eqs. (1)–(4); rank/performance not independently verified |
| [S39] | DREAM: Dual-Memory Reasoning for Anticipatory 3D Perception in Autonomous Driving, anonymous indexed CVPR 2026 submission PDF | A: indexed primary abstract/header only; direct full-text access 403; accepted venue/authorship U |
| [S40] | Zhang et al., Tiny-objective segmentation for spot signs on multi-phase CT angiography via contrastive learning with dynamic-updated positive–negative memory banks, Computers in Biology and Medicine (2025), DOI 10.1016/j.compbiomed.2025.110613 | A: paper abstract reproduced in PubMed, PMID 40651197; publisher direct access 403; bank internals U |

## Search record and limits

Live web searches were executed on 2026-09-11, followed by direct primary-page/PDF reads and citation-following. The numbered queries below are actual submitted query strings captured during the run; repeats are retained. They are not a claim that every returned hit was screened or that the search engine exposed all relevant papers. Direct URL opens and within-document section searches are summarized by the source register rather than counted as new keyword searches.

The search covered named baseline families, exact repair/preserve and error-memory phrases, accepted-transition/rollback wording, positive/negative memory, and recent SAM2/SAM3 memory work. Several exact queries produced irrelevant software-memory or non-segmentation results. No retrieved hit with those words alone was treated as evidence of the desired semantic mechanism.

Third-party indexes, aggregators, and social posts were discovery aids only; matrix claims are linked to primary paper sources. In particular, a social claim of DREAM workshop acceptance was not substituted for an accessible proceedings or OpenReview decision record. CVF PDF web-reader errors were partly recovered through ordinary direct PDF downloads and text extraction. DREAM's 403/challenge remained unresolved; ICBNet remained abstract-only. Some publisher abstracts expose only high-level mechanisms, so U/NE entries must not become categorical negatives.

Coverage is targeted, with a strong focus on the requested families and recent memory/correction work. It does not exhaust test-time adaptation, continual segmentation, interactive annotation tools, every SAM3 extension, non-English literature, patents, or every paper released on the cutoff date. Recent preprint mechanisms are relevant prior disclosure even when acceptance and independent reproduction are unknown.

### Captured keyword queries

1. "Recurrent Mask Refinement" ICCV 2021
2. ICBNet iterative correction segmentation
3. SAMURAI SAM2Long memory selection quality motion confidence
4. segmentation "repair" "preserve" "memory" correction
5. ICBNet iterative context boundary network medical segmentation
6. SAMURAI arxiv motion aware memory
7. SAM2Long arxiv memory tree
8. "segmentation" "dual" "error memory"
9. "DREAM" "Dual-Memory Reasoning" segmentation paper
10. "interactive segmentation" "memory" positive negative
11. "segmentation" "correction memory"
12. "segmentation" "repair" "preserve" network
13. RITM Reviving Iterative Training with Mask Guidance interactive segmentation arxiv
14. FocalClick Towards Practical Interactive Image Segmentation CVPR 2022
15. PseudoClick Interactive Image Segmentation Click Imitation arxiv
16. Recurrent Mask Refinement Few Shot Medical ICCV 2021 pdf
17. "ICBNet" pdf
18. "Mask2Former" masked attention mask prediction previous layer paper
19. "ConvGRU" "Learning Video Object Segmentation"
20. "R2U-Net" recurrent residual convolutional neural network medical image segmentation
21. "ICBNet" "pdf"
22. "Auto-Context" "Tu" "Bai" 2010
23. Human Pose Estimation Iterative Error Feedback CVPR 2016
24. Feedback Networks CVPR 2017 Zamir paper
25. "STM" "Video Object Segmentation using Space-Time Memory Networks" paper
26. "STCN" "Rethinking Space-Time Networks" paper
27. "XMem" long term video segmentation memory model paper
28. "Deep Snake" contour instance segmentation CVPR 2020
29. "STCN" "Rethinking Space-Time" arxiv
30. "Video Object Segmentation using Space-Time Memory Networks" arxiv
31. "Mamba" "U-Mamba" "VM-UNet" medical segmentation
32. "segmentation" "error" "preserve" "memory" 2026
33. site:arxiv.org STCN Cheng Tai Tang 2021
34. site:arxiv.org U-Mamba Enhancing Long range dependency biomedical segmentation
35. site:arxiv.org VM-UNet Vision Mamba UNet medical image segmentation
36. site:arxiv.org "DREAM" "error memory"
37. Rethinking Space-Time Networks with Improved Memory Coverage Efficient Video Object Segmentation
38. PraNet reverse attention network polyp segmentation arxiv
39. ErrorNet learning error representations robust medical segmentation
40. "segmentation" "repair-preserve"
41. "ErrorNet" segmentation paper
42. "segmentation" "positive negative" "memory" correction
43. "segmentation" "rollback" "memory"
44. "segmentation" "accepted" "transition" "memory" correction
45. SAM2 memory quality gated refinement 2026 DAM4SAM SAM2Long
46. SAM2 memory error correction rollback reliability 2026
47. "segmentation" "repair memory"
48. "segmentation" "preservation memory"
49. "SAM 2" "Segment Anything in Images and Videos" arxiv
50. "Segment Anything" Kirillov ICCV 2023 paper
51. "SAM2Refiner" arxiv
52. "DREAM" "Dual-Memory" "Perception" -site:linkedin.com -site:reddit.com
53. "segmentation" "repair" "preservation" "memory" neural
54. "DREAM: Dual-Memory Reasoning" "3."
55. "DREAM: Dual-Memory Reasoning" "error" "training"
56. "segmentation" "commit" "rollback" mask -fault -software -database
57. "PRISM" "interactive" "refinement" "segmentation" paper
58. "segmentation" "preserve" "correctly" "loss" correction
59. "segmentation" "error memory" "correct"
60. "segmentation" "regression" "previously correct" refinement
61. "DREAM" "Dual-Memory" "error" "memory" segmentation
62. "A Distractor-Aware Memory" "3.2"
63. "Segment Anything" "computed once" "previous" mask
64. "XMem" "sensory memory" "GRU" paper
65. "MAIS" "Memory-Attention" interactive segmentation
66. "Correction-aware interactive 3D tumor segmentation"
67. "LeCor" "2026" segmentation
68. "PRISM" "confidence" "corrective" segmentation
69. "SENTRY" "SAM" segmentation "2026"
70. "StreamDAM" memory "2026"
71. "UAMP" memory propagation segmentation
72. "Competitive Memory Readout" segmentation
73. "Tiny-objective segmentation" "positive–negative memory"
74. "A deep learning-based interactive medical image segmentation framework with sequential memory"
75. "Reliable Propagation-Correction Modulation" segmentation
76. "ErrorNet" "shape" "error" segmentation
77. "Reliable Propagation-Correction Modulation" "reliability filter" "threshold"
78. "Tiny-objective segmentation" "positive" "memory banks"
79. "SENTRY" "2606.24449"
80. "StreamDAM" "2608.03912"

81. site:sciencedirect.com/science/article/abs/pii/S0010482525009643 "positive"
82. site:sciencedirect.com/science/article/pii/S1361841522002316 "Volumetric"
83. "Volumetric Memory Network" "2023" segmentation
84. "Tiny-objective segmentation for spot signs" "negative" "positive"

### Coordinator-forwarded follow-up

The coordinator supplied [Subtract or Replay, arXiv:2607.27539](https://arxiv.org/abs/2607.27539) as a cross-domain rollback/replay lead from an independent review. This audit did not inspect its primary text; its detailed mechanism, publication status and relation to segmentation-state rollback remain U here. It is not counted among the verified matrix sources.

## Primary URLs

[S01]: https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_Recurrent_Mask_Refinement_for_Few-Shot_Medical_Image_Segmentation_ICCV_2021_paper.pdf
[S02]: https://ieeexplore.ieee.org/document/9995022/
[S03]: https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf
[S04]: https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf
[S05]: https://arxiv.org/abs/2102.06583
[S06]: https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_FocalClick_Towards_Practical_Interactive_Image_Segmentation_CVPR_2022_paper.pdf
[S07]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660717.pdf
[S08]: https://arxiv.org/html/2505.07511v1
[S09]: https://www.sciencedirect.com/science/article/abs/pii/S0169260724000348
[S10]: https://www.sciencedirect.com/science/article/pii/S1361841522002316
[S11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12128912/
[S12]: https://link.springer.com/article/10.1007/s00371-026-04560-5
[S13]: https://arxiv.org/html/2609.09477v1
[S14]: https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf
[S15]: https://openaccess.thecvf.com/content_cvpr_2016/papers/Carreira_Human_Pose_Estimation_CVPR_2016_paper.pdf
[S16]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Zamir_Feedback_Networks_CVPR_2017_paper.pdf
[S17]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Tokmakov_Learning_Video_Object_ICCV_2017_paper.pdf
[S18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6435980/
[S19]: https://openaccess.thecvf.com/content_CVPR_2020/html/Peng_Deep_Snake_for_Real-Time_Instance_Segmentation_CVPR_2020_paper.html
[S20]: https://arxiv.org/abs/2006.11392
[S21]: https://arxiv.org/abs/2504.10986v2
[S22]: https://arxiv.org/abs/1910.04814
[S23]: https://arxiv.org/html/2401.04722v1
[S24]: https://arxiv.org/html/2402.02491v1
[S25]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.pdf
[S26]: https://proceedings.neurips.cc/paper_files/paper/2021/file/61b4a64be663682e8cb037d9719ad8cd-Paper.pdf
[S27]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880633.pdf
[S28]: https://openaccess.thecvf.com/content/ICCV2023/papers/Bekuzarov_XMem_Production-level_Video_Segmentation_From_Few_Annotated_Frames_ICCV_2023_paper.pdf
[S29]: https://ojs.aaai.org/index.php/AAAI/article/download/20200/19959
[S30]: https://arxiv.org/html/2408.00714v1
[S31]: https://arxiv.org/html/2411.11922v1
[S32]: https://arxiv.org/html/2410.16268v1
[S33]: https://openaccess.thecvf.com/content/CVPR2025/papers/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.pdf
[S34]: https://arxiv.org/html/2505.00739v1
[S35]: https://arxiv.org/html/2606.24449v1
[S36]: https://arxiv.org/html/2608.03912v1
[S37]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353156
[S38]: https://arxiv.org/html/2608.22064v1
[S39]: https://openreview.net/pdf/a033cc4979b05be1f7f285f72a259e6378a133b1.pdf
[S40]: https://pubmed.ncbi.nlm.nih.gov/40651197/
