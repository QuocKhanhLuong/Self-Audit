# Supervision and provenance ledger

| Item | Allowed source / implementation | Assumption and possible failure | Status |
|---|---|---|---|
| Input | Raw-shaped ACDC ED/ES NIfTI image intensities; all slices, no ROI crop | Expert phase availability; full cine not locally copied | Image contents inventoried and hashed |
| Historical collection | Prior script copied image pairs by enumerating `_gt` filenames | Mask-presence-dependent availability is indirect provenance, even with no mask-pixel crop | **Strict whole acquisition no-GT certification NOT established** |
| Independent cohort check | 100 Info.cfg files, only ED/ES frame numbers used for comparison | Dataset phase metadata is not a segmentation mask; it is still human-curated metadata | All 200 available images match the 100 patients' ED/ES declarations |
| Current split | Existing seed42 manifest, hash bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a | 80 train / 20 development patients, no new split | Patient disjointness checked |
| Preprocessing | Volume image percentiles 1/99; full FOV aspect-preserving resize/pad224 | MRI intensity is not standardized tissue density; thin MYO may disappear | No mask-derived crop, no slice exclusion |
| C0 features/init | Intensity, mean5, 0.10 normalized xy; quantile initialization; 15 Lloyd iterations | Appearance groups need not be anatomy; spatial feature is a weak positional prior | Fixed before results |
| Neural init | PyTorch random initialization, seeds17/29 | CNN inductive bias is not semantic supervision | No checkpoint download or foundation model |
| C1-cache targets | Fixed C0 anonymous partitions, sorted by image intensity | Consistent cluster IDs by appearance, not named anatomy; systematic teacher errors persist | No target refresh or GT matching |
| C1-direct target | Image itself, soft region means, TV and conditional entropy | Smoothness/low entropy can collapse or preserve wrong regions | No equal-area anatomical prior |
| R1 BG | Production ontology: groups with >=10% 1-pixel FOV border share; largest fallback | Blood/organ can touch FOV boundary, especially atypical acquisition | Fixed source rule, not calibrated correctness |
| R2 | Merge smallest foreground group into most adjacent until <=3 remain | Appearance over-clustering can erase useful distinctions | Reused unchanged; measured separately |
| R3 MYO/LV | Holes under 4-connectivity, enclosure >=0.90, unique strongest pair | Complete ring is not universal at base/apex; one-pixel gap can destroy enclosure | Handwritten SAX prior; ring-gap test on actual resolver |
| R4 RV | Remainder adjacent to MYO | Non-cardiac tissue may be adjacent; no guarantee of RV | Handwritten relationship prior |
| R5 intensity | Darker draft MYO, bright pools, unresolved when no enclosure | Intensity ambiguity, pathology/bias/acquisition variation | Drafts not exported as confident labels |
| Orientation | Not supplied to new mapper; no guessed left/right axis | Cannot resolve all LV/RV ambiguities | Explicitly absent |
| Unknown | Validity0 becomes exported ID255, with separate validity NIfTI | Abstention can hide errors if metric ignores it | Primary Dice counts unknown as missed GT anatomy |
| Prior penalty | Current ontology returns constant zero, keeps diagnostic violations | Not an active candidate scorer | Intentional implementation, not declared a bug |
| Checkpoint | Fixed update1024, last only | May be underfit; no inference of impossibility | No GT selection |
| Baseline choice | Preregistered reproduction/coverage/latency rule | Engineering admission does not certify correct anatomy | Selection recorded before reference scoring |
| References | Separate evaluator after hashes/config/mapping/checkpoints frozen | Development labels have already informed earlier project work | This is setting (a) method mask-free + (b) label-observed research; **not (c)** |

The firewall logs actual opens and blocks reference data filenames. An image-only
mirror removes paired reference files from the training input tree. This is a defense
against accidental reads, not proof against arbitrary malicious relabeling of a mask
as an image, and not proof of historically clean preprocessing. Raw image hashes are
checked against the prior copy inventory; no historical pixel crop was present in the
copy script, but the frame-availability issue above remains disclosed.

No atlas, GT Hungarian assignment, clinical quality head, pretrained encoder, auditor,
or hidden recurrent state enters the new method. Oracle matching in evaluation is
reported only as a partition diagnostic and never saved as output or training target.
