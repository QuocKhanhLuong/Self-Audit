# SGSCN native protocol lock

Required datasets: PH2 and SYSU-US. Paper: https://arxiv.org/pdf/2107.04934.
Source: osmond332/Spatial_Guided_Self_Supervised_Clustering@
592efb6e72ceeef15c8be0630a4673eda5dce6f5, GPL-3.0.

Preserved official-code reference settings: 3-channel OpenCV BGR / 255 input;
100 channels; nConv=2; 1x1 final head; SGD momentum 0.9; CE coefficient 1;
spatial L1 coefficient 5; enabled context loss with epsilon=0.001; maxIter=50;
stop on pre-update active labels <= 3 after applying that update; final forward
in train mode. No softmax, clamping, altered normalization, loss repair, or class
count floor. PH2 learning rate=0.1; SYSU-US learning rate=0.05 (paper-supported).

Context-loss evidence: the paper (section 2.6) defines the overall loss as the
unweighted sum of cross-entropy, sparse spatial and context-based consistency
losses, so context loss is VERIFIED_PAPER with weight 1. In the official demo it
is enabled only by the non-default --center flag (README usage omits it). The
*_official_reference profiles therefore record reference_invocation = official
arithmetic with --center enabled; they are not the demo's default invocation.
The code spatial weight 5 (--stepsize_ss) is an official implementation detail
that conflicts with the paper's unweighted sum; this is an explicit paper gate.

Evidence audit 2026-10-01 (reports/dss_us_sgscn_paper_protocol_evidence_20261001.md)
adds: the paper writes Eqs. 1-2 as pixel sums while the code averages them (in addition
to the spatial weight 5); paper stopping has no criterion; SYSU-US sampling has no seed
or list; PH2 input format is unspecified. Track B: the overlap measure and ties are
unspecified, and reported HM values above 100% exclude the 1-Jaccard Hammoude form, so
HM/XOR cannot be reconstructed.

Paper profiles remain BLOCKED_PROTOCOL because architecture prose (three 3x3
convolutions versus the demo's final 1x1), the spatial loss weight and stability
stopping differ from the released demo. No undocumented stability criterion is
invented. Separate *_official_reference profiles are executable for verification;
they are explicitly official-code reference runs, NOT exact paper reproduction.

Paper Track B selects the cluster with largest GT overlap. DSC is specified;
exact overlap-tie handling, HM/XOR definitions and their normalization are not
fully evidenced. The original evaluator gate remains BLOCKED_PROTOCOL.
The Dice-only primitive is a partial diagnostic, never a full paper evaluator.

PH2 complete native scope: 200 studies. SYSU-US: 100 images, 5 per sequence from
20 sequences. Original sampled filenames/seed are not provided. No GT-based
sample filtering. Missing either dataset yields PARTIAL reproduction. Native
Track A adapter unavailability does not affect native reproduction completeness.

PAPER_FAITHFUL_REIMPLEMENTATION profiles (`*_paper_faithful.yaml`, src/sgscn/paper_faithful.py)
implement the paper text and equations literally and are separate from the official-code
reference profiles. Every setting the paper leaves open is a required field; they run only
after the user instantiates a new profile with explicit values
(scripts/instantiate_native_profile.py). See reports/dss_us_sgscn_paper_faithful_20261001.md.
