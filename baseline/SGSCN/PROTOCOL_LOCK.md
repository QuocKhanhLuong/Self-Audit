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

Paper profiles remain BLOCKED_PROTOCOL because architecture prose and stability
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
