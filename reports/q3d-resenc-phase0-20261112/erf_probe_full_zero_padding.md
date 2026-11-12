# ResEnc-L empirical receptive-field probe

This CPU-only probe used deterministic random Z-score inputs and the locked encoder checkpoint. It is architecture evidence, not a patient experiment.

Status: **REVIEW_MEAN_OFFSET_OVER_THRESHOLD**. Samples: 16. Absolute mean centroid offset from nominal main phase (per-axis max): **2.5820 voxels**; threshold: 2.0000.

Mean centroid offset (Z,Y,X): `[0.7005635499954224, 2.5819993019104004, 2.4083452224731445]`; maximum individual absolute offset (reported, not gated): `3.8777`. Mean r90: `46.9202` voxels; mean r95: `51.2134` voxels.

Logical probe shape: `[33, 97, 97]`; the full six-stage teacher shape contract pads high-side to `[34, 128, 128]` with `[1, 31, 31]` voxels. The selected logical stage-3 cell remains `[8,6,6]` with nominal phase `[16,48,48]`.

Channel aggregation is the signed mean of all stage-3 channels before differentiation; the centroid uses absolute input gradients. The full per-sample receipt, checkpoint hash, registry hash, limitations, and exact command parameters are in `erf_probe.json`.
