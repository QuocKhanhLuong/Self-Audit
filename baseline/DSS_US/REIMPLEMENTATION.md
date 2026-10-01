# Independent implementation boundary

DSS-US algorithm code in src/dss_us is newly written from published mathematics
and behavioral specifications. Unlicensed DSS-US / deep-spectral-segmentation
source is never vendored, imported or executed. No literal upstream code is used
as a parity fixture. Local references are fetched only for factual audits.

Licensed dependencies (NumPy/SciPy/scikit-learn/OpenCV/PyTorch/DINO) are used via
their public APIs. DINO can be loaded from an explicitly pinned Apache-2.0 checkout
and an explicitly hashed checkpoint; no mutable torch.hub download is permitted.

Do not describe equation-kernel smoke tests as paper-profile reproduction.
Incomplete profiles fail before reading data or allocating an output directory.
