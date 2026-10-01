# Baseline-only historical-224 ACDC manifest

This v7 freeze preserves the verified v6 image-only source records, patient split, ED/ES selection, and all-Z inventory (train=1526, dev=376, test=0). It changes only the CUTS/DFC baseline input contract: the historical image preprocessing branch is reproduced at 224x224, followed by the loader volume normalization, with no 224-to-256 resize. No NIfTI/GT access or runtime algorithm workload occurred while creating this freeze. v6 checkpoints/raw/semantic artifacts remain retained but cannot be reused.
