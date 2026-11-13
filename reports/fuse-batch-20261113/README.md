# User-authorized FUSE larger teacher batch

Hardware profile on the largest native TRAIN patient057 (8,512,428), bf16, 2 warmup + 5 timed steps per case. This measures capacity/short throughput, not efficacy. T1: 2.1641s/frame, 9.00GiB allocated; T2: 2.1348s/frame, 16.62GiB. Measured difference about 1.4%, too small to claim a robust campaign speedup. T2 is selected to use available memory per user request, keeping native geometry. J2 failed OOM at 22.49GiB allocated, so J stays oneL+oneU. Initial J1 instrumentation failed with NameError in the probe helper; corrected and rerun separately, preserving original result.

Implementation keeps frame-wise losses and averages over the true stacked batch; sampler groups two remaining ED/ES frames from one patient with a shared augmentation. Preserves50 teacher epochs/8000 frame exposures, but fewer optimizer steps; the latter is logged separately. J100 epochs/8000 paired updates unchanged. Explicit spec amendment records changed optimizer composition and augmentation correlation. Existing data, masks, geometry and model untouched.

SOFTWARE_ONLY: worker ran all tests/test_fuse_*.py, 70 passed in14.95s; check_repo PASS. Synthetic tests cover loss/gradient reduction, exactly-once/resumable sampler, phase-boundary splitting, and resumed batch2 training. Full historical pytest NOT_RUN under existing user speed amendment.

GPU campaign paused at committed T exposure1100, epoch7 (zero-based epoch6); model/optimizer/RNG state is preserved. Migration and resume receipts will record final deployment.
