# User-authorized FUSE larger teacher batch

Hardware profile on the largest native TRAIN patient057 (8,512,428), bf16, 2 warmup + 5 timed steps per case. This measures capacity/short throughput, not efficacy. T1: 2.1641s/frame, 9.00GiB allocated; T2: 2.1348s/frame, 16.62GiB. Measured difference about 1.4%, too small to claim a robust campaign speedup. T2 is selected to use available memory per user request, keeping native geometry. J2 failed OOM at 22.49GiB allocated, so J stays oneL+oneU. Initial J1 instrumentation failed with NameError in the probe helper; corrected and the full matrix rerun, preserving original result.

Implementation keeps frame-wise losses and averages over the true stacked batch; sampler groups two remaining ED/ES frames from one patient with a shared augmentation. Preserves50 teacher epochs/8000 frame exposures, but fewer optimizer steps; the latter is logged separately. J100 epochs/8000 paired updates unchanged. Explicit spec amendment records changed optimizer composition and augmentation correlation. Existing data, masks, geometry and model untouched.

SOFTWARE_ONLY: worker ran all tests/test_fuse_*.py, 70 passed in14.95s; check_repo PASS. Synthetic tests cover loss/gradient reduction, exactly-once/resumable sampler, phase-boundary splitting, and resumed batch2 training. Full historical pytest NOT_RUN under existing user speed amendment.

GPU campaign paused at committed T exposure1100, epoch7 (zero-based epoch6); model/optimizer/RNG state is preserved. Migration and resume receipts will record final deployment.

Corrected repeat (BATCH_PROBE_FIXED.json): T1 2.1511s/frame, T2 2.1239s/frame (about1.3%); same peak allocation. J1 largest L+U paired: PASS, 2.3660s/pair and17.11GiB allocated. J2 repeats OOM; retained1L+1U.

Migration at exposure1100 archived the original checkpoint/config/metrics/selected-teacher and verified all training payload except the declared config identity remains bit-identical by recursive tensor/RNG/optimizer fingerprint. First real CUDA resume found an existing RNG byte-tensor transport bug before any optimizer step; fixed in5f2485d by restoring RNG state on CPU. Ten focused resume tests passed; coordinator also ran15 batching/operations tests (15 passed in1.45s). No additional Claude reviewer was used.

Confirmed real CUDA resume on source5f2485d: teacher epoch8, exposure1124, optimizer step1120, physical batch2; new checkpoint saved at exposure1120. Supervisor campaign and420second watchdog running;600second Luna monitor enabled. Live window prints batch/examples/optimizer steps. Prior batch1 states and failed pre-step RNG attempt retained for audit. Raw timestamped evidence: BATCH_RESUME_RECEIPT.json. Old reports remain historical. Training is running; final VAL/TEST outcomes NOT_RUN.
