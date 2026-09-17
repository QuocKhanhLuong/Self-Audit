# Unit and software evidence

TESTED on local CPU using `/Users/alvinluong/miniforge3/bin/python`, Torch 2.14.0. No CUDA was used. Initial complete implementation: **14 passed in 1.42 s**. After the periodic-evaluation fix/test: **15 passed in 1.36 s**. Final combined implementation: **16 passed in 3.65 s**, including the full trainer GT-permutation test. Receipts: `evidence/unit_final.txt` and `evidence/unit_final.xml`.

Command:

```sh
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_event_audit.py -q
```

The suite checks no GT/target/error-map arguments on Auditor/Trigger/RF APIs; rejection of raw label targets; cloned detached observations; unchanged RF targets under GT permutation; gradients in both intended learners with no cross-loss leakage; frozen anchor; exact never-audit identity and zero q invocations; always-audit behavior; positive/negative learned-value fixtures and batch caps; guidance changing coordinates/output; fixed coordinates blocking the guidance effect; nonzero controller gradients; identical twin starting states; no history leakage; deterministic expiring replay; detached oracle metrics; semantic class-permutation blindness; no corruption metadata field; separate invocation/update counters; tie-aware ranking and censored thresholds; complete-case/patient aggregation; periodic evaluation schedule; and a full trainer GT-permutation comparison.

Software PASS does not establish that the learned score ranks true improvement, that source sensitivity is beneficial, that either action occurs at useful rates on real data, or that training converges faster. Those require the bounded measurements in 10–12.
