# Native-volume smoke deadline repair

Date: 2026-10-26
Scope: `scripts/volume_smoke.py`, `tests/test_volume_cli.py`

## Repair

The smoke now checks the cooperative deadline immediately after every trainer
transaction, after the final functional forward, before and after the optional
correction/export witness, and before teacher allocation and after the final
teacher forward. CUDA checks synchronize before reading the clock, so the final
`elapsed_seconds` receipt is measured after all witnesses and device work have
finished. The timeout is deliberately cooperative: a kernel already executing
cannot be preempted; the next synchronized check fails closed once it returns.

The earlier clock-controlled reproduction (`perf_counter` values
`0, 0, 1000`) now raises `TimeoutError` at `after transaction 0` instead of
returning `PASS` with an elapsed value beyond the configured one-second bound.

## Owned verification

```text
python -m pytest -q tests/test_volume_cli.py
19 passed in 7.70s
```

The suite includes no-sleep clock-controlled timeout tests for the final
transaction, correction witness, and teacher witness, plus a real bounded CPU
smoke assertion that the final elapsed receipt is finite and positive. Existing
NaN and oversized runtime-override rejection tests remain in the same suite.

Source hashes at this snapshot:

```text
f76dff0c01a83a1a49d21367e459fa526a19096aa6cc6f07907f91bbe747c849  scripts/volume_smoke.py
c573b35aaaaeaac3e3e38ca6963af9c1c0d48b2a194c6da2177a651684ba28f4  tests/test_volume_cli.py
```

No full patient training, GPU run, download, server action, or full test suite
was run in this scoped repair.

## Follow-up protocol guard

The smoke now validates the explicit correction sidecar before model
construction. I00 and I01 fail closed when the sidecar is omitted or when its
protocol, mode, dataset role, or optional click/supervision budget disagrees
with the row's declared `correction` section; no filename is consulted to infer
an algorithm. Rows without a declared correction (such as V04) may opt in with
an explicit sidecar, and the receipt records sidecar protocol/mode/role/budget
alongside the absence of a row-level declaration.
Because this command is a manufactured fixture smoke, every accepted sidecar
must declare `dataset_role="FIXTURE"`; TRAIN correction sidecars fail before
model allocation.

The follow-up CLI/test evidence is bounded synthetic CPU software evidence:

```text
python -m pytest -q tests/test_volume_cli.py
24 passed in 4.69s

I01 without --correction-config: FAIL before builder, explicit matching sidecar required
I01 with I00_correction.json: FAIL before builder, protocol mismatch
I00 with I00_correction.json: PASS, prompt_count=0, supervision_budget=0
I01 with I01_correction.json: PASS, prompt_count=1, supervision_budget=1
```

The matched I00/I01 receipts share frozen student probability SHA-256
`62ae6615bc3143c68af60399e1cab60e7ce8f410656974df8afb4251028d69a0`; I00
preserves that prior (zero prompts), while I01 applies one fixture click. The
updated source hashes are:

```text
5cb0a63295b58a8cac7658e93869974812ac6c64143a9192273762a5919b8225  scripts/volume_smoke.py
fa5147da073e76ba70a464280c539d36a1028176e35d5ffe059f86c9ee0668a6  tests/test_volume_cli.py
```
