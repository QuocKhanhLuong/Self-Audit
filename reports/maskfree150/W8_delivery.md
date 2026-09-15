# W8 delivery — independent leakage and scientific red team

Worker: W8, dispatched through Orca, task `task_3ccf6465970a`, dispatch `ctx_1197c0a977c6`.
Requested provider/model: Opus (reassigned from AGY per the architecture contract).
Resolved provider/model as observed from this session: **not exposed to the worker.**
The session identifies itself as Claude Opus 5 (`claude-opus-5`) in its own runtime
preamble; that is a self-report, not launch-transcript provenance, so the
coordinator should take the resolved model from the actual Orca launch record.

Verdict: **REVISE**. Full evidence in `reports/maskfree150/red_team.md`.

## Files changed

| Path | Status | Note |
| --- | --- | --- |
| `tests/test_maskfree_firewall.py` | new, 220 lines | owned; three adversarial firewall tests |
| `reports/maskfree150/red_team.md` | new | owned; findings, evidence, limitations |
| `reports/maskfree150/W8_delivery.md` | new | this file |

No production file was read-write. No other worker's file was touched. Nothing was
staged, committed or pushed. No GPU, no remote host, no worktree, no real data.

## Focused checks (3, per budget)

```
$ cd /Users/alvinluong/Self-Audit
$ PYTHONPATH=.:src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_firewall.py -rxX -q
xxx                                                                      [100%]
XFAIL test_target_informed_candidate_cannot_win_the_selection_score
XFAIL test_sealed_verify_observations_cannot_be_scored_on_demand
XFAIL test_scoring_support_must_be_disjoint_from_the_fitted_support
3 xfailed in 0.47s
```

Each test asserts the firewall invariant the contract promises. All three
invariants fail today, so they are recorded as `xfail(strict=True)`: the suite
stays green, the gap stays visible and owned, and the suite turns **red** the
moment an owner implements the guard (XPASS under `strict`), which is the signal
to drop the marker and keep the test as a regression guard. No test asserts that
today's behaviour is acceptable.

## Findings, with owners

| ID | Severity | Where | Owner | Reproduced measurement |
| --- | --- | --- | --- | --- |
| R1 | CRITICAL | `contracts.py:64`, `observation.py:360` | coordinator + W2 | target-informed 3-pixel edit gains **0.011719 nats/pixel** on `O_select`, 11.7× the 0.001 adoption threshold, while scoring *worse* on the fit side |
| R2 | HIGH | `contracts.py:46`, `observation.py:630` | coordinator + W6 | `role="verify"` scored **10×** with no freeze receipt, no counter, identical totals |
| R3 | HIGH | `observation.py:360/630` | W1 | scoring a fit on its own fitting support is accepted: **-0.170704 (n=768)** vs honest **-0.120941 (n=256)**, a 0.049763 nats/pixel spurious advantage |
| R4 | MEDIUM | `contracts.py:26/29` | coordinator + W3 | `FittingView.metadata` accepts `{'withheld_mean': ..., 'withheld_std': ...}`; contract line 41 promises a whitelist, code has none |
| R5 | LOW | `observation.py:407-417` | W1 | reported fit NLL is one half-step stale: components from bias *k−1*, likelihood at bias *k* |

R1 is the finding that matters scientifically: nothing in the frozen API records
where a candidate partition came from, so the `O_fit` / `O_select` firewall has
**zero runtime enforcement** and rests entirely on W2's and W3's discipline. A
single candidate-generation bug turns "predictive evidence improves labels" into
"the selector saw the answer", with every downstream metric still looking healthy.
Concrete fixes for R1–R4 are proposed in `red_team.md`, each local to one owner.

## What the owners got right

`fit` takes no scoring argument at all, so target blindness is structural rather
than a check; `FittingView.validate` physically verifies zeros in all three context
channels; the recipe and hypothesis-digest guards W1 added mid-review close
unequal-capacity comparison and post-fit mutation; validity is excluded from the
scored likelihood; missing support is `available=False` with a reason, never zero;
`cine_predictive` raises instead of silently degrading. No `self_audit` or
`self_audit_candidate_c` import, no `pretrained` flag, no `torch.hub`, no download,
no external checkpoint anywhere in the package.

## Coverage — stated truthfully

The package grew during this review. Inspected: `contracts.py`, `observation.py`
(619 → 750 lines mid-review; all findings re-verified against the 750-line version).
Forbidden-dependency grep only: `ontology.py`, `models.py`, `config.py`,
`runtime.py`, which landed after the test budget was spent. **Absent and therefore
untested:** `hypotheses.py`, `auditor.py`, `data/`, `experiments.py`, `evaluation/`,
`export.py`, launchers and configs.

So the prompt's remaining leak targets — mask-path reads, GT-selected
crop/split/threshold/checkpoint, stale bank/source/partition hashes, misreported
compute, covered-pixel-only metric inflation, misleading 150-epoch completion,
student budget parity — run through files that do not exist yet. They are **NOT YET
TESTABLE**, not passing. A second W8 pass is required once W2/W3/W6/W7 land.

## Operational note for the coordinator

W1 changed the canonical import path mid-review from `src.self_audit_maskfree.*` to
`self_audit_maskfree.*` with `PYTHONPATH=.:src`; the firewall tests follow the new
convention. There is no `conftest.py`, `pytest.ini` or `pyproject.toml` in the
repository, so **every maskfree test needs `PYTHONPATH=.:src` explicitly (coordinator-confirmed canonical path)** and a plain
`pytest tests/` collection fails. Worth a coordinator-owned `conftest.py` before the
combined integration gate.

## Limitations

**Software:** three tests by assignment, so absence of a fourth finding is not
evidence of absence; no integration path was exercised — no trainer, bank, auditor,
export or freeze; two-thirds of the package does not exist yet.

**Scientific:** all evidence is CPU synthetic phantoms with well-separated Gaussian
region means, which is software evidence about the implemented equations and
nothing more. No real data, no GPU, no 150-epoch run, no reference masks exist in
this checkout, so nothing here bears on Dice, coverage, semantic resolution, or
whether the audited student beats the unaudited one. The measured exploit
magnitudes establish that the guards are absent and that their absence is
decision-relevant at the contract's own threshold; they do not predict effect sizes
on ACDC or M&Ms. This review checks the implemented firewall against the written
contract; it does not validate the contract, and it is not evidence for or against
the underlying hypothesis.

## Coordinator follow-ups processed

Two messages were received and honoured before completion:

- *Shared integration constraints* — canonical import path `self_audit_maskfree` with
  `PYTHONPATH=.:src`, no `src.` prefix (duplicate dataclass identities). The firewall
  tests and every command quoted above use it. No anatomy score penalty was added;
  `prior_penalty` default 0 is used as given, and `background_modes=1` is the primary
  configuration exercised.
- *Review timing* — the overall red-team verdict is **NOT settled as PASS**. It is
  REVISE with the untested surfaces listed explicitly as pending, and a second W8 pass
  over the full dependency integration is requested once `hypotheses.py`, `auditor.py`,
  `data/`, `experiments.py`, `evaluation/`, `export.py` and the launchers land. The W1
  follow-up `ctx_2438a3990031` (mutable fitted-hypothesis alias, recipe mismatch) had
  already landed when this review re-verified against the 750-line `observation.py`;
  those two guards are credited above and are not among R1–R5.
