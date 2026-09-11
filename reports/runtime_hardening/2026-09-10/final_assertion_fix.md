# Final Assertion Fix — `test_max_steps_bounded_checkpoint_is_refused_for_resume`

**Date:** 2026-09-10
**Scope:** one test assertion in `tests/test_runtime_resume.py`. No production code changed.

## Symptom

Full suite on Python 3.10 / torch 2.4.1: **897 passed, 1 failed in 791.29s**. The single
failure was `tests/test_runtime_resume.py::test_max_steps_bounded_checkpoint_is_refused_for_resume`
at the `pytest.raises` guard.

The test expected the refusal message to match `non-resumable|incomplete`. The
`ValueError` actually raised is:

```
Cannot resume from <path>: 'resumable' is False, expected True.
```

## Diagnosis

Not a production bug. `resume_from_checkpoint` refuses the bounded checkpoint exactly as
the test intends — it raises `ValueError` and blocks the resume. Only the wording changed
when affirmative-flag validation was tightened: the guard now reports the specific flag and
its observed-vs-required value (`unified_trainer.py:1749`) instead of the older
"non-resumable" phrasing. The assertion regex was left stale by that refactor.

Stricter message is the better one — it names which flag failed and what was expected — so
the fix is to update the assertion, not the error text.

## Change

`tests/test_runtime_resume.py`, one line:

```diff
-    with pytest.raises(ValueError, match="non-resumable|incomplete"):
+    with pytest.raises(ValueError, match=r"'resumable' is False, expected True"):
```

`pytest.raises(ValueError, ...)` is retained, and every other guard in the test is
untouched — including the pre-resume payload check
`assert payload["resumable"] is False or payload["incomplete_epoch"] is True`, which still
proves the bounded `last.pt` is written as non-resumable before the refusal is exercised.

## Verification

Single-test rerun is owned by the root reviewer, who holds the full failing traceback and
runs the regression gate independently; it is deliberately not duplicated here.

## Non-goals honored

No production edits, no other tests modified, no new fixtures, no broad test runs, no
commits or pushes.
