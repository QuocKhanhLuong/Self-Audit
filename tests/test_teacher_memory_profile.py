import pytest

from scripts.profile_teacher_memory import (
    _is_cuda_oom,
    operator_cost_counts,
    parse_arms,
    parse_batches,
    summarize_run,
)


def test_parse_bounded_profile_selection():
    assert parse_arms("A0,a3,A6") == ("A0", "A3", "A6")
    assert parse_batches("4, 8,16") == (4, 8, 16)
    with pytest.raises(Exception):
        parse_arms("A0,A0")
    with pytest.raises(Exception):
        parse_batches("4,0")


def test_summary_counts_actual_labeled_audit_and_unlabeled_examples():
    rows = [
        {
            "elapsed_seconds": 2.0,
            "supervised": {"samples": [{}, {}, {}]},
            "audit": {"samples": [{}, {}, {}]},
            "unlabeled": {"samples": [{}, {}, {}]},
        },
        {
            "elapsed_seconds": 3.5,
            "supervised": {"samples": [{}, {}]},
            "audit": {"samples": [{}, {}]},
            "unlabeled": {"samples": [{}, {}]},
        },
        {
            "elapsed_seconds": 5.0,
            "supervised": {"samples": [{}, {}]},
            "audit": {"samples": [{}, {}]},
            "unlabeled": {"samples": [{}, {}]},
        },
    ]
    report = {"supervised_updates": 3, "pseudo_accepted": 3, "ema_updates": 3}
    summary = summarize_run(rows, report, batch_size=2, warmup_updates=1)
    assert summary["measured_updates"] == 2
    assert summary["supervised_labeled_examples"] == 4
    assert summary["audit_labeled_examples_not_used_for_quality"] == 4
    assert summary["unlabeled_examples"] == 4
    assert summary["labeled_plus_unlabeled_examples_per_second"] == pytest.approx(8 / 3.0)
    assert summary["optimizer_step_counts"]["body_optimizer_steps"] == 6
    assert summary["optimizer_step_counts"]["head_optimizer_steps"] == 6
    assert summary["optimizer_step_counts"]["ema_updates"] == 3


def test_oom_is_classified_as_failure_not_a_profile():
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried allocating 1 GiB"))
    assert not _is_cuda_oom(RuntimeError("invalid config"))


def test_native_operator_cost_counts_match_shared_l_u_transaction_protocol():
    counts = operator_cost_counts(batch_size=4, measured_updates=2,
                                 total_updates=3, measured_seconds=5.0)
    assert counts["supervised_labeled_examples"] == 8
    assert counts["unlabeled_examples"] == 8
    assert counts["labeled_plus_unlabeled_examples"] == 16
    assert counts["labeled_plus_unlabeled_examples_per_second"] == pytest.approx(3.2)
    assert counts["optimizer_step_counts"] == {
        "supervised_transactions": 3,
        "pseudo_transactions": 3,
        "body_optimizer_steps": 6,
        "head_optimizer_steps": 6,
        "ema_updates": 3,
    }
