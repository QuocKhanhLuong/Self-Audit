import ast
from pathlib import Path

import pytest

from scripts.retry_completed_s0_backup import OLD_STATUS, repair_retry_source, require_completed_run


def test_backup_status_repair_keeps_selected_best_without_duplicate_kwargs():
    original = ('def retry_backup():\n'
                '    common={"best_step":None,"best_macro4":None}\n'
                '    best={"step":8000,"val_macro4":.8}\n'
                '    run_root="fixture"; result={"status":"VERIFIED"}\n'
                '    return _write_run_status(\n' + OLD_STATUS + '    )\n')
    repaired = repair_retry_source(original)
    ast.parse(repaired)
    namespace = {"_write_run_status": lambda root, status, **kw: {"root": root, "status": status, **kw}}
    exec(compile(repaired, "<fixture>", "exec"), namespace)
    result = namespace["retry_backup"]()
    assert result["status"] == "COMPLETED" and result["best_step"] == 8000
    assert result["best_macro4"] == .8
    with pytest.raises(ValueError, match="exactly one"):
        repair_retry_source(repaired)


@pytest.mark.parametrize("step,complete,status", [(9999, True, "UPLOAD_PENDING"),
                         (10000, False, "UPLOAD_PENDING"), (10000, True, "TRAINING")])
def test_backup_only_refuses_incomplete_or_running_jobs(tmp_path, step, complete, status):
    with pytest.raises(ValueError, match="already completed"):
        require_completed_run({"status": status, "training_complete": complete},
                              (tmp_path / "final", "a" * 64, step), "a" * 64)


def test_backup_only_binds_exact_existing_final_identity(tmp_path):
    final = tmp_path / "final"
    status = {"status": "UPLOAD_PENDING", "training_complete": True,
              "final_checkpoint": str(final), "final_checkpoint_manifest_sha256": "a" * 64}
    require_completed_run(status, (final, "a" * 64, 10000), "a" * 64)
    with pytest.raises(ValueError, match="identity"):
        require_completed_run(status, (final, "a" * 64, 10000), "b" * 64)
