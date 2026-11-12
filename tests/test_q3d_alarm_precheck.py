"""SOFTWARE_ONLY: SSH errors must request inspection, never look healthy."""
import subprocess

import pytest

from scripts import q3d_alarm_precheck as alarm


@pytest.mark.parametrize('code,stdout,expected', [
    (1, '{"attention_required":false}', 1),
    (0, '{"attention_required":true}', 0),
    (1, '', 0),  # remote Python exception is also exit 1
    (255, '', 0),
    (0, '{"attention_required":false}', 0),  # contradictory receipt
])
def test_precheck_requires_valid_remote_receipt(monkeypatch, code, stdout, expected):
    monkeypatch.setattr('sys.argv', ['q3d_alarm_precheck.py'])
    monkeypatch.setattr(alarm.subprocess, 'run',
                        lambda *args, **kwargs: subprocess.CompletedProcess([], code, stdout, ''))
    assert alarm.main() == expected
