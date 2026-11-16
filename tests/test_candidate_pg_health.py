import json
import subprocess
from unittest.mock import patch

from scripts.pg_health import inspect_run, main


def test_failure_and_completed_both_wake_agent(tmp_path):
    for status in ('FAILED', 'COMPLETED', 'WAITING'):
        (tmp_path/'QUEUE_STATE.json').write_text(json.dumps({'status': status}))
        assert inspect_run(tmp_path)['needs_attention'] is True


def test_ssh_failure_does_not_silently_skip_repair():
    with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 255, stdout='', stderr='connection refused')):
        assert main([]) == 0


def test_only_verified_healthy_response_skips_agent():
    payload = json.dumps({'status': 'RUNNING', 'needs_attention': False})
    with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 3, stdout=payload, stderr='')):
        assert main([]) == 3
