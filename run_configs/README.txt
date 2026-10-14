Frozen Q/Z pilot diagnostic evaluator

evaluate_guidance_pilot.py is the adapter used for the completed VAL18 evaluation.
It loads private pilot weights with explicit plan/state binding checks, preserves
the original student model and export gates, and requires exact reproduction of
the original baseline labels before scoring. It never promotes pilot weights
to regular student artifacts. The executed adapter bytes are preserved.

The report and complete execution receipts are in:
../reports/guidance_signal_execution_20261014/README.txt

The adapter has two sequential modes for an execution-host job directory:
  python run_configs/evaluate_guidance_pilot.py GENERATE --job <private-job-dir>
  python run_configs/evaluate_guidance_pilot.py SCORE --job <private-job-dir>
The job must contain an original PROTOCOL.json with paths to the verified
original checkpoint, bank, recipe, images, pilot and baseline diagnostics.
Use a fresh job directory; existing generated/scored outputs are never replaced.
Keep the archived exporter backend environment when reproducing this run.
Sanitized report PROTOCOL/FROZEN copies are evidence, not runnable job configs.
Full-cine probability arrays can use several GB and are private artifacts.

Run the isolated software checks from the repository root:
  python -B -m pytest -q -p no:cacheprovider -c /dev/null --rootdir=. --confcutdir=. experiments/guidance_signal_20261014/tests run_configs/tests/test_pilot_evaluation.py
The runtime loads its eleven hash-pinned modules from the existing published
world-v2 source bundle. That bundle must remain unchanged.
