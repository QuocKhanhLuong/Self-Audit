"""Run the pinned ACDC_M3HL_train.py with a labeled-patient count that its hard-coded ``patients_to_slices`` table does not contain.

The table ``{"1": 32, "3": 68, "7": 136, ...}`` is the SSL4MIS ACDC patient -> slice map.  For M&Ms-1 with 10% labels (15 patients) the true ED+ES slice total
(recorded in H5_RECEIPT.json as ``labeled_slices``) is added as the key ``"<n>"`` by a literal substitution in the in-memory source; the pinned file on disk is
untouched and nothing else changes.  Usage: m3hl_acdc_run.py <H5_RECEIPT.json> <ACDC_M3HL_train.py> [script args]  (``--labelnum`` must equal ``n_labeled``).
"""
import json
import sys

receipt = json.load(open(sys.argv[1]))
script = sys.argv[2]
n, slices = int(receipt["n_labeled"]), int(receipt["labeled_slices"])
src = open(script).read()
needle = '"7": 136,'
assert src.count(needle) == 1, "unexpected patients_to_slices table"
src = src.replace(needle, f'"7": 136, "{n}": {slices},')
sys.argv = [script] + sys.argv[3:]
exec(compile(src, script, "exec"), {"__name__": "__main__", "__file__": script})
