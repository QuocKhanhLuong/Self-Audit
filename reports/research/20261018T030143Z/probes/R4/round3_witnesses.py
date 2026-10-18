"""Six finite-array witnesses only: no patient inputs, labels, weights or training."""
import hashlib
import json
import math


def close(a, b):
    assert abs(a - b) < 1e-12, (a, b)


def squared_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def rms_distance(target, prediction, weights=None):
    if weights is None:
        weights = [1.0] * len(target)
    mass = sum(weights)
    if mass == 0:
        return None  # Unsupported measure: not a zero-error certificate.
    return math.sqrt(sum(w * squared_distance(a, p)
                         for w, a, p in zip(weights, target, prediction)) / mass)


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


results = {}
proposal = [(0.1, 0.9), (0.1, 0.9)]
prediction = [(0.1, 0.9), (0.9, 0.1)]
weighted = rms_distance(proposal, prediction, [1.0, 0.0])
dense = rms_distance(proposal, prediction)
close(weighted, 0.0)
close(dense, 0.8)
assert rms_distance(proposal, prediction, [0.0, 0.0]) is None
results["support_measure_mismatch"] = {
    "weighted_rms": weighted, "full_grid_rms": dense,
    "empty_support_rms": None,
    "zero_weight_row_objective_unchanged": True,
    "meaning": "Weighted agreement does not bound the missing row."}

# Channel convention BG/RV/MYO/LV; exchange RV and LV jointly.
a, p = (0.1, 0.2, 0.3, 0.4), (0.2, 0.1, 0.5, 0.2)
permutation = (0, 3, 2, 1)
ap = tuple(a[i] for i in permutation)
pp = tuple(p[i] for i in permutation)
close(squared_distance(a, p), squared_distance(ap, pp))
close(entropy(a), entropy(ap))
results["joint_named_channel_permutation"] = {
    "distance_squared_before": squared_distance(a, p),
    "distance_squared_after": squared_distance(ap, pp),
    "entropy_before": entropy(a), "entropy_after": entropy(ap),
    "meaning": "Label-blind diagnostics cannot choose the biological naming."}

# This manufactured law defines the complete V; no hidden random target remains.
mapping = {"v0": (0.1, 0.9), "v1": (0.7, 0.3)}
rows = [(v, mapping[v]) for v in ("v0", "v0", "v1", "v1")]
residual = sum(squared_distance(t, mapping[v]) for v, t in rows) / len(rows)
close(residual, 0.0)
results["deterministic_same_input_target"] = {
    "conditional_residual_squared": residual,
    "meaning": "No conditional-averaging residual in this explicitly deterministic law."}

# Two equally likely hidden states share exactly the same deployed input V.
hidden_targets = ((0.1, 0.9), (0.9, 0.1))
mean_target = tuple(sum(t[k] for t in hidden_targets) / 2 for k in range(2))
floor_squared = sum(squared_distance(t, mean_target) for t in hidden_targets) / 2
candidate = (0.2, 0.8)
candidate_error = sum(squared_distance(t, candidate) for t in hidden_targets) / 2
close(floor_squared, 0.32)
close(candidate_error, floor_squared + squared_distance(candidate, mean_target))
assert hidden_targets[0] != hidden_targets[1]
results["identical_visible_input_hidden_proposals"] = {
    "V": "same_visible_input", "H_states": ["h0", "h1"],
    "targets": hidden_targets, "best_deterministic_prediction": mean_target,
    "minimum_transfer_mse": floor_squared,
    "minimum_transfer_rms": math.sqrt(floor_squared),
    "candidate_transfer_mse": candidate_error,
    "meaning": "A larger deterministic same-V student cannot distinguish H."}

# Duplicate records keep their original lineage and the total lineage weight.
source = {"source_A": (0.1, 0.9), "source_B": (0.7, 0.3)}
lineage_weights = {"source_A": 0.25, "source_B": 0.75}
original_records = [(name, target) for name, target in source.items()]
duplicate_records = original_records + [("source_A", source["source_A"])]


def lineage_mix(records):
    grouped = {}
    for name, target in records:
        if name in grouped:
            assert grouped[name] == target
        grouped[name] = target
    return tuple(sum(lineage_weights[name] * target[k]
                     for name, target in grouped.items()) for k in range(2))


assert lineage_mix(original_records) == lineage_mix(duplicate_records)
assert len({name for name, _ in duplicate_records}) == 2
results["exact_source_duplication"] = {
    "record_counts": [len(original_records), len(duplicate_records)],
    "distinct_lineages": 2, "lineage_weighted_target": lineage_mix(duplicate_records),
    "meaning": "Duplicated observations do not supply another independent lineage."}

artifact = {"shape": [2, 2], "channel_order": ["c0", "c1"],
            "pixel_order": ["pixel0", "pixel1"], "valid_support": [True, True],
            "training_weights": [1.0, 0.0], "proposal": proposal}
serialized_copy = json.loads(json.dumps(artifact))
changed_geometry = dict(artifact, pixel_order=["pixel1", "pixel0"])
assert digest(artifact) == digest(serialized_copy)
assert digest(artifact) != digest(changed_geometry)
results["frozen_proposal_geometry_receipt"] = {
    "original_sha256": digest(artifact),
    "copy_sha256": digest(serialized_copy),
    "changed_geometry_sha256": digest(changed_geometry),
    "meaning": "A checksum is a transfer identity check, not a correctness proof."}

print(json.dumps({"status": "MANUFACTURED_ARRAY_WITNESSES_ONLY",
                  "passed_checks": len(results), "checks": results},
                 indent=2, sort_keys=True, allow_nan=False))
