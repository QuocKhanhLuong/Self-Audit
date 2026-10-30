"""Deterministic mechanism witnesses; no patient efficacy claims."""
import copy
import json
import torch
from torch import nn

from self_audit.compact_control import apply_compact_update
from self_audit.compact_utility import compute_probe_kd_loss, validate_utility_probe


class TransferWitness(nn.Module):
    def __init__(self, nonfinite_probe=False):
        super().__init__()
        self.position = nn.Parameter(torch.tensor(0.0))
        self.nonfinite_probe = nonfinite_probe

    def forward(self, x):
        # Proposal step helps RV on the probe, but large steps flip audit RV to BG.
        is_probe = x[:, :1]
        bg = (1 - is_probe) * (4 * self.position)
        rv = (1 - is_probe) * 2 + is_probe * (2 * self.position)
        other = torch.full_like(rv, -8.)
        result = torch.cat((bg, rv, other, other), dim=1)
        if self.nonfinite_probe and bool(is_probe.any()) and self.position > .2:
            result = result * float('nan')
        return result


def fixture():
    x = torch.zeros(1, 1, 2, 2)
    y = torch.ones(1, 2, 2, dtype=torch.long)
    valid = torch.ones_like(y, dtype=torch.bool)
    def audit():
        yield {'x': x, 'y': y, 'valid': valid, 'patient':'L', 'frame':0}
    targets = torch.zeros(1, 4, 2, 2)
    targets[:, 0] = .05
    targets[:, 1] = .95
    probe = {'x': torch.ones_like(x), 'targets': targets, 'valid':valid,
             'confidence': .7, 'metadata': {'patient':'U_probe', 'crop': {'flip':False}}}
    candidates = {'directions':[{'position':torch.tensor(1.)}],
                  'source_ids':['pseudo_class_1'], 'denominator':4}
    return audit, probe, candidates


def test_highest_probe_gain_can_be_rejected_by_hard_dice():
    model = TransferWitness()
    audit, probe, candidates = fixture()
    frozen = validate_utility_probe(probe)
    base = {'position':torch.tensor(0.)}
    large_loss, _, _ = compute_probe_kd_loss(model, {'position':torch.tensor(1.)}, {}, frozen)
    ref_loss, _, _ = compute_probe_kd_loss(model, base, {}, frozen)
    receipt = apply_compact_update(model, candidates, method='scalar_guard', audit_batches=audit,
        config={'selector':'cross_patient_utility'}, utility_probe=probe)
    assert receipt['trials'][0]['per_class_dice_ok'] is False
    assert receipt['trials'][0]['probe_loss'] is None
    assert receipt['scale'] == .25
    assert model.position.item() == .25  # Single tested additive displacement.
    feasible = [t for t in receipt['trials'] if t['guard_passed']]
    chosen = next(t for t in feasible if t['accepted'])
    assert chosen['utility_gain'] == max(t['utility_gain'] for t in feasible)
    assert ref_loss - large_loss > chosen['utility_gain'] > 0
    assert receipt['macro_dice_after']['RV'] == receipt['base_macro_dice']['RV'] == 1.


def test_actual_nonfinite_trial_loss_yields_zero_and_json_null():
    model = TransferWitness(nonfinite_probe=True)
    audit, probe, candidates = fixture()
    before = copy.deepcopy(model.state_dict())
    receipt = apply_compact_update(model, candidates, method='scalar_guard', audit_batches=audit,
        config={'selector':'cross_patient_utility'}, utility_probe=probe)
    assert receipt['reason'] == 'nonfinite_probe_loss'
    assert receipt['status'] == 'zero_displacement'
    assert torch.equal(model.position, before['position'])
    json.dumps(receipt, allow_nan=False)


def test_frozen_probe_inputs_masks_and_targets_identical_during_trials(monkeypatch):
    import self_audit.compact_utility as utility
    calls = []
    original = utility.compute_probe_kd_loss
    def capture(model, params, buffers, probe, **kwargs):
        calls.append({k: probe[k].clone() for k in ('x','targets','valid')})
        return original(model, params, buffers, probe, **kwargs)
    monkeypatch.setattr(utility, 'compute_probe_kd_loss', capture)
    audit, probe, candidates = fixture()
    receipt = apply_compact_update(TransferWitness(), candidates, method='scalar_guard',
        audit_batches=audit, config={'selector':'cross_patient_utility'}, utility_probe=probe)
    assert len(calls) == 7  # Reference + six feasible nonzero trials; zero reuses reference.
    assert all(torch.equal(calls[0][k], c[k]) for c in calls for k in calls[0])
    identity = receipt['telemetry']['probe_identity_sha256']
    for trial in receipt['trials']:
        if trial['probe_loss'] is not None:
            assert trial['probe_identity_sha256'] == identity
            assert abs(sum(trial['probe_classwise_kd_loss'].values()) - trial['probe_loss']) < 1e-6


def test_cancelling_directions_are_zero_not_reject_all():
    audit, probe, candidates = fixture()
    candidates['directions'].append({'position':torch.tensor(-1.)})
    candidates['source_ids'].append('pseudo_class_0')
    receipt = apply_compact_update(TransferWitness(), candidates, method='scalar_guard',
        audit_batches=audit, config={'selector':'cross_patient_utility'}, utility_probe=probe)
    assert receipt['status'] == 'zero_displacement'
    assert receipt['forward_counts']['probe_trial_evaluations'] == 0
    assert receipt['accepted'] is False
