"""Bounded CPU interventions on random weights. No medical data or GT is opened."""
import copy
import hashlib
import json
import platform
from pathlib import Path

import torch
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher


def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = CinePseudoTeacher().eval()
    cur = torch.rand(2, 3, 64, 64)
    prev, nxt = torch.roll(cur, -2, -1), torch.roll(cur, 2, -1)
    with torch.no_grad():
        reference = model(prev, cur, nxt)
        results = {}

        def record(name, output):
            results[name] = {
                'accepted_fraction': output['valid'].float().mean().item(),
                'unknown_fraction': (~output['valid']).float().mean().item(),
                'soft_prob_mean_abs_change': (output['soft_label'] - reference['soft_label']).abs().mean().item(),
                'soft_prob_max_abs_change': (output['soft_label'] - reference['soft_label']).abs().max().item(),
                'argmax_change_fraction': (output['soft_label'].argmax(1) != reference['soft_label'].argmax(1)).float().mean().item(),
                'pseudo_mask_change_fraction': (output['pseudo_label'] != reference['pseudo_label']).float().mean().item(),
                'motion_mean_abs_change': (output['motion_features'] - reference['motion_features']).abs().mean().item(),
            }

        record('baseline', reference)
        for name, hook in [('zero_motion_descriptor', lambda m, a, o: torch.zeros_like(o)),
                           ('shuffle_motion_between_cases', lambda m, a, o: o.flip(0))]:
            handle = model.motion.register_forward_hook(hook)
            record(name, model(prev, cur, nxt))
            handle.remove()
        record('zero_temporal_differences', model(cur, cur, cur))
        record('swap_temporal_neighbours', model(nxt, cur, prev))
        shuffled = cur.clone()
        shuffled[:, [0, 2]] = cur.flip(0)[:, [0, 2]]
        record('shuffle_z_neighbours_keep_center', model(prev, shuffled, nxt))
        record('zero_evidence', model(prev, cur, nxt, evidence_logits=torch.zeros(2, 12, 4)))
        record('random_evidence', model(prev, cur, nxt, evidence_logits=3 * torch.randn(2, 12, 4)))
        evidence = torch.zeros(2, 12, 4)
        evidence[..., 3] = 20
        record('strong_LV_evidence', model(prev, cur, nxt, evidence_logits=evidence))

        # An implemented function can remain unchanged under prototype permutation.
        permuted = copy.deepcopy(model)
        permuted.regions.p.copy_(model.regions.p.flip(0))
        q_perm = permuted(prev, cur, nxt)
        prototype_invariance = (q_perm['soft_label'] - reference['soft_label']).abs().max().item()
        # Named output rows, however, can be arbitrarily relabelled with no image SSL cost.
        class_permutation = [0, 3, 2, 1]
        permuted = copy.deepcopy(model)
        permuted.semantic[-1].weight.copy_(model.semantic[-1].weight[class_permutation])
        permuted.semantic[-1].bias.copy_(model.semantic[-1].bias[class_permutation])
        c_perm = permuted(prev, cur, nxt)
        semantic_equivariance = (c_perm['soft_label'] - reference['soft_label'][:, class_permutation]).abs().max().item()

        # Force four equally weighted, individually confident but conflicting regions.
        ambiguity = CinePseudoTeacher(k=4).eval()
        ambiguity.semantic[-1].weight.zero_()
        ambiguity.semantic[-1].bias.zero_()
        handle = ambiguity.regions.register_forward_hook(lambda m, a, o: torch.full_like(o, .25))
        conflicting = 40 * torch.eye(4).unsqueeze(0).expand(2, -1, -1)
        bad = ambiguity(prev, cur, nxt, evidence_logits=conflicting)
        handle.remove()
        ambiguity_result = {
            'accepted_fraction': bad['valid'].float().mean().item(),
            'mean_dense_max_probability': bad['soft_label'].max(1).values.mean().item(),
            'mean_dense_margin': bad['soft_label'].topk(2, 1).values.diff(dim=1).abs().mean().item(),
        }
    source = Path('src/self_audit_pseudolabel/system_v3.py')
    result = {
        'evidence_kind': 'synthetic_random_weights_only',
        'seed': 17, 'torch': torch.__version__, 'platform': platform.platform(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'interventions': results,
        'prototype_permutation_dense_max_error': prototype_invariance,
        'named_class_permutation_equivariance_max_error': semantic_equivariance,
        'conflicting_confident_regions': ambiguity_result,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
