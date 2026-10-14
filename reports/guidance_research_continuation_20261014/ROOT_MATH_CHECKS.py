"""Tiny mathematical sanity checks only; no images, references, model or training."""
import json
import math
from pathlib import Path

def softmax(z):
    m = max(z)
    es = [math.exp(t-m) for t in z]
    return [e/sum(es) for e in es]

def ce(z, q, w, denominator):
    return -w * sum(a*math.log(b) for a,b in zip(q,softmax(z))) / denominator

q = [.4,.2,.2,.2]
p = [.25]*4
entropy = -sum(x*math.log(x) for x in q)
w = 1-entropy/math.log(4)  # reliability=1; same entropy weighting form
assert w > 0
# A second positive-weight foreground target activates the real batch rule.
# The first pixel alone would be inactive because its target argmax is BG.
foreground_q = [.1,.1,.7,.1]
foreground_entropy = -sum(x*math.log(x) for x in foreground_q)
foreground_weight = 1-foreground_entropy/math.log(4)
batch_active = foreground_weight > 0 and max(range(4), key=foreground_q.__getitem__) > 0
assert batch_active
denominator = max(w+foreground_weight,1e-8)
gradient = [w*(a-b)/denominator for a,b in zip(p,q)]
uniform_gradient = [w*(a-.25)/denominator for a in p]
centered_target = [w*(a-.25)/denominator for a in q]
assert all(abs(a-(b-c)) < 1e-15 for a,b,c in
           zip(gradient,uniform_gradient,centered_target))
eps = 1e-5
finite = []
for c in range(4):
    plus,minus = [0.]*4,[0.]*4
    plus[c],minus[c] = eps,-eps
    finite.append((ce(plus,q,w,denominator)-ce(minus,q,w,denominator))/(2*eps))
assert max(abs(a-b) for a,b in zip(gradient,finite)) < 1e-9
assert w*q[1] > 0 and gradient[1] > 0
# Positive derivative -> gradient descent decreases RV logit from uniform p,
# despite positive RV target mass. This is no claim about actual anatomy.
loss_at_teacher = -sum(a*math.log(a) for a in q)
kl_at_teacher = sum(a*math.log(a/b) for a,b in zip(q,q))
assert abs(loss_at_teacher-entropy-kl_at_teacher) < 1e-15
assert max(q) < .7
# Support-only overlap bounds from published rounded disagreement fractions.
native,projected,disagreement = 215428,121350,119330
lower = math.ceil((native+projected-disagreement)/2)
upper = min(native,projected,native+projected-disagreement)
assert 0 <= lower <= upper
result = {
    "kind":"MATHEMATICAL_SANITY_CHECK_NOT_SEGMENTATION_EVIDENCE",
    "source_sha":"b937e3b746608725e43d3bc3cef8861666d152dc",
    "soft_mass_counterexample":{
        "q":q,"p":p,"entropy":entropy,"weight":w,
        "other_pixel_q":foreground_q,"other_pixel_weight":foreground_weight,
        "batch_active":batch_active,"batch_denominator":denominator,
        "rv_target_mass":w*q[1],"rv_logit_gradient":gradient[1],
        "max_finite_difference_error":max(abs(a-b) for a,b in zip(gradient,finite)),
        "teacher_match_kl":kl_at_teacher,"first_pixel_teacher_match_accepted":False,
        "interpretation":"Positive RV target mass can accompany suppression of RV logit; it is not evidence of a named RV-favoring region."
    },
    "baseline_selected_batch_overlap":{
        "native_accepted":native,"projected_accepted":projected,
        "reconstructed_label_disagreement":disagreement,
        "accepted_support_intersection_lower":lower,
        "accepted_support_intersection_upper":upper,
        "native_only_lower":native-upper,"native_only_upper":native-lower,
        "native_only_fraction_lower":(native-upper)/native,
        "native_only_fraction_upper":(native-lower)/native,
        "scope":"80 selected TRAIN batches, original checkpoint, support membership only, no anatomical scoring"
    },
    "checks_passed":True
}
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

