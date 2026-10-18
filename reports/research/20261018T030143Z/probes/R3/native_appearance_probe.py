"""Manufactured CPU algebra/gradient probes; no patient, weights or dataset IO."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.layer_guided_student_v1.layer_upgrade.layers import zero_mean_detail
from experiments.layer_guided_student_v1.layer_upgrade.model import LayerGuidedStudent

torch.set_num_threads(1)
torch.manual_seed(3)
DTYPE = torch.float64

def h(x):
    return x - x.mean()

def case(name, x, y, mu_b, mu_m, t0=.25, eta=2.0):
    x = torch.tensor(x, dtype=DTYPE)
    y = torch.tensor(y, dtype=torch.bool)
    s = torch.zeros_like(x, requires_grad=True)
    a = (x-mu_b).square()-(x-mu_m).square()
    energy = (torch.sigmoid(t0+2*s)*a+(x-mu_m).square()).sum()
    grad, = torch.autograd.grad(energy, s)
    projected = h(grad)
    next_s = -eta*projected
    after = (torch.sigmoid(t0+2*next_s)*a+(x-mu_m).square()).sum()
    return {
        "name": name, "energy_before": energy.item(), "energy_after": after.item(),
        "projected_gradient_norm": projected.norm().item(),
        "detail_max_abs": next_s.abs().max().item(),
        "coarse_detail_mean": next_s.mean().item(),
        "binary_errors_before": int(((torch.full_like(x,t0)>0)!=y).sum()),
        "binary_errors_after": int(((t0+2*next_s>0)!=y).sum()),
        "projected_gradient": projected.tolist(),
    }

cases = [
    case("mixed_correct_noisy_anchor_means", [.25]*8+[.75]*8, [0]*8+[1]*8,.72,.28),
    case("mixed_swapped_anchors", [.25]*8+[.75]*8, [0]*8+[1]*8,.28,.72),
    case("constant_measurement", [.5]*16, [0]*8+[1]*8,.72,.28),
    case("collapsed_appearance_means", [.25]*8+[.75]*8, [0]*8+[1]*8,.5,.5),
    case("pure_blood_noise_creates_false_tissue", [.65]*8+[.85]*8, [1]*16,.75,.25,t0=.05),
    case("changed_anchor_midpoint", [.25]*8+[.75]*8, [0]*8+[1]*8,.82,.38),
]

# Explicit output-space != network-gradient-space counterexample.
x = torch.tensor([.25]*8+[.75]*8,dtype=DTYPE)
gradient_alignment = {}
for name, feature in [("constant_feature",torch.ones_like(x)),("intensity_feature",x)]:
    w = torch.zeros((), dtype=DTYPE, requires_grad=True)
    upstream = feature.clone().requires_grad_(True)
    detail = h(w*upstream)
    a = (x-.72).square()-(x-.28).square()
    e = (torch.sigmoid(.25+2*detail)*a).sum()
    gw, gf = torch.autograd.grad(e,(w,upstream))
    gradient_alignment[name] = {"head_weight_gradient":gw.item(), "upstream_gradient_norm":gf.norm().item()}

class TinyEncoder(nn.Module):
    channels = (8,8,8,8)
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([nn.Conv2d(3,8,4,4),nn.Conv2d(8,8,2,2),nn.Conv2d(8,8,2,2),nn.Conv2d(8,8,2,2)])
    def forward(self,x):
        out=[]
        for stage in self.stages:
            x=stage(x); out.append(x)
        return out

model = LayerGuidedStudent(TinyEncoder(),decoder_channels=(16,16,16,16),teacher_dim=4)
row = torch.tensor([.25,.25,.75,.75]*8)
image = row[None,None,None,:].expand(1,3,32,32).contiguous()
out = model(image)
logits = out["logits_padded"]
logits.retain_grad()
t = torch.logsumexp(logits[:,[1,3]],dim=1)-logits[:,2]
pix = image[:,1]
cost_b, cost_m = (pix-.72).square(), (pix-.28).square()
energy = (t.sigmoid()*cost_b+(1-t.sigmoid())*cost_m).mean()
energy.backward()
before = model.detail_head.weight.detach().clone()
grad_norms = {name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters() if name.startswith(("detail_head", "detail_stem.0", "detail_fuse.0"))}
projected_logit_grad_norm = float(zero_mean_detail(logits.grad,4).norm())
opt = torch.optim.SGD([model.detail_head.weight,model.detail_head.bias],lr=.1)
opt.step()
with torch.no_grad():
    later=model(image)
    later_t=torch.logsumexp(later["logits_padded"][:,[1,3]],dim=1)-later["logits_padded"][:,2]
    later_e=(later_t.sigmoid()*cost_b+(1-later_t.sigmoid())*cost_m).mean()
network = {
    "fixture":"current LayerGuidedStudent with manufactured tiny encoder, random local initialization",
    "parameters":sum(p.numel() for p in model.parameters()),
    "projected_logit_gradient_norm":projected_logit_grad_norm,
    "gradient_norms":grad_norms,
    "detail_weight_change_norm":float((model.detail_head.weight-before).norm()),
    "native_detail_norm_after":float(zero_mean_detail(later["bounded_detail"],4).norm()),
    "energy_before":float(energy.detach()),"energy_after":float(later_e),
}

assert cases[0]["binary_errors_after"] == 0
assert cases[1]["energy_after"] < cases[1]["energy_before"]
assert cases[1]["binary_errors_after"] > cases[1]["binary_errors_before"]
assert cases[2]["projected_gradient_norm"] == 0
assert cases[3]["projected_gradient_norm"] == 0
assert cases[4]["binary_errors_after"] > cases[4]["binary_errors_before"]
assert torch.allclose(torch.tensor(cases[0]["projected_gradient"]),torch.tensor(cases[5]["projected_gradient"]))
assert gradient_alignment["constant_feature"]["head_weight_gradient"] == 0
assert abs(gradient_alignment["intensity_feature"]["head_weight_gradient"]) > 0
assert network["detail_weight_change_norm"] > 0
assert network["native_detail_norm_after"] > 0
payload={"status":"MANUFACTURED_CPU_ONLY", "torch_version":torch.__version__,"cases":cases,"gradient_alignment":gradient_alignment,"network":network}
dest=Path(__file__).with_name("RESULTS.json")
dest.write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"result":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"case_summaries":[{k:v for k,v in c.items() if k!='projected_gradient'} for c in cases],"gradient_alignment":gradient_alignment,"network":network},indent=2))
