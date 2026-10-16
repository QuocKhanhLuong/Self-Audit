"""Full-size RANDOM architecture check; no data, checkpoints or downloads."""
from __future__ import annotations
import json, platform, time
from pathlib import Path
import torch, torchvision
from layer_upgrade.model import build_random_convnext_demo
from layer_upgrade.layers import block_mean
from layer_upgrade.losses import soft_ce


def main():
    torch.set_num_threads(2);torch.manual_seed(123)
    started=time.perf_counter()
    model=build_random_convnext_demo().eval()
    x=torch.rand(1,3,65,97)
    out=model(x)
    coarse=out['coarse_logits']
    q=torch.randn_like(coarse).softmax(1)
    loss=soft_ce(coarse,q,torch.ones_like(q[:,0]))
    # Coarse-only guidance has no path into DETAIL-ONLY head parameters.
    loss.backward()
    head_grad=sum(float(p.grad.abs().sum()) for p in model.detail_head.parameters() if p.grad is not None)
    assert head_grad==0
    model.zero_grad(set_to_none=True)
    out=model(x)
    q_native=torch.randn_like(out['logits']).softmax(1)
    loss_native=soft_ce(out['logits'],q_native,torch.ones_like(q_native[:,0]))
    loss_native.backward()
    native_detail_grad=sum(float(p.grad.norm()) for p in model.detail_head.parameters() if p.grad is not None)
    assert native_detail_grad>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result=dict(
        status='PASS_RANDOM_FULL_SIZE_FORWARD_BACKWARD_NOT_PRETRAINED_NOT_ANATOMY',
        source='torchvision ConvNeXt-Tiny random encoder plus newly implemented decoder/operators',
        parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        other_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.encoder.parameters()),
        inputs=list(x.shape),outputs={k:list(v.shape) for k,v in out.items()},
        coarse_projection_max_error=float((block_mean(out['logits_padded'])-out['coarse_logits']).abs().max().detach()),
        direct_coarse_loss_to_detail_head_gradient_sum=head_grad,
        native_loss_to_detail_head_gradient_norm_sum=native_detail_grad,
        feature_head_supervised_in_this_smoke=False,
        pretrained_weights_loaded=False,patient_arrays_read=False,
        optimizer_steps=0,device='cpu',torch=torch.__version__,torchvision=torchvision.__version__,
        python=platform.python_version(),elapsed_seconds=time.perf_counter()-started)
    Path('FULL_MODEL_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
