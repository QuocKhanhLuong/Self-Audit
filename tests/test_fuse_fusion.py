import torch
import pytest

from self_audit.candidate_q3d.alignment import align_teacher_to_student4

from self_audit.candidate_fuse.fusion import GatedFusion, align_feature
from self_audit.candidate_fuse.student import FuseStudent, sample_keep_mask, teacher_dropout_probability
from self_audit.candidate_fuse.teacher import build_teacher
from self_audit.candidate_fuse.config import default_config


def _config():
    return {"model": {"student": {"channels": [4, 8, 12, 16, 32]}}}


def _geometry(shape):
    return {"shape": shape, "spacing_zyx": (1., 1., 1.), "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}


def test_gate_initialization_and_detached_teacher():
    fusion = GatedFusion(4, 6)
    assert torch.count_nonzero(fusion.gate.weight) == 0
    assert torch.all(fusion.gate.bias == -2)
    assert torch.allclose(torch.sigmoid(fusion.gate.bias), torch.full_like(fusion.gate.bias, torch.sigmoid(torch.tensor(-2.)).item()))
    x = torch.randn(2, 4, 2, 3, 3, requires_grad=True)
    teacher = torch.randn(2, 6, 2, 3, 3, requires_grad=True)
    mask = torch.ones(2, 1, 1, 1, 1)
    y, gate = fusion(x, teacher, mask)
    y.sum().backward()
    assert gate.shape == x.shape
    assert teacher.grad is None
    assert x.grad is not None


def test_student_pure_and_all_dropped_paths_and_mimic():
    torch.manual_seed(3)
    model = FuseStudent(_config(), teacher_channels=(4,8,12,16,32))
    model.train()
    image = torch.rand(2,1,5,17,19)
    geometry = _geometry((5,17,19))
    pure = model(image, geometry)
    teachers = tuple(torch.randn(2,c,*s,requires_grad=True) for c,s in zip((4,8,12,16,32),((5,17,19),(5,9,10),(5,5,5),(3,3,3),(3,2,2))))
    dropped = model(image, geometry, teacher_features=teachers, keep_mask=torch.zeros(2,1,1,1,1))
    assert torch.equal(pure["logits"], dropped["logits"])
    assert torch.isfinite(dropped["mimic_loss"])
    assert dropped["mimic_loss"].item() >= 0
    dropped["mimic_loss"].backward()
    assert all(t.grad is None for t in teachers)
    fused = model(image, geometry, teacher_features=tuple(t.detach() for t in teachers), keep_mask=torch.ones(2,1,1,1,1))
    assert fused["mimic_loss"].item() == 0
    assert len(fused["features"]) == len(fused["gate_means"]) == 5
    for bad in (torch.full((2,1,1,1,1), 0.5), torch.full((2,1,1,1,1), float("nan"))):
        with pytest.raises(ValueError, match="keep_mask"):
            model(image, geometry, teacher_features=teachers, keep_mask=bad)


def test_dropout_schedule_and_phase_alignment_odd_native():
    assert teacher_dropout_probability(0, 5) == 0
    assert teacher_dropout_probability(4, 5) == 0.5
    assert sample_keep_mask(8, 4, 5, device="cpu").shape == (8,1,1,1,1)
    teacher = torch.arange(1*2*3*5*5,dtype=torch.float32).reshape(1,2,3,5,5)
    aligned=align_feature(teacher,(5,9,9),((2,2,2),(0.,0.,0.)),((2,4,4),(0.,0.,0.)),(3,3,3))
    assert aligned.shape == (1,2,3,3,3)
    assert torch.equal(aligned[:,:,0,:2,:2], teacher[:,:,0,::2,::2][:,:,:2,:2])


@pytest.mark.parametrize("shape", [(33, 65, 67), (35, 67, 69)])
def test_actual_student_and_teacher_phase_coordinate_probe(shape):
    config = default_config(synthetic=True)
    model = FuseStudent(config, teacher_channels=(4,8,12,16,20), include_fusion=False, include_mimic=False).eval()
    teacher, _ = build_teacher(config, synthetic=True)
    teacher.eval()
    d,h,w=shape
    z,y,x=torch.meshgrid(torch.arange(d),torch.arange(h),torch.arange(w),indexing="ij")
    ramps=torch.stack((z,y,x),dim=0).unsqueeze(0).float()
    with torch.no_grad():
      # Centered box filters make each channel a coordinate ramp after a
      # convolution, independent of random learned weights.
      def box_normalize(module):
        volume=module.in_channels*module.kernel_size[0]*module.kernel_size[1]*module.kernel_size[2]
        module.weight.fill_(1.0 / volume)
        if module.bias is not None: module.bias.zero_()
      for module in model.modules():
        if isinstance(module,torch.nn.Conv3d): box_normalize(module)
      for module in teacher.model.encoder.modules():
        if isinstance(module,torch.nn.Conv3d): box_normalize(module)
      student_feats=[]
      for ramp in ramps[0]:
        src=ramp[None,None]
        padded,_=model._pad(src)
        out=[model.stem[0].conv(padded)]
        out.append(model.stage1[0].conv(out[-1])); out.append(model.stage2[0].conv(out[-1]))
        out.append(model.stage3.down.conv(out[-1])); out.append(model.stage4.down.conv(out[-1]))
        student_feats.append(out)
      # Run the actual student and teacher main-path convolution chains.
      teacher_feats=[]
      encoder=teacher.model.encoder
      for ramp in ramps[0]:
        padded,_=teacher.model._pad_high(ramp[None,None])
        out=encoder.stem.convs[0].conv(padded)
        features=[]
        for stage in encoder.stages[:5]:
          block=stage.blocks[0]
          out=block.conv1.conv(out)
          features.append(out)
        teacher_feats.append(features)
      for level,(student_phase,teacher_phase) in enumerate(zip(model.phases,teacher.feature_phases)):
        for axis in range(3):
          student_coord=ramps[0,axis:axis+1]
          # Replicate box weights across channels so all channels preserve
          # the same ramp and cross-correlation remains a normalized average.
          sf=student_feats[axis][level]
          tf=teacher_feats[axis][level]
          # Average feature channels; interior points equal native coordinates.
          sf=sf.mean(1,keepdim=True); tf=tf.mean(1,keepdim=True)
          aligned=align_teacher_to_student4(tf,shape,teacher_phase,student_phase)
          step,offset=student_phase
          coords=[offset[i]+torch.arange(aligned.shape[i+2])*step[i] for i in range(3)]
          expected=coords[axis].view(1,1,-1,1,1) if axis==0 else (coords[axis].view(1,1,1,-1,1) if axis==1 else coords[axis].view(1,1,1,1,-1))
          valid=torch.ones_like(aligned,dtype=torch.bool)
          # Ignore padding and kernel boundary support where replication alters
          # the ramp response; compare fully supported centers only.
          receptive=[1,1,1]
          cumulative=[1,1,1]
          for conv in (model.stem[0].conv,*[stage[0].conv for stage in (model.stage1,model.stage2)],model.stage3.down.conv,model.stage4.down.conv)[:level+1]:
            for dim in range(3):
              receptive[dim]+=(conv.kernel_size[dim]-1)*conv.dilation[dim]*cumulative[dim]
              cumulative[dim]*=conv.stride[dim]
          teacher_radius=[1,1,1]
          teacher_cumulative=[1,1,1]
          for conv in (encoder.stem.convs[0].conv,*[stage.blocks[0].conv1.conv for stage in encoder.stages[:5]][:level+1]):
            for dim in range(3):
              teacher_radius[dim]+=(conv.kernel_size[dim]-1)*conv.dilation[dim]*teacher_cumulative[dim]
              teacher_cumulative[dim]*=conv.stride[dim]
          for dim,coord in enumerate(coords):
            view=[1,1,1,1,1]; view[dim+2]=-1
            margin=max((receptive[dim]+1)//2,(teacher_radius[dim]+1)//2)
            valid &= ((coord>=margin)&(coord<shape[dim]-margin)).view(view)
          assert valid.any()
          error=torch.max(torch.abs(aligned[valid]-expected.expand_as(aligned)[valid])).item()
          assert error <= 0.5, (level,axis,error,student_phase,teacher_phase)
