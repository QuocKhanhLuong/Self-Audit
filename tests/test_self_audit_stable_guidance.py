"""Manufactured image/provider contract tests; NOT a CineMA/ACDC accuracy run."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F
from self_audit_guidance.core import (GuidanceConfig,anchored_solve,build_graph,
    temporal_matches,unit_features,anatomical_anchors,make_bank)
from self_audit_guidance.teacher import extract_tiled,starts,validate_teacher_config,validate_device
from self_audit_guidance.student import GuidanceStudent,guidance_loss,train_student,load_student,predict_cine
from self_audit_guidance.io import write_json,freeze,verify,digest,read_splits,discover,safe_image,load_bank_patient
from self_audit_guidance.pipeline import prepare,choose_patients


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('seed',range(8))
def test_solver_matches_exact(seed):
    r=np.random.default_rng(seed);p=r.uniform(size=(12,12));p/=p.sum(1,keepdims=True)
    r0=r.dirichlet(np.ones(4),size=12);alpha=.1+.1*seed
    q,receipt=anchored_solve(p,r0,alpha,tolerance=1e-9)
    expected=np.linalg.solve(np.eye(12)-(1-alpha)*p,alpha*r0)
    assert np.abs(q-expected).max()<=receipt['error_bound']+1e-14
    assert receipt['error_bound']<=1e-9
    assert np.allclose(q.sum(1),1) and (q>=0).all()
    q2,_=anchored_solve(p,r0,alpha,tolerance=1e-9,initial=r.random((12,4)))
    assert np.allclose(q,q2,atol=2e-9)


def test_negative_controls_stable_not_useful():
    p=np.array([[.5,.5],[.5,.5]])
    for r0 in (np.ones((2,4))*.25,np.tile([1.,0,0,0],(2,1))):
        q,_=anchored_solve(p,r0)
        assert np.allclose(q,r0)


def test_nonconvergence_does_not_claim_success():
    p=np.array([[0.,1.],[1.,0.]])
    with pytest.raises(RuntimeError,match='NOT_CONVERGED'):
        anchored_solve(p,np.eye(2),.1,1e-12,1)


@pytest.mark.parametrize('p,r0',[(np.eye(2)*2,np.eye(2)),(np.array([[2.,-1.],[0,1]]),np.eye(2)),
    (np.eye(2),np.array([[2.,-1.],[0,1]])),(np.eye(3),np.eye(2)),(np.eye(2),np.full((2,2),np.nan))])
def test_solver_bad_inputs(p,r0):
    with pytest.raises(ValueError):anchored_solve(p,r0)


@pytest.mark.parametrize('kw',[{'alpha':0},{'grid':1},{'temporal':1},{'clusters':[]},
    {'match_radius':0},{'spatial_scale':float('nan')},{'feature_dim':True},{'min_probability':2}])
def test_invalid_config(kw):
    with pytest.raises(ValueError):GuidanceConfig(**kw)


def test_grid_selfloops_and_no_implicit_cross_slice():
    image=np.zeros((2,2,8,8),np.float32);feat=np.ones((*image.shape,4),np.float32)
    p,m=build_graph(image,feat,GuidanceConfig(grid=8))
    assert np.allclose(p.sum(1),1);assert (p.diagonal()>0).all()
    assert m['temporal_edges']==0  # tied feature matches do not assert static correspondence
    assert p[:64,64:].nnz==0


def test_graph_node_budget_is_explicit():
    image=np.zeros((2,2,8,8));feat=np.ones((*image.shape,4))
    with pytest.raises(ValueError,match='nodes'):
        build_graph(image,feat,GuidanceConfig(grid=8,max_nodes=16))


def test_temporal_matches_actual_displacement_not_fixed_grid():
    rng=np.random.default_rng(4);a=unit_features(rng.normal(size=(8,8,16)))
    b=np.zeros_like(a);b[:,1:]=a[:,:-1];b[:,0]=unit_features(rng.normal(size=(8,16)))
    im=np.zeros((8,8));cfg=GuidanceConfig(grid=8,match_radius=1,match_cosine=.95,match_margin=.03)
    src,dst,score=temporal_matches(a,b,im,im,cfg)
    assert len(src)>=48 and np.all(dst==src+1)
    assert np.all(score>.95)


def test_photometric_veto():
    a=unit_features(np.random.default_rng(5).normal(size=(8,8,16)))
    src,_,_=temporal_matches(a,a,np.zeros((8,8)),np.ones((8,8))*2,GuidanceConfig(grid=8))
    assert len(src)==0


def annulus():
    y,x=np.mgrid[:32,:32];r=np.hypot(x-16,y-16)
    image=np.full((2,1,32,32),-.5,np.float32)
    image[:,:,r<10]=.2;image[:,:,r<6]=1.5
    codes=np.zeros((32,32),int);codes[r<10]=1;codes[r<6]=2
    feat=np.tile(np.eye(4,dtype=np.float32)[codes],(2,1,1,1,1))
    return image,feat


def test_readout_supplied_features_not_neural_head():
    image,feat=annulus();cfg=GuidanceConfig(grid=32,feature_dim=4,clusters=(3,4),min_region_pixels=3)
    a,s,d=anatomical_anchors(image,feat,'+x',cfg)
    assert d['wall_cavity_hypotheses']>0
    assert np.allclose(a.sum(-1),1) and np.any(a[...,2]>.25) and np.any(a[...,3]>.25)
    # A nonexistent RV is never invented from an orientation-only prior.
    assert not (a[...,1]>.25).any()


def test_absent_anatomy_still_produces_feature_bank():
    image=np.zeros((2,1,16,16),np.float32);f=np.ones((*image.shape,4),np.float32)
    bank,report=make_bank(image,f,'',GuidanceConfig(grid=16,feature_dim=4))
    assert np.all(bank['labels']==255)
    assert np.all(bank['reliability']==0)
    assert np.allclose(bank['q'],.25)
    assert report['solver']['converged'] and report['status']=='NO_FOREGROUND_SUPPORT'


def test_anatomy_inputs_not_mutated():
    image,f=annulus();a,b=image.copy(),f.copy()
    make_bank(image,f,'+x',GuidanceConfig(grid=32,feature_dim=4,temporal=False))
    assert np.array_equal(a,image) and np.array_equal(b,f)


class TokenMock(nn.Module):
    def __init__(self,tile=(16,16,4),token_grid=(4,4,4)):
        super().__init__();self.token_grid=token_grid;self.tile=tile
        self.register_parameter('scale',nn.Parameter(torch.ones(()),requires_grad=False));self.eval()
    def feature_forward(self,d):
        a=d['sax']
        out=F.adaptive_avg_pool3d(a,self.token_grid).permute(0,2,3,4,1)
        feat=torch.cat([out,out**2,torch.ones_like(out),torch.sin(out)],-1)
        return {'sax':feat.reshape(len(a),-1,4)}


def test_full_coverage_tiling_batch_invariance():
    cine=np.random.default_rng(8).normal(size=(2,6,25,29)).astype(np.float32)
    m=TokenMock();kw=dict(tile=m.tile,token_grid=m.token_grid,device='cpu',grid=12,feature_dim=4)
    a=extract_tiled(m,cine,tile_batch=1,**kw);b=extract_tiled(m,cine,tile_batch=3,**kw)
    assert a.shape==(2,6,12,12,4) and np.isfinite(a).all()
    assert np.allclose(a,b,atol=1e-3)
    assert np.all(np.linalg.norm(a.astype(float),axis=-1)>.99)


def test_padding_no_uncovered_small_image():
    m=TokenMock();cine=np.ones((2,2,7,5),np.float32)
    f=extract_tiled(m,cine,m.tile,m.token_grid,'cpu',8,4)
    assert f.shape==(2,2,8,8,4)


def test_trainable_teacher_refused():
    m=TokenMock();m.train()
    with pytest.raises(ValueError,match='frozen'):
        extract_tiled(m,np.zeros((2,1,8,8),np.float32),m.tile,m.token_grid,'cpu',8,4)


def test_starts_cover_every_native_position():
    for n in [1,16,17,24,29,100]:
        support=np.zeros(max(n,16),bool)
        for s in starts(n,16):support[s:s+16]=1
        assert support[:n].all()


def test_features_learn_with_zero_semantic_support():
    model=GuidanceStudent(4,8);x=torch.randn(2,3,17,23)
    feat=torch.randn(2,4,8,8,requires_grad=True);q=torch.full((2,4,8,8),.25,requires_grad=True)
    rel=torch.zeros(2,8,8,requires_grad=True)
    fl,sl,active=guidance_loss(model(x),feat,q,rel,False)
    (fl+sl).backward()
    assert fl.item()>0 and sl.item()==0 and not active
    assert model.enc0[0].weight.grad.abs().sum()>0
    assert feat.grad is None and q.grad is None and rel.grad is None


def test_zero_features_do_not_silently_pass():
    m=GuidanceStudent(4,8)
    with pytest.raises(ValueError,match='NO_FEATURE_SIGNAL'):
        guidance_loss(m(torch.randn(1,3,8,8)),torch.zeros(1,4,8,8),torch.full((1,4,8,8),.25),torch.zeros(1,8,8))


def test_soft_semantic_loss_and_gradients():
    m=GuidanceStudent(4,8);q=torch.zeros(1,4,8,8);q[:,3]=1
    fl,sl,active=guidance_loss(m(torch.randn(1,3,16,16)),torch.randn(1,4,8,8),q,torch.ones(1,8,8))
    (fl+sl).backward();assert active and sl>0 and m.head.weight.grad.abs().sum()>0


def test_no_semantic_export_from_representation_checkpoint():
    m=GuidanceStudent(4,8).eval()
    labels=predict_cine(m,{'status':'REPRESENTATION_ONLY'},np.zeros((2,2,16,16),np.float32))
    assert np.all(labels==255)


@pytest.mark.parametrize('path',['/tmp/masks/patient001.nii.gz','/tmp/x_gt.nii','/tmp/reference/a.nii','/tmp/labels/x.nii'])
def test_reference_paths_blocked_before_read(path):
    with pytest.raises(ValueError,match='reference'):safe_image(path)


def test_patient_leakage_and_selection(tmp_path):
    p=tmp_path/'split.json';write_json(p,{'train':['patient001'],'test':['patient001']})
    with pytest.raises(ValueError,match='leakage'):read_splits(p)
    write_json(p,{'train':['patient001','patient002','patient003'],'val':['patient004']})
    splits=read_splits(p);assert choose_patients(splits,2,42)==choose_patients(splits,2,42)
    assert len(choose_patients(splits,2,42))==2


def test_discovery_does_not_open_reference_or_info(tmp_path):
    d=tmp_path/'patient001';d.mkdir();(d/'patient001_4d.nii').write_text('image')
    (d/'Info.cfg').write_text('garbage not read');(d/'patient001_gt.nii').write_text('not read')
    paths=discover(tmp_path,'acdc',['patient001'])
    assert paths['patient001'].name=='patient001_4d.nii'
    (d/'another4d.nii').write_text('duplicate')
    with pytest.raises(ValueError,match='expected one'):discover(tmp_path,'acdc',['patient001'])


def test_artifact_detects_modified_and_added_file(tmp_path):
    (tmp_path/'a').write_text('one');freeze(tmp_path,dict(role='test'))
    assert verify(tmp_path,'test')['role']=='test'
    (tmp_path/'a').write_text('two')
    with pytest.raises(ValueError,match='hash'):verify(tmp_path,'test')
    (tmp_path/'a').write_text('one');(tmp_path/'extra').write_text('bad')
    with pytest.raises(ValueError,match='inventory'):verify(tmp_path,'test')


class FrozenSyntheticProvider:
    provenance={'provider':'SYNTHETIC_CONTRACT_ONLY','frozen':True,'checkpoint_sha256':'not-a-real-model'}
    def extract(self,cine,grid,feature_dim,tile_batch):
        # Explicit controlled features, no fallback in production CLI.
        small=F.interpolate(torch.from_numpy(cine.copy()).flatten(0,1)[:,None],(grid,grid),mode='bilinear',align_corners=False)
        small=small.reshape(*cine.shape[:2],grid,grid,1).numpy()
        base=np.concatenate([small,np.ones_like(small),np.sin(small),small**2],-1)
        return np.resize(base,(*base.shape[:-1],feature_dim)).astype(np.float16)


def fake_reader(path):
    rng=np.random.default_rng(123);cine=rng.normal(size=(2,2,16,20)).astype(np.float32)
    return cine,dict(image_sha256=digest(path),patient_left_axis='',raw_image=str(path),
        affine=np.eye(4).tolist(),shape_xyzt=[20,16,2,2],preprocessing='SYNTHETIC_CONTRACT_ONLY')


def test_prepare_train_reload_export_unknown_roundtrip(tmp_path):
    root=tmp_path/'raw';root.mkdir()
    for pid in ['patient001','patient002']:
        d=root/pid;d.mkdir();(d/(pid+'_4d.nii')).write_text('manufactured input marker')
    split=tmp_path/'split.json';write_json(split,dict(train=['patient001','patient002'],val=['patient003']))
    config=tmp_path/'cfg.json';write_json(config,GuidanceConfig(grid=8,feature_dim=4,clusters=(2,),temporal=False).dictionary())
    bank=tmp_path/'bank'
    report=prepare(root,'acdc',split,bank,config,patients=2,device='cpu',teacher=FrozenSyntheticProvider(),reader=fake_reader)
    assert report['status']=='INSUFFICIENT_SEMANTIC_SUPPORT_FEATURE_BANK_READY'
    frozen_before=digest(bank/'FROZEN.json');m=verify(bank,'guidance_bank')
    assert m['teacher']['provider']=='SYNTHETIC_CONTRACT_ONLY'
    data=load_bank_patient(bank,'patient001');assert data['features'].shape==(2,2,8,8,4)
    out=tmp_path/'student';ck=train_student(bank,out,epochs=2,batch_size=3,device='cpu',width=8,threads=1)
    assert ck['status']=='REPRESENTATION_ONLY' and ck['semantic_updates']==0
    assert digest(bank/'FROZEN.json')==frozen_before
    history=json.loads((out/'training.json').read_text());assert len(history)==2
    assert all(r['feature_loss']>0 for r in history) and all(r['semantic_loss']==0 for r in history)
    model,loaded=load_student(out,'cpu')
    labels=predict_cine(model,loaded,data['image'],'cpu',2);assert (labels==255).all()
    with pytest.raises(FileExistsError):train_student(bank,out,epochs=1,device='cpu',width=8)


def test_teacher_config_refuses_segmentation_settings():
    with pytest.raises((KeyError,ValueError)):
        validate_teacher_config({'model':{'num_classes':4},'data':{}})
