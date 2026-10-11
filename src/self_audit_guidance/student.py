"""Student learns frozen features even when all semantic labels abstain."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .io import verify,load_bank_patient,freeze,write_json,digest
from .teacher import validate_device


class Block(nn.Sequential):
    def __init__(self,ci,co,stride=1):
        super().__init__(nn.Conv2d(ci,co,3,stride,1,bias=False),nn.GroupNorm(4,co),nn.GELU(),
                         nn.Conv2d(co,co,3,1,1,bias=False),nn.GroupNorm(4,co),nn.GELU())


class GuidanceStudent(nn.Module):
    def __init__(self,feature_dim=64,width=32):
        super().__init__()
        if type(width) is not int or width<8 or width%4:raise ValueError('student width must be a multiple of 4 >=8')
        self.enc0=Block(3,width);self.enc1=Block(width,2*width,2);self.enc2=Block(2*width,4*width,2)
        self.project=nn.Conv2d(4*width,feature_dim,1)
        self.decode=Block(5*width,width);self.head=nn.Conv2d(width,4,1)
        self.feature_dim=feature_dim;self.width=width

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('student expects [B,3,H,W]')
        low=self.enc0(x);high=self.enc2(self.enc1(low))
        decoder=self.decode(torch.cat([low,F.interpolate(high,low.shape[-2:],mode='bilinear',align_corners=False)],1))
        return dict(features=self.project(high),logits=self.head(decoder))

    def semantic_parameters(self):
        return [*self.decode.parameters(),*self.head.parameters()]


def guidance_loss(out,teacher,q,reliability,semantic_enabled=True):
    """Targets detached. No CQA readiness gate on feature learning."""
    teacher=teacher.detach().float();q=q.detach().float();w=reliability.detach().float()
    if (teacher.ndim!=4 or q.ndim!=4 or q.shape[1]!=4 or w.shape!=(q.shape[0],*q.shape[-2:])
        or teacher.shape[0]!=q.shape[0] or teacher.shape[-2:]!=q.shape[-2:]
        or not all(torch.isfinite(a).all() for a in (teacher,q,w))
        or (q<0).any() or (w<0).any() or (w>1).any()
        or not torch.allclose(q.sum(1),torch.ones_like(w),atol=2e-4,rtol=0)):
        raise ValueError('invalid frozen guidance tensors')
    feature=F.interpolate(out['features'].float(),teacher.shape[-2:],mode='bilinear',align_corners=False)
    if feature.shape!=teacher.shape:raise ValueError('teacher/student feature mismatch')
    # Exclude exactly-zero features from cosine but never use semantic confidence as the feature gate.
    support=teacher.square().sum(1)>1e-10
    if not support.any():raise ValueError('NO_FEATURE_SIGNAL: teacher emitted only zero vectors')
    feature_loss=(1-F.cosine_similarity(feature,teacher,dim=1))[support].mean()
    logits=F.interpolate(out['logits'].float(),q.shape[-2:],mode='bilinear',align_corners=False)
    # Local BG-only batches may not push an unsupported decoder toward all BG.
    active=bool(semantic_enabled and ((q.argmax(1)!=0)&(w>0)).any())
    semantic=logits.sum()*0
    if active:
        semantic=(-(q*logits.log_softmax(1)).sum(1)*w).sum()/w.sum().clamp_min(1e-8)
    return feature_loss,semantic,active


def contexts(image,indices):
    return np.stack([np.stack([image[t,max(0,z-1)],image[t,z],image[t,min(image.shape[1]-1,z+1)]]) for t,z in indices])


def train_student(bank,out,epochs=10,batch_size=8,device='cuda',width=32,lr=3e-4,seed=42,threads=4):
    if type(epochs) is not int or type(batch_size) is not int or min(epochs,batch_size,threads)<1 or not math.isfinite(lr) or lr<=0:
        raise ValueError('invalid training budget')
    d=validate_device(device);torch.set_num_threads(threads);torch.manual_seed(seed)
    bank=Path(bank);m=verify(bank,'guidance_bank')
    if m.get('split')!='train' or not m.get('patients'):raise ValueError('student accepts a TRAIN-only bank')
    if any(pid not in m['locked_splits']['train'] for pid in m['patients']):raise ValueError('non-TRAIN bank patient')
    bank_manifest_sha=digest(bank/'FROZEN.json')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=m['config'];model=GuidanceStudent(cfg['feature_dim'],width).to(d)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    class_patients=[sum(m['patient_reports'][pid]['class_pixels'][c]>0 for pid in m['patients']) for c in range(4)]
    # This is a safety permission, not class accuracy certification. Feature learning always proceeds.
    allowed_classes=[n>=2 for n in class_patients]
    semantic_enabled=any(allowed_classes[1:])
    history=[];successful_semantic=0
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        order=rng.permutation(m['patients']);epoch_steps=0;sum_feat=0.;sum_sem=0.
        for pid in order:
            data=load_bank_patient(bank,str(pid));image=data['image']
            ids=np.array(list(np.ndindex(image.shape[:2])));rng.shuffle(ids)
            for start in range(0,len(ids),batch_size):
                current=ids[start:start+batch_size]
                x=torch.from_numpy(contexts(image,current)).to(d)
                t,z=current[:,0],current[:,1]
                feat=torch.from_numpy(data['features'][t,z].astype(np.float32)).permute(0,3,1,2).to(d)
                q=torch.from_numpy(data['q'][t,z]).permute(0,3,1,2).to(d)
                rel=torch.from_numpy(data['reliability'][t,z]).to(d)
                rel=rel*torch.tensor(allowed_classes,device=d)[q.argmax(1)]
                result=model(x);fl,sl,active=guidance_loss(result,feat,q,rel,semantic_enabled)
                loss=fl+sl
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite student loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                if not active:
                    for p in model.semantic_parameters():p.grad=None
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                opt.step();successful_semantic+=int(active);epoch_steps+=1
                sum_feat+=float(fl.detach());sum_sem+=float(sl.detach())
        row=dict(epoch=epoch+1,steps=epoch_steps,feature_loss=sum_feat/max(epoch_steps,1),semantic_loss=sum_sem/max(epoch_steps,1))
        history.append(row);print('[STUDENT]',row,flush=True)
    if digest(bank/'FROZEN.json')!=bank_manifest_sha:raise ValueError('bank manifest changed during training')
    verify(bank,'guidance_bank')
    status=('SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED' if all(allowed_classes) else 'PARTIAL_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED') if successful_semantic else 'REPRESENTATION_ONLY'
    payload=dict(model=model.state_dict(),feature_dim=cfg['feature_dim'],width=width,
        status=status,semantic_updates=successful_semantic,bank_manifest_sha256=bank_manifest_sha,
        teacher=m['teacher'],config=cfg,train_patients=m['patients'],locked_splits=m['locked_splits'],
        class_patients=class_patients,allowed_classes=allowed_classes,selection='fixed_last_epoch_no_GT',seed=seed)
    torch.save(payload,out/'last.pt');write_json(out/'training.json',history)
    write_json(out/'REPORT.json',dict(status=status,semantic_updates=successful_semantic,class_patients=class_patients,
                                    batch_size=batch_size,epochs=epochs))
    freeze(out,dict(role='student',status=status,bank_manifest_sha256=bank_manifest_sha,
        teacher=m['teacher'],training=dict(epochs=epochs,batch_size=batch_size,lr=lr,width=width,seed=seed),
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    return payload


def load_student(folder,device):
    m=verify(folder,'student');d=validate_device(device)
    ck=torch.load(Path(folder)/'last.pt',map_location='cpu',weights_only=True)
    if ck['status']!=m['status']:raise ValueError('student status mismatch')
    model=GuidanceStudent(ck['feature_dim'],ck['width']);model.load_state_dict(ck['model'],strict=True)
    return model.eval().to(d),ck


@torch.inference_mode()
def predict_cine(model,ck,cine,device='cpu',batch_size=8):
    if type(batch_size) is not int or batch_size<1:raise ValueError('batch_size must be positive')
    t,z,h,w=cine.shape;result=np.full((t,z,h,w),255,np.uint8)
    if ck['status']=='REPRESENTATION_ONLY':return result
    cfg=ck['config'];ids=list(np.ndindex((t,z)))
    for start in range(0,len(ids),batch_size):
        current=ids[start:start+batch_size]
        x=torch.from_numpy(contexts(cine,current)).to(device)
        p=model(x)['logits'].float().softmax(1);top=p.topk(2,1).values
        valid=(top[:,0]>=cfg['min_probability'])&(top[:,0]-top[:,1]>=cfg['min_margin'])
        valid &= torch.tensor(ck['allowed_classes'],device=p.device)[p.argmax(1)]
        labels=torch.where(valid,p.argmax(1),255).cpu().numpy().astype(np.uint8)
        for pos,(ti,zi) in enumerate(current):result[ti,zi]=labels[pos]
    return result
