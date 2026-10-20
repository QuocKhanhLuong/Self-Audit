#!/usr/bin/env python3
"""Frozen image-only TEST inference. No mask discovery, loading or scoring."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from self_audit.data import TrainingData
from self_audit.engine import load_checkpoint
from self_audit.model import digest


def infer(a):
    if a.out.exists():raise ValueError('new output required')
    model,meta=load_checkpoint(a.checkpoint,a.checkpoint_sha256,'cuda')
    if meta['synthetic']:raise ValueError('patient TEST requires non-synthetic checkpoint')
    ids=[f'patient{i:03d}' for i in range(101,151)]
    if set(ids)&set(meta['labeled_ids']+meta['unlabeled_ids']):raise ValueError('TEST/TRAIN overlap')
    a.out.mkdir(parents=True);receipts=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors=ListedColormap(['black','#377eb8','#e41a1c','#4daf4a'])
    with torch.inference_mode():
        for pid in ids:
            paths=list((a.root/pid).glob(pid+'_4d.nii*'))
            if len(paths)!=1:raise ValueError('missing unique TEST cine')
            source=paths[0];ni=nib.load(source);raw=np.asarray(ni.dataobj)
            if raw.ndim!=4 or not np.isfinite(raw).all():raise ValueError('invalid TEST cine')
            lo,hi=np.percentile(raw,[.5,99.5])
            x=np.clip((raw-lo)/max(float(hi-lo),1e-6),0,1).transpose(3,2,1,0).astype(np.float32)
            pred=np.empty(x.shape,dtype=np.uint8);indices=[(t,z) for t in range(x.shape[0]) for z in range(x.shape[1])]
            for offset in range(0,len(indices),a.batch):
                chunk=indices[offset:offset+a.batch]
                inputs=torch.cat([TrainingData.context(x,t,z) for t,z in chunk]).cuda()
                labels=model(inputs).argmax(1).cpu().numpy().astype(np.uint8)
                for (t,z),y in zip(chunk,labels):pred[t,z]=y
            op=a.out/(pid+'_prediction.nii.gz');header=ni.header.copy();header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(pred.transpose(3,2,1,0),ni.affine,header=header),op)
            t,z=x.shape[0]//2,x.shape[1]//2
            fig,ax=plt.subplots(figsize=(4,4));ax.imshow(x[t,z],cmap='gray');ax.imshow(np.ma.masked_where(pred[t,z]==0,pred[t,z]),cmap=colors,vmin=0,vmax=3,alpha=.6);ax.axis('off');ax.set_title(pid+' TEST inference; no GT')
            fig.savefig(a.out/(pid+'_overlay.png'),dpi=150,bbox_inches='tight');plt.close(fig)
            receipts.append({'patient':pid,'cine_sha256':digest(source),'prediction':op.name,'prediction_sha256':digest(op),'shape_xyzt':list(ni.shape)})
            (a.out/'INFERENCE.json').write_text(json.dumps({'checkpoint_sha256':a.checkpoint_sha256,'scope':'TEST_IMAGE_ONLY; provenance of Drive testing copy not independently certified','test_metrics':'NOT_COMPUTED; no GT accessed','batch_size':a.batch,'normalization':'image_only_percentiles_0.5_99.5','patients':receipts,'complete':len(receipts)==50},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=16);a=p.parse_args()
    if a.batch<1:p.error('positive batch required')
    infer(a)
