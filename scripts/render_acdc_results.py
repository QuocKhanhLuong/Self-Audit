#!/usr/bin/env python3
"""Render measured frozen evaluation artifacts; no fitting or training access."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def render(a):
    if a.out.exists(): raise ValueError('new figure output required')
    a.out.mkdir(parents=True)
    metrics=json.loads((a.evaluation/'METRICS.json').read_text())
    manifest=json.loads(a.data.read_text())
    patients={r['id']:r for r in manifest['patients']}
    resolve=lambda p: Path(p) if Path(p).is_absolute() else a.data.parent/p
    cmap=ListedColormap(['black','#377eb8','#e41a1c','#4daf4a'])
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),dpi=220,bbox_inches='tight')
        fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    cm=np.asarray(metrics['pooled_confusion'])
    fig,ax=plt.subplots(figsize=(6,4));im=ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(5),xticklabels=['BG','RV','MYO','LV','UNKNOWN'],yticks=range(4),yticklabels=['BG','RV','MYO','LV'],xlabel='Prediction',ylabel='Reference',title='VAL voxel confusion — frozen student')
    for i in range(4):
        for j in range(5):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=ax);save(fig,'confusion')
    fig,ax=plt.subplots(figsize=(5,4));names=['RV','MYO','LV']
    values=[[v[c] for v in metrics['patients'].values() if v[c] is not None] for c in names]
    ax.boxplot(values,tick_labels=names,showmeans=True)
    ax.set(ylabel='Patient macro Dice',ylim=(0,1),title='Exposed VAL; no confirmation claim');save(fig,'patient_dice')
    def fg_dice(row):
        v=[x['dice'] for x in row['raw'][1:] if x['dice'] is not None]
        return float(np.mean(v)) if v else float('inf')
    chosen=sorted(metrics['frames'],key=lambda r:(fg_dice(r),r['patient'],r['frame']))[:3]
    for rank,row in enumerate(chosen):
        r=patients[row['patient']];ti=row['frame'];mr=r['masks'][str(ti)]
        cine=np.load(resolve(r['image']),mmap_mode='r',allow_pickle=False)
        y=np.load(resolve(mr['path']),allow_pickle=False)
        p=np.load(a.evaluation/'predictions'/(r['id']+'.npy'),mmap_mode='r',allow_pickle=False)
        areas=((y>0)&(y<4)).sum((1,2));z=int(areas.argmax())
        image=cine[ti,z];gt=y[z];prob=np.asarray(p[ti,z]);pred=prob.argmax(0)
        confidence=prob.max(0);entropy=-(prob*np.log(np.clip(prob,1e-8,1))).sum(0)/np.log(4)
        fig,axs=plt.subplots(2,4,figsize=(14,7))
        axs[0,0].imshow(image,cmap='gray');axs[0,0].set_title('Image')
        for ax,seg,title in [(axs[0,1],gt,'GT'),(axs[0,2],pred,'Prediction')]:
            ax.imshow(image,cmap='gray');ax.imshow(np.ma.masked_where((seg==0)|(seg==255),seg),cmap=cmap,vmin=0,vmax=3,alpha=.6);ax.set_title(title)
        axs[0,3].imshow(confidence,vmin=0,vmax=1,cmap='viridis');axs[0,3].set_title('Confidence')
        axs[1,0].imshow(entropy,vmin=0,vmax=1,cmap='magma');axs[1,0].set_title('Normalized entropy')
        for k,ax in enumerate(axs[1,1:],1):ax.imshow(prob[k],vmin=0,vmax=1,cmap='viridis');ax.set_title(names[k-1]+' probability')
        for ax in axs.flat:ax.axis('off')
        fig.suptitle(f"Failure rank {rank+1}: {r['id']}, frame {ti+1}, slice {z}; frame foreground Dice={fg_dice(row):.3f}")
        save(fig,f'failure_{rank+1}_{r["id"]}')
        fig,axs=plt.subplots(1,4,figsize=(12,3))
        for k,ax in enumerate(axs):ax.imshow(prob[k],vmin=0,vmax=1,cmap='viridis');ax.set_title(['BG','RV','MYO','LV'][k]);ax.axis('off')
        save(fig,f'probabilities_{rank+1}_{r["id"]}')
    if a.train_run:
        rows=[json.loads(s) for s in (a.train_run/'steps.jsonl').read_text().splitlines()]
        fig,axs=plt.subplots(1,3,figsize=(12,3));steps=[r['step'] for r in rows]
        axs[0].plot(steps,[r['supervised_loss'] for r in rows]);axs[0].set(ylabel='TRAIN supervised loss',xlabel='Step')
        scales=[r.get('control',{}).get('scale',0) for r in rows]
        axs[1].plot(steps,np.cumsum(np.array(scales)>0));axs[1].set(ylabel='Cumulative accepted pseudo updates',xlabel='Step')
        axs[2].plot(steps,[r.get('cuda_peak_allocated',0)/1024**3 for r in rows]);axs[2].set(ylabel='Peak allocated GiB',xlabel='Step')
        fig.tight_layout();save(fig,'training_cost_and_acceptance')
    (a.out/'PROVENANCE.json').write_text(json.dumps({'evaluation_metrics':str(a.evaluation/'METRICS.json'),'evaluation_recipe':json.loads((a.evaluation/'RECIPE.json').read_text()),'selection':'lowest frame mean foreground Dice; max GT-area slice within frame; visualization only','validation_curve':'not available: final-checkpoint evaluation only'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--train-run',type=Path)
    render(p.parse_args())
