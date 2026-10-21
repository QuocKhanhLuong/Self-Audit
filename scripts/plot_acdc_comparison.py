#!/usr/bin/env python3
"""Aggregate measured VAL/cost figures only; no patient images."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.results/'logs/COMPLETED_SUMMARY.json').read_text())['completed_methods']
if not rows:raise SystemExit('No scored runs')
names=[r['method'] for r in rows];x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(8,4))
for j,c in enumerate(['RV','MYO','LV']):ax.bar(x+(j-1)*.25,[100*r['macro_patient'][c]['dice'] for r in rows],width=.25,label=c)
ax.set(xticks=x,xticklabels=names,ylabel='Patient macro Dice (%)',title='Exposed VAL18; L8/seed42; one completed seed');ax.legend();fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(a.results/'figures'/('comparison_val_dice.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for r in rows:
 hours=r['elapsed_seconds']/3600;mean=100*np.mean([r['macro_patient'][c]['dice'] for c in ['RV','MYO','LV']]);ax.scatter(hours,mean);ax.annotate(r['method'],(hours,mean),xytext=(4,4),textcoords='offset points',fontsize=8)
ax.set(xlabel='Timed training/final-save wall hours (startup excluded)',ylabel='Mean foreground patient Dice (%)',title='Instrumented compute; exposed VAL18; one seed');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(a.results/'figures'/('accuracy_cost.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[r['pseudo_accepted'] for r in rows]);ax.set(ylabel='Applied pseudo transactions',title='TRAIN optimization counts; not a quality certificate');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(a.results/'figures'/('acceptance_comparison.'+ext),dpi=200,bbox_inches='tight')
