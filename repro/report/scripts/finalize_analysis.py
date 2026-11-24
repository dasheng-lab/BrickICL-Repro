"""Final-result comparisons and publication figures; called by reproduce.py."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
def build(root):
 from reproduce import PRIMARY,IDS,case_values
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
 def read(name):return json.loads((root/'data'/name).read_text(encoding='utf-8'))
 new=read('completion_100.json');phys=read('physical.json');stats=json.loads((root/'generated/statistics.json').read_text(encoding='utf-8'))['models'];rng=np.random.default_rng(20261123);ix=rng.integers(0,100,size=(10000,100))
 arrays={a:case_values(r) for a,r in new.items()};comparisons={}
 for a,b in [('post_pt','frozen_pt'),('regularized_g_wd010_do015','regularized_g'),('grpo_pt','masked')]:
  diff=arrays[a][:,2]-arrays[b][:,2];comparisons[a+'__'+b]={'stretched_f1_difference':float(diff.mean()),'ci95':np.percentile(diff[ix].mean(1),[2.5,97.5]).tolist()}
 for a,b in [('regularized_g','direct_g'),('grpo_pt','masked'),('regularized_g_wd010_do015','regularized_g')]:
  x=np.array([r['collision_free_length'] for r in phys[a]['no_eos']]);y=np.array([r['collision_free_length'] for r in phys[b]['no_eos']]);r=np.random.default_rng(20261123);est=x[r.integers(len(x),size=(10000,len(x)))].mean(1)-y[r.integers(len(y),size=(10000,len(y)))].mean(1)
  comparisons.setdefault(a+'__'+b,{})['no_eos_collision_free_difference']={'mean':float(x.mean()-y.mean()),'ci95':np.percentile(est,[2.5,97.5]).tolist(),'bootstrap':'independent output resamples, 64 per model'}
 truncation={}
 for arm in PRIMARY:
  ss=[s for r in new[arm]['cases'] for s in r['samples']];perfect=[s for s in ss if s['pose_f1']==1.0]
  truncation[arm]={'reported_f1_mean':float(np.mean([s['pose_f1'] for s in ss])),'untruncated_f1_mean':float(np.mean([s['pose_f1_untruncated'] for s in ss])),'perfect_truncated_draws':len(perfect),'perfect_with_extra_parts':sum(s['n_generated']>s['target_parts'] for s in perfect)}
 audit={'comparisons':comparisons,'truncation':truncation,'not_evaluated':'Full-G-regularized-start GRPO was not located/verified; B-RL is v2.1/400 from nzd enhanced PT.'}
 (root/'generated/final_comparisons.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 fig,axes=plt.subplots(1,2,figsize=(7.1,4.15),sharey=True,gridspec_kw={'width_ratios':[1.25,1]});ys=np.arange(len(PRIMARY))[::-1]
 for y,arm in zip(ys,PRIMARY):
  s=stats[arm]
  for k,off,col,label in [(1,.13,'#9ba9b9','Normal'),(2,-.13,'#177f7b','Stretched')]:
   v=s['completion_mean'][k];lo,hi=s['completion_ci95'][k];axes[0].errorbar(v,y+off,xerr=[[v-lo],[hi-v]],fmt='o',ms=3.2,lw=.8,color=col,capsize=1.8,label=label if arm=='frozen_pt' else None)
  v=s['completion_mean'][0];lo,hi=s['completion_ci95'][0];axes[1].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',ms=3.5,lw=.9,color='#c5762d',capsize=2)
 axes[0].set_yticks(ys,[IDS[a] for a in PRIMARY]);axes[0].set(xlim=(0,.65),xlabel='Pose F1 (target-length truncated)');axes[1].set(xlim=(0,3.7),xlabel='Sampled log-odds shift');axes[0].legend(frameon=False,ncol=2,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.01))
 for ax in axes:
  ax.grid(axis='x',alpha=.16);ax.set_ylim(-.65,len(PRIMARY)-.35);ax.tick_params(labelsize=9);ax.set_axisbelow(True)
  for y in [6.5,10.5]:ax.axhline(y,color='#d7dde4',lw=.7)
 fig.tight_layout(pad=.7)
 for ext in ['pdf','png']:fig.savefig(root/'figures'/('completion_overview.'+ext),dpi=200,bbox_inches='tight',pad_inches=.03)
 plt.close(fig)
 # Two separate stopping policies with group-independent intervals; neither is a latent-ability estimate.
 chosen=['frozen_pt','post_pt','direct_m','direct_g','regularized_g','regularized_g_wd010_do015','masked','grpo_pt'];ys=np.arange(len(chosen))[::-1];fig,ax=plt.subplots(figsize=(7.1,3.3))
 for y,arm in zip(ys,chosen):
  for mode,off,col,label in [('natural',.13,'#97a7bb','EOS allowed'),('no_eos',-.13,'#167d79','EOS suppressed')]:
   s=stats[arm][mode+'_collision_free_length'];v=s['mean'];lo,hi=s['ci95'];ax.errorbar(v,y+off,xerr=[[v-lo],[hi-v]],fmt='o',color=col,ms=4,capsize=2,lw=1,label=label if arm==chosen[0] else None)
 ax.set_yticks(ys,[IDS[a] for a in chosen]);ax.set(xlabel='Mean collision-free prefix length (bricks)',xlim=(-.5,30),ylim=(-.6,len(chosen)-.4));ax.legend(frameon=False,fontsize=9,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01));ax.grid(axis='x',alpha=.17);fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(root/'figures'/('stopping_overview.'+ext),dpi=200,bbox_inches='tight',pad_inches=.03)
 plt.close(fig)
 lines=[r'\begin{table}[htbp]\centering\small',r'\caption{New100 uncertainty by source. Each cell is a mean with a 95\% percentile interval from 10,000 source-level bootstrap resamples. Conditions and four draws remain grouped. These are fixed-checkpoint intervals, not training-seed uncertainty.}\label{tab:ci}',r'\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lrrr@{}}\toprule',r'ID & $\Delta\ell$ & Normal F1 & Stretched F1 \\ \midrule']
 for arm in PRIMARY:
  s=stats[arm];cells=[f'{m:.3f} [{c[0]:.3f}, {c[1]:.3f}]' for m,c in zip(s['completion_mean'],s['completion_ci95'])];lines.append(IDS[arm]+' & '+' & '.join(cells)+r' \\')
 lines+=[r'\bottomrule\end{tabular}\end{table}'];(root/'tables/confidence_intervals.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(comparisons,indent=2),flush=True)
if __name__=='__main__':build(Path(__file__).resolve().parents[1])

