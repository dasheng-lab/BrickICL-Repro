"""Regenerate every quantitative manuscript figure/table from bundled per-case observations.
No checkpoint, GPU, network or cluster account is needed.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
PRIMARY=['frozen_pt','post_pt','fullg_pt','plain','masked','direct_m','direct_g','nzd_v2_plain','nzd_v2_masked','regularized_g','regularized_g_wd010_do015','grpo_pt']
IDS=dict(zip(PRIMARY,['B0','B-M','B-G','B-DFS','B-LATE','Q-M','Q-G','Q-DFS','Q-LATE','Q-G-R1','Q-G-R2','B-RL']))
LABELS=dict(zip(PRIMARY,['BrickNet released','BrickNet + ours / M-only','BrickNet + ours / Full-G','BrickNet + paired DFS','BrickNet + substructure-last','Qwen + ours / M-only','Qwen + ours / Full-G','Qwen + paired DFS','Qwen + substructure-last','Qwen Full-G / WD .05, DO .10','Qwen Full-G / WD .10, DO .15','BrickNet substructure-last + RL']))
def read(name):return json.loads((ROOT/'data'/name).read_text(encoding='utf-8'))
def savefig(fig,name):
 fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.04);fig.savefig(ROOT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=.04);plt.close(fig)
def case_values(record):
 by={}
 for r in record['cases']:by.setdefault(r['source_key'],{})[r['condition']]=r['samples']
 out=[]
 for source in sorted(by):
  o,l=by[source]['original'],by[source]['stretched'];ko=sum(s['long_feature'] for s in o);kl=sum(s['long_feature'] for s in l)
  out.append([math.log((kl+.5)/(4-kl+.5))-math.log((ko+.5)/(4-ko+.5)),np.mean([s['pose_f1'] for s in o]),np.mean([s['pose_f1'] for s in l])])
 return np.array(out)
def main():
 for p in ['figures','tables','generated']:(ROOT/p).mkdir(exist_ok=True)
 new,old,phys,train=map(read,['completion_100.json','completion_18.json','physical.json','training.json']);rng=np.random.default_rng(20261123);ix=rng.integers(0,100,size=(10000,100));statistics={};arrays={}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
 for arm,r in new.items():
  a=case_values(r);assert a.shape==(100,3);point=a.mean(0);expected=r['metrics'];assert abs(point[0]-expected['delta'])<1e-10 and abs(point[2]-expected['f1']['stretched'])<1e-10
  boot=a[ix].mean(1);statistics[arm]={'completion_mean':point.tolist(),'completion_ci95':np.percentile(boot,[2.5,97.5],axis=0).T.tolist(),'completion_avg_f1':float(point[1:].mean()),'completion_avg_f1_ci95':np.percentile(boot[:,1:].mean(1),[2.5,97.5]).tolist()};arrays[arm]=a
  for mode in ['natural','no_eos']:
   for metric in ['valid_length','collision_free_length']:
    vals=np.array([s[metric] for s in phys[arm][mode] if s[metric] is not None],float);idx=np.random.default_rng(20261123).integers(len(vals),size=(10000,len(vals)));statistics[arm][mode+'_'+metric]={'mean':float(vals.mean()),'ci95':np.percentile(vals[idx].mean(1),[2.5,97.5]).tolist(),'n':len(vals)}
 effects=[('post_pt','fullg_pt','B: M-only - Full-G'),('direct_m','direct_g','Q: M-only - Full-G'),('masked','plain','B: last - DFS'),('nzd_v2_masked','nzd_v2_plain','Q: last - DFS'),('regularized_g','direct_g','Q: R1 - unregularized'),('regularized_g_wd010_do015','direct_g','Q: R2 - unregularized'),('grpo_pt','masked','B: RL - its parent')]
 effectdata=[]
 for a,b,label in effects:
  diff=arrays[a][:,2]-arrays[b][:,2];ci=np.percentile(diff[ix].mean(1),[2.5,97.5]);effectdata.append({'a':a,'b':b,'label':label,'stretched_F1_difference':float(diff.mean()),'ci95':ci.tolist()})
 (ROOT/'generated/statistics.json').write_text(json.dumps({'bootstrap_seed':20261123,'resamples':10000,'unit':'source assembly; both conditions and all four samples kept together','models':statistics,'paired_effects':effectdata},indent=2))
 # Main table: only method-comparable current recipes; retired data recipes remain in appendix.
 lines=[r'\begin{table}[H]',r'\centering\small',r'\caption{Completion on 100 strict-DFS sources and independent generation from the same checkpoints. $V_{\neg\mathrm{EOS}}$ and $C_{\neg\mathrm{EOS}}$ are mean valid/collision-free lengths (bricks) with EOS suppressed, from 64 draws. B/Q denote BrickNet/raw-Qwen initialization; R1/R2 jointly vary decay and dropout. Bold marks the largest point estimate per column, not statistical significance.}',r'\label{tab:main}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llrrrrr@{}}\toprule',r'ID & Dev step & $\Delta\ell$ & F1 normal & F1 long & $V_{\neg\mathrm{EOS}}$ & $C_{\neg\mathrm{EOS}}$ \\ \midrule']
 table_values={arm:statistics[arm]['completion_mean']+[statistics[arm]['no_eos_valid_length']['mean'],statistics[arm]['no_eos_collision_free_length']['mean']] for arm in PRIMARY}
 maxima=np.max(list(table_values.values()),axis=0)
 for arm in PRIMARY:
  step=new[arm]['model'].get('step','--');values=table_values[arm];cells=[f'{values[0]:+.3f}',f'{values[1]:.3f}',f'{values[2]:.3f}',f'{values[3]:.2f}',f'{values[4]:.2f}'];cells=[r'\textbf{'+v+'}' if abs(values[i]-maxima[i])<1e-12 else v for i,v in enumerate(cells)];lines.append(f'{IDS[arm]} & {step} & '+' & '.join(cells)+r' \\')
 lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(ROOT/'tables/main.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{table}[ht]\centering\small',r'\caption{Both stopping policies, 64 draws each. All lengths count bricks and are capped at 100. No-EOS is a separately sampled intervention, not deletion of EOS from a saved output.}',r'\label{tab:stopping}',r'\begin{tabular}{@{}lrrrr@{}}\toprule',r'ID & $V$ & $C$ & $V_{\neg\mathrm{EOS}}$ & $C_{\neg\mathrm{EOS}}$ \\ \midrule']
 for arm in PRIMARY:
  s=statistics[arm];v=[s[k]['mean'] for k in ['natural_valid_length','natural_collision_free_length','no_eos_valid_length','no_eos_collision_free_length']];lines.append(IDS[arm]+' & '+' & '.join(f'{x:.2f}' for x in v)+r' \\')
 lines += [r'\bottomrule\end{tabular}\end{table}'];(ROOT/'tables/stopping.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{table}[ht]\centering\small',r'\caption{Historical SFT branch on Old18 only. Completion receives no caption; independent generation uses 64 official validation captions. These rows are not part of the main PT comparison.}',r'\label{tab:sft}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',r'Model & $\Delta\ell$ & F1 normal & F1 long & Valid & Coll.-free \\ \midrule']
 for arm,label in [('frozen_sft','Released SFT'),('post_sft','SFT + ours / M-only'),('fullg_sft','SFT + ours / Full-G')]:
  c=old[arm]['metrics'];rs=phys[arm]['natural'];v=np.mean([x['valid_length'] for x in rs]);f=np.mean([x['collision_free_length'] for x in rs if x['collision_free_length'] is not None]);lines.append(f'{label} & {c["delta"]:+.3f} & {c["f1"]["original"]:.3f} & {c["f1"]["stretched"]:.3f} & {v:.2f} & {f:.2f} '+r'\\')
 lines += [r'\bottomrule\end{tabular}\end{table}'];(ROOT/'tables/sft.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{table}[htbp]\centering\small',r'\caption{All PT recipes on both suites. Old18 retains the original frontier-style inputs; New100 uses strict DFS and adds 82 constructed official-validation cases. A difference between columns is not an order-only treatment effect.}\label{tab:all}',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r' & \multicolumn{3}{c}{Old18} & \multicolumn{3}{c}{New100} \\',r'ID & $\Delta\ell$ & F1 normal & F1 long & $\Delta\ell$ & F1 normal & F1 long \\ \midrule']
 for arm in PRIMARY+['direct_bricknet','nzd_plain','nzd_masked']:
  if arm not in new:continue
  a=old[arm]['metrics'];b=new[arm]['metrics'];v=[a['delta'],a['f1']['original'],a['f1']['stretched'],b['delta'],b['f1']['original'],b['f1']['stretched']];label=IDS.get(arm,{'direct_bricknet':'Q-official70k','nzd_plain':'Q-v4-DFS','nzd_masked':'Q-v4-last'}.get(arm,arm));lines.append(label+' & '+' & '.join(f'{x:.3f}' for x in v)+r' \\')
 lines += [r'\bottomrule\end{tabular}\end{table}'];(ROOT/'tables/all_suites.tex').write_text('\n'.join(lines).replace('}\\\n\\toprule','}\\\\\n\\toprule')+'\n')
 with (ROOT/'generated/summary.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['id','arm','delta','f1_normal','f1_long','natural_valid','natural_collision_free','no_eos_valid','no_eos_collision_free']);
  for arm in new:
   s=statistics[arm];w.writerow([IDS.get(arm,arm),arm,*s['completion_mean'],*[s[k]['mean'] for k in ['natural_valid_length','natural_collision_free_length','no_eos_valid_length','no_eos_collision_free_length']]])
 # Paired effects; linear scale and assembly-level bootstrap.
 fig,ax=plt.subplots(figsize=(7.1,3.5));ys=np.arange(len(effectdata))[::-1]
 for y,e in zip(ys,effectdata):
  mean_=e['stretched_F1_difference'];lo,hi=e['ci95'];color='#157f79' if lo>0 else '#657389';ax.errorbar(mean_,y,xerr=[[mean_-lo],[hi-mean_]],fmt='o',color=color,ecolor=color,capsize=3,ms=5,lw=1.3)
 ax.axvline(0,color='#9aa2aa',lw=1,ls='--');ax.set_yticks(ys,[e['label'] for e in effectdata]);ax.set_xlabel('Difference in stretched-context Pose F1');ax.grid(axis='x',alpha=.18);fig.tight_layout();savefig(fig,'paired_effects')
 # Completion versus collision-free continuation, with separate empirical sampling uncertainty.
 fig,ax=plt.subplots(figsize=(7.1,4.2));offsets={'frozen_pt':(7,-13),'post_pt':(8,10),'fullg_pt':(7,-3),'plain':(-8,-15),'masked':(-15,7),'grpo_pt':(8,2),'direct_m':(7,10),'direct_g':(7,-10),'nzd_v2_plain':(-16,-17),'nzd_v2_masked':(4,10),'regularized_g':(4,-18),'regularized_g_wd010_do015':(-35,11)}
 for arm in PRIMARY:
  s=statistics[arm];x=s['no_eos_collision_free_length']['mean'];y=s['completion_avg_f1'];xl,xh=s['no_eos_collision_free_length']['ci95'];yl,yh=s['completion_avg_f1_ci95'];color='#2b6f99' if arm in ['frozen_pt','post_pt','fullg_pt','plain','masked','grpo_pt'] else '#d47a2f'
  ax.errorbar(x,y,xerr=[[x-xl],[xh-x]],yerr=[[y-yl],[yh-y]],fmt='*' if arm=='grpo_pt' else 'o',ms=10 if arm=='grpo_pt' else 5,color=color,ecolor=color,alpha=.8,lw=.8,capsize=2)
  ax.annotate(IDS[arm],(x,y),xytext=offsets[arm],textcoords='offset points',fontsize=9,color=color,bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.3})
 ax.set(xlabel='Collision-free prefix length with EOS suppressed (bricks)',ylabel='Mean completion Pose F1',xlim=(-1,32),ylim=(.20,.56));ax.grid(alpha=.18);fig.tight_layout();savefig(fig,'completion_vs_generation')
 # Training curves: CE losses are kept on separate axes for different objectives.
 fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
 settings=[('post_pt','fullg_pt','BrickNet initialization'),('direct_m','direct_g','Raw Qwen initialization')]
 curves={}
 for ax,(first,second,title) in zip(axes,settings):
  for arm,color in [(first,'#157f79'),(second,'#d47a2f')]:
   log=train[arm]['logs'];tr={x['optimizer_step']:x['loss'] for x in log if x.get('event')=='train'};dev={x['optimizer_step']:x['dev']['mean'] for x in log if x.get('event') in ['dev','baseline_dev']};best=train[arm]['best'];vals=np.array(list(tr.values()));smooth=[vals[max(0,i-49):i+1].mean() for i in range(len(vals))];keep=[s for s in dev if s>=250]
   ax.plot(list(tr),smooth,color=color,lw=1,label=IDS[arm]+' train');ax.plot(keep,[dev[s] for s in keep],'o--',color=color,lw=1,ms=3,label=IDS[arm]+' val');ax.axvline(best['optimizer_step'],color=color,lw=.7,ls=':');curves[arm]={'best':best,'final_dev':dev[max(dev)],'last50_train':float(np.mean(vals[-50:]))}
  ax.set_title(title,fontsize=10,fontweight='bold');ax.set(xlabel='Optimizer step',ylabel='Cross-entropy',xlim=(250,2250));ax.grid(alpha=.18);ax.legend(fontsize=9,frameon=False)
  # Exclude startup transient so validation divergence remains visible, without hiding data in the CSV.
  shown=[]
  for arm in [first,second]:
   shown.extend(x['dev']['mean'] for x in train[arm]['logs'] if x.get('event')=='dev' and x['optimizer_step']>=250)
  ax.set_ylim(0,max(shown)*1.2)
 fig.tight_layout();savefig(fig,'training_curves');(ROOT/'generated/loss_summary.json').write_text(json.dumps(curves,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(8.4,3.3))
 for arm,color in [('direct_g','#2b6f99'),('regularized_g','#d47a2f'),('regularized_g_wd010_do015','#157f79')]:
  log=train[arm]['logs'];tr={r['optimizer_step']:r['loss'] for r in log if r.get('event')=='train'};v=np.array(list(tr.values()));axes[0].plot(list(tr),[v[max(0,i-49):i+1].mean() for i in range(len(v))],color=color,label=IDS[arm],lw=1.3);dev=[r for r in log if r.get('event')=='dev' and r['optimizer_step']>=250];axes[1].plot([r['optimizer_step'] for r in dev],[r['dev']['mean'] for r in dev],'o-',color=color,label=IDS[arm],lw=1,ms=3)
 for ax,title in zip(axes,['Training CE: 50-step mean','Validation CE: steps 250 onward']):ax.set(title=title,xlabel='Optimizer step',ylabel='Cross-entropy');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
 axes[0].set_xlim(250,2250);axes[0].set_ylim(.15,.40);fig.tight_layout();savefig(fig,'regularization_curves')
 # GRPO's training objective has no comparable CE validation loss; report rollout/dev rewards instead.
 rl=read('grpo_training.json');fig,axes=plt.subplots(1,2,figsize=(8.4,3.0));logs=rl['train_log'];dev=rl['dev_log'];x=[r['step'] for r in logs];rewards=np.array([r['rollout_reward'] for r in logs]);axes[0].plot(x,[rewards[max(0,i-19):i+1].mean() for i in range(len(x))],label='Training rollout reward (mean20)',color='#157f79');axes[0].plot([r['step'] for r in dev],[r['reward'] for r in dev],'o-',label='Selection dev reward',color='#d47a2f',ms=3);axes[0].set(xlabel='RL update',ylabel='Pose F2 reward');axes[0].legend(fontsize=7,frameon=False)
 axes[1].plot(x,[r['kl'] for r in logs],color='#2b6f99',lw=1);axes[1].set(xlabel='RL update',ylabel='Logged reference KL estimate')
 for ax in axes:ax.grid(alpha=.18)
 fig.tight_layout();savefig(fig,'rl_curves')
 fig,axes=plt.subplots(1,2,figsize=(8.4,3.0))
 for arm,color in [('plain','#2b6f99'),('masked','#d47a2f')]:
  d=train[arm].get('summary')
  if d:axes[0].plot([x[0] for x in d['dev_curve']],[x[1] for x in d['dev_curve']],'o-',ms=3,lw=1,label=IDS[arm],color=color)
 for arm,color in [('nzd_v2_plain','#2b6f99'),('nzd_v2_masked','#d47a2f')]:
  logs=train[arm]['logs'];tr=[r for r in logs if r.get('event')=='train'];v=np.array([r['loss'] for r in tr]);axes[1].plot([r['optimizer_step'] for r in tr],[v[max(0,i-49):i+1].mean() for i in range(len(v))],lw=1,color=color,label=IDS[arm]+' train');dv=[r for r in logs if r.get('event')=='dev'];axes[1].plot([r['optimizer_step'] for r in dv],[r['dev']['mean'] for r in dv],'o--',ms=3,color=color,label=IDS[arm]+' val')
 for ax,title in zip(axes,['BrickNet paired data: validation CE','Qwen paired data: train / validation CE']):ax.set(title=title,xlabel='Optimizer step',ylabel='Cross-entropy');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=7)
 axes[1].set_xlim(250,2250);axes[1].set_ylim(.33,.52);fig.tight_layout();savefig(fig,'order_curves')
 # Stopping policies, shown for the RL parent and child.
 fig,axes=plt.subplots(1,2,figsize=(7.3,3.0));xx=np.arange(2);width=.34
 for ax,metric,title in zip(axes,['valid_length','collision_free_length'],['Parseable prefix','Collision-free prefix']):
  for j,(mode,label,color) in enumerate([('natural','EOS allowed','#8597ad'),('no_eos','EOS suppressed','#157f79')]):
   vals=[statistics[a][mode+'_'+metric]['mean'] for a in ['masked','grpo_pt']];cis=[statistics[a][mode+'_'+metric]['ci95'] for a in ['masked','grpo_pt']];ax.bar(xx+(j-.5)*width,vals,width,label=label,color=color);ax.errorbar(xx+(j-.5)*width,vals,yerr=np.array([[v-lo,hi-v] for v,(lo,hi) in zip(vals,cis)]).T,fmt='none',ecolor='#374151',capsize=3,lw=.8)
  ax.set_xticks(xx,['B-LATE (parent)','B-RL']);ax.set_title(title,fontsize=11);ax.set_ylabel('Mean length (bricks)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();savefig(fig,'eos_effect')
 # Length census, not an uncertainty estimate: each input counts once.
 ds=read('dataset_summary.json');fig,ax=plt.subplots(figsize=(7.2,2.8));bins=np.arange(0,101,10)-.5;center=(bins[:-1]+bins[1:])/2
 for c,color,off in [('original','#2b6f99',-2),('stretched','#d47a2f',2)]:
  vals=[r['n_prefix'] for r in ds['lengths'] if r['condition']==c];ax.bar(center+off,np.histogram(vals,bins)[0],3.8,label=c,color=color)
 ax.set(xlabel='Given context length (bricks, including the first target brick)',ylabel='Number of sources',xlim=(0,100));ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();savefig(fig,'context_lengths')
 print(json.dumps({'models':len(statistics),'paired_effects':effectdata},indent=2))
if __name__=='__main__':
 main()
 from finalize_analysis import build
 build(ROOT)
 import subprocess, sys
 subprocess.run([sys.executable, str(ROOT / "scripts/apply_proposal_dlo.py")], check=True)
 subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_manuscript_tables.py")], check=True)



