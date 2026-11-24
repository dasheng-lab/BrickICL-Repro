import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=ROOT/'delivery';OUT.mkdir(exist_ok=True)
RUNS=[('nzd_v2_plain','qwen3_nzd_v2_plain_eval_20261120','qwen3_nzd_v2_20261120/training/plain','Qwen: DFS control'),('nzd_v2_masked','qwen3_nzd_v2_masked_eval_20261120','qwen3_nzd_v2_20261120/training/masked','Qwen: order augmentation'),('regularized_g_wd010_do015','qwen3_fullg_reg_wd010_do015_eval_20261121','qwen3_fullg_reg_wd010_do015_20261121/training/Full_G_reg','Qwen: Full-G, WD .10 / dropout .15')]
def j(p):return json.loads(p.read_text())
def losses(folder):
 tr={};dev={}
 for line in (folder/'metrics.jsonl').open():
  r=json.loads(line);s=r['optimizer_step']
  if r['event']=='train':tr[s]=r['loss']
  elif r['event'] in ('dev','baseline_dev'):dev[s]=r['dev']['mean']
 best=j(folder/'best.json');vals=list(tr.values());avg=[float(np.mean(vals[max(0,i-49):i+1])) for i in range(len(vals))]
 return tr,dev,avg,best
def main():
 data=[];lossdata=[]
 for arm,ev,train,label in RUNS:
  p=REPO/'runs'/ev;assert (p/'FOX_PATCH_VERIFIED.json').exists() and (p/'no_eos/COMPLETED.json').exists()
  m=j(p/'results'/arm/'metrics.json');m['no_eos']=j(p/'no_eos/results'/arm/'metrics.json');m['evaluation_root']=str(p)
  assert m['completion']['1']['samples']==144 and m['physical']['samples']==64 and m['no_eos']['eos_suppression_verified']
  folder=REPO/'runs'/train;tr,dev,avg,best=losses(folder);final=max(tr)
  m['loss']={'best_step':best['optimizer_step'],'best_val':best['dev_mean'],'final_val':dev[final],'final_train_mean50':avg[-1],'steps':final,'hours':j(folder/'COMPLETED.json')['elapsed_seconds']/3600}
  data.append(m);lossdata.append((tr,dev,avg,best,label))
  with (OUT/f'{arm}_loss.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['step','train','train_mean50','validation']);w.writerows([s,tr.get(s),avg[s-1] if s in tr else '',dev.get(s)] for s in sorted(set(tr)|set(dev)))
 (OUT/'metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(18,5))
 for ax,(tr,dev,avg,best,label) in zip(axes,lossdata):
  ax.plot(list(tr),list(tr.values()),color='#7faebd',alpha=.25,lw=.4,label='Train raw')
  ax.plot(list(tr),avg,color='#087f8c',lw=1.4,label='Train: 50-step mean')
  ax.plot(list(dev),list(dev.values()),'o-',color='#c56d27',ms=4,lw=1.4,label='Validation')
  ax.axvline(best['optimizer_step'],ls=':',color='#777',lw=1)
  ax.set(title=label+'\nDev-best step '+str(best['optimizer_step']),xlabel='Optimizer step',ylabel='Cross-entropy');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Completed Qwen3-0.6B post-training runs',fontsize=16,fontweight='bold');fig.tight_layout(rect=[0,.035,1,.94])
 fig.text(.02,.015,'Selection uses validation loss, not test scores. Control/augmentation use paired dev sequences; Full-G uses a different dev set.',fontsize=9)
 for ext in ('png','pdf'):fig.savefig(OUT/f'loss_three_models.{ext}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 comparison=[]
 for folder,label,color in [('qwen3_direct_pt_20261117/training/Full_G','WD .01 / dropout .05','#2166ac'),('qwen3_fullg_reg_20261119/training/Full_G_reg','WD .05 / dropout .10','#c65d16'),(RUNS[2][2],'WD .10 / dropout .15','#17845f')]:
  tr,dev,avg,best=losses(REPO/'runs'/folder);steps=[s for s in dev if s>=250]
  axes[0].plot(list(tr),avg,label=label,color=color,lw=1.5);axes[1].plot(steps,[dev[s] for s in steps],'o-',label=label,color=color,ms=4,lw=1.5)
  comparison.append({'label':label,'best':best,'final_validation':dev[max(dev)]})
 for ax,title in zip(axes,['Training CE (50-step mean)','Validation CE (from step 250)']):
  ax.set(title=title,xlabel='Optimizer step',ylabel='Cross-entropy');ax.grid(alpha=.2);ax.legend(fontsize=9)
 fig.suptitle('Full-G regularization: identical data and initialization',fontweight='bold');fig.tight_layout(rect=[0,0,1,.93])
 for ext in ('png','pdf'):fig.savefig(OUT/f'regularization_comparison.{ext}',dpi=180)
 (OUT/'regularization_loss.json').write_text(json.dumps(comparison,indent=2))
 print(json.dumps([{**m['loss'],'model':m['model']['key']} for m in data]),flush=True)
if __name__=='__main__':main()
