"""Save regularization loss comparisons on the cluster only; no web publishing."""
import csv,fcntl,json,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT,BASELINE,atomic

OUT=ROOT/'loss_curves'
def load(folder):
    train={};dev={}
    file=folder/'metrics.jsonl'
    if file.exists():
        for line in file.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            s=r['optimizer_step']
            if r['event']=='train':
                train={k:v for k,v in train.items() if k<s};dev={k:v for k,v in dev.items() if k<s};train[s]=r
            elif r['event'] in ('dev','baseline_dev'):dev[s]=r['dev']
    return train,dev
def render():
    OUT.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(12.5,4.6));report={}
    for key,folder,color,label in [('baseline',BASELINE/'training/Full_G','#2166ac','Original: dropout .05 / decay .01'),
                                  ('regularized',ROOT/'training/Full_G_reg','#c65d16','Regularized: dropout .10 / decay .05')]:
        train,dev=load(folder);values=list(train.values());loss=np.asarray([v['loss'] for v in values]);steps=list(train)
        means=[float(loss[max(0,i-49):i+1].mean()) for i in range(len(loss))]
        if values:axes[0].plot(steps,means,color=color,label=label,lw=1.5)
        if dev:axes[1].plot(sorted(dev),[dev[s]['mean'] for s in sorted(dev)],'o-',ms=4,color=color,label=label,lw=1.4)
        for kind,rows in [('train',[dict(step=s,loss=r['loss'],mean_50=m,learning_rate=r['lr']) for (s,r),m in zip(train.items(),means)]),
                          ('dev',[dict(step=s,**dev[s]) for s in sorted(dev)])]:
            if rows:
                with (OUT/f'{key}_{kind}.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'state':'preflight'}
        recent=values[-101:];rate=None
        if len(recent)>10:rate=(recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds'])/(recent[-1]['optimizer_step']-recent[0]['optimizer_step'])
        report[key]={'step':steps[-1] if steps else 0,'state':state['state'],'latest_loss_mean50':means[-1] if means else None,
                     'latest_dev_loss':dev[max(dev)]['mean'] if dev else None,'best_dev':min((d['mean'] for d in dev.values()),default=None),
                     'seconds_per_step':rate,'estimated_remaining_hours':(2188-steps[-1])*rate/3600 if rate else None}
    for ax,title in zip(axes,('Training cross-entropy (50-step mean)','Validation cross-entropy (no penalty term)')):
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold');ax.set_xlabel('Optimizer step');ax.set_ylabel('Loss');ax.set_xlim(0,2240)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
    fig.suptitle('Qwen3 direct Full-G: regularization comparison',fontsize=15,fontweight='bold')
    fig.text(.055,.018,'Same data, initialization, rank, learning rate, batch order and step budget. '+stamp,fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.04,1,.93])
    for ext in ('png','pdf'):
        temp=OUT/f'loss_comparison.{ext}.tmp';fig.savefig(temp,format=ext,dpi=180);temp.replace(OUT/f'loss_comparison.{ext}')
    plt.close(fig);report['updated_local']=stamp;report['all_completed']=(ROOT/'COMPLETED.json').exists()
    atomic(OUT/'summary.json',report);print(json.dumps(report),flush=True)
    return report['all_completed']
def main():
    lock=(ROOT/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if render():break
        time.sleep(60)
if __name__=='__main__':main()
