import csv,io,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,TRAIN,atomic

def main():
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,5.1));summaries={}
    for ax,arm,label in zip(axes,('BrickNet','M_only','Full_G'),('Qwen3 / BrickNet data','Qwen3 / M-only','Qwen3 / Full-G')):
        run=TRAIN/arm;events=[json.loads(s) for s in (run/'metrics.jsonl').open()]
        tr={r['optimizer_step']:r for r in events if r['event']=='train'}
        dev={r['optimizer_step']:r['dev'] for r in events if r['event'] in ('dev','baseline_dev')}
        assert list(tr)==list(range(1,2189)) and sorted(dev)==[0,250,500,750,1000,1250,1500,1750,2000,2188]
        best=json.loads((run/'best.json').read_text());loss=np.array([r['loss'] for r in tr.values()]);avg=[loss[max(0,i-49):i+1].mean() for i in range(len(loss))]
        ax.plot(list(tr),loss,lw=.55,alpha=.5,color='#9bbac2',label='Train raw')
        ax.plot(list(tr),avg,lw=1.5,color='#087f8c',label='Train mean (50 steps)')
        ax.plot(sorted(dev),[dev[s]['mean'] for s in sorted(dev)],'o-',ms=4,lw=1.4,color='#c56d27',label='Validation')
        ax.axvline(best['optimizer_step'],ls=':',color='#666666',lw=1,label='Dev-best')
        ax.set_title(label+f' | best step {best["optimizer_step"]}',fontsize=12,fontweight='bold',loc='left')
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Loss (nats/token; sample mean)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8);ax.set_xlim(0,2240)
        for name,rows in [('train',[{'step':s,'loss':r['loss'],'mean_50':float(m),'learning_rate':r['lr']} for (s,r),m in zip(tr.items(),avg)]),
                          ('val',[{'step':s,**dev[s]} for s in sorted(dev)])]:
            with (out/f'{arm}_{name}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summaries[arm]={'best_step':best['optimizer_step'],'best_val_loss':best['dev_mean'],'final_val_loss':dev[2188]['mean'],
                        'initial_val_loss':dev[0]['mean'],'final_train_loss':loss[-1],'steps':2188}
    fig.suptitle('Direct Qwen3-0.6B PT | training and validation loss',fontsize=16,fontweight='bold')
    fig.text(.045,.025,'All runs completed. M-only supervises the missing part; the other two supervise full paths. Full-G and BrickNet data share the same validation set.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.055,1,.92])
    for ext in ('png','pdf','svg'):fig.savefig(out/f'training_val_loss.{ext}',dpi=200)
    plt.close(fig);atomic(out/'loss_summary.json',summaries)
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
