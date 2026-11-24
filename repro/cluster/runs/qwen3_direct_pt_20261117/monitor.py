import csv,fcntl,io,json,shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT,atomic

OUT=ROOT/'loss_curves';LIVE=ROOT.parents[2]/'brickicl-viewer/baseline/repro/viewer/training_qwen3_direct'
LABELS={'M_only':'Raw Qwen3 / M-only','Full_G':'Raw Qwen3 / Full-G','BrickNet':'Raw Qwen3 / BrickNet data'}
def render():
    OUT.mkdir(exist_ok=True);LIVE.mkdir(parents=True,exist_ok=True);summaries={}
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,(arm,label) in zip(axes,LABELS.items()):
        folder=ROOT/'training'/arm;events=[];train={};dev={}
        if (folder/'metrics.jsonl').exists():
            for line in (folder/'metrics.jsonl').read_text().splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:continue
        for row in events:
            step=row['optimizer_step']
            if row['event']=='train':
                train={s:r for s,r in train.items() if s<step};dev={s:r for s,r in dev.items() if s<step};train[step]=row
            elif row['event'] in ('dev','baseline_dev'):dev[step]=row['dev']
        values=list(train.values());loss=np.asarray([r['loss'] for r in values]);steps=list(train)
        smoothed=[float(loss[max(0,i-49):i+1].mean()) for i in range(len(loss))]
        if values:
            ax.plot(steps,loss,color='#9bbac2',alpha=.55,lw=.6,label='Train raw')
            ax.plot(steps,smoothed,color='#087f8c',lw=1.6,label='Train mean (50 steps)')
        if dev:ax.plot(list(dev),[r['mean'] for r in dev.values()],'o-',color='#c56d27',ms=4,label='Dev')
        ax.set_title(f'{label} | {steps[-1] if steps else 0}/2188',fontsize=11,fontweight='bold',loc='left')
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Loss (nats/token, sample mean)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        if values or dev:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Queued',ha='center',transform=ax.transAxes)
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'state':'queued'}
        recent=values[-100:];rate=None
        if len(recent)>10:rate=(recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds'])/(recent[-1]['optimizer_step']-recent[0]['optimizer_step'])
        summaries[arm]={'step':steps[-1] if steps else 0,'state':state['state'],'latest_train_loss':float(loss[-1]) if len(loss) else None,
            'latest_dev_loss':dev[max(dev)]['mean'] if dev else None,'seconds_per_step':rate,
            'estimated_remaining_hours':(2188-steps[-1])*rate/3600 if rate else None}
        for kind,rows in [('train',[dict(step=s,loss=r['loss'],mean_50=m,learning_rate=r['lr'],elapsed_seconds=r['elapsed_seconds']) for (s,r),m in zip(train.items(),smoothed)]),
                          ('dev',[dict(step=s,**r) for s,r in dev.items()])]:
            if not rows:continue
            buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (OUT/f'{arm}_{kind}.csv').write_text(buf.getvalue())
    stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
    fig.suptitle('Direct Qwen3-0.6B PT | fresh LoRA r=64 | no BrickNet initialization',fontsize=14,fontweight='bold')
    fig.text(.04,.015,'M-only masks context; Full-G / BrickNet supervise the full path. Different objective losses are not directly comparable. Updated '+stamp,fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.04,1,.93]);tmp=OUT/'loss_curves.png.tmp';fig.savefig(tmp,format='png',dpi=180);tmp.replace(OUT/'loss_curves.png');plt.close(fig)
    status={'updated_local':stamp,'arms':summaries,'all_completed':(ROOT/'COMPLETED.json').exists()}
    for name in ('QUEUE_STATUS','QUEUE_FAILED','official_build_status','OFFICIAL_READY','OFFICIAL_FAILED','SMOKE_PASSED'):
        if (ROOT/f'{name}.json').exists():status[name]=json.loads((ROOT/f'{name}.json').read_text())
    atomic(OUT/'summary.json',status)
    (OUT/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="120"><title>Qwen3 直接 PT</title>
<style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:20px;background:#f5f7fa}}img{{width:100%}}</style>
<h1>原始 Qwen3-0.6B 直接 PT：BrickNet / M-only / Full-G</h1><p>LoRA r=64、alpha=128；每组 7 万条，有效 batch 32，共 2188 步。无 BrickNet 权重、无 caption。{stamp}</p>
<p>更新步数：{' | '.join(a+': '+str(r['step'])+'/2188 ('+r['state']+')' for a,r in summaries.items())}</p>
<img src="loss_curves.png?v={int(time.time())}"><p><a href="summary.json">状态和耗时估计</a></p>
<p>M-only loss 只计算待补部分，Full-G / BrickNet 计算全序列；不同训练目标 loss 不直接比较。Full-G 和 BrickNet 使用相同开发集。</p>''')
    for f in OUT.iterdir():
        if f.suffix not in ('.json','.png','.csv','.html'):continue
        tmp=LIVE/(f.name+'.tmp');shutil.copy2(f,tmp);tmp.replace(LIVE/f.name)
    print(json.dumps(status,ensure_ascii=False),flush=True);return status['all_completed']
def main():
    lock=(ROOT/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if render():break
        time.sleep(60)
if __name__=='__main__':main()
