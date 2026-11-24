"""Loss artifacts and ETA, stored on the cluster only."""
import csv,fcntl,json,time
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT,atomic
REPO=ROOT.parents[1];OUT=ROOT/'loss_curves';TOTAL=2211
def events(folder):
    train={};dev={};p=folder/'metrics.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            s=r['optimizer_step']
            if r['event']=='train':
                train={k:v for k,v in train.items() if k<s};dev={k:v for k,v in dev.items() if k<s};train[s]=r
            elif r['event'] in ('dev','baseline_dev'):dev[s]=r['dev']
    return train,dev
def rate(rows):
    recent=list(rows.values())[-101:]
    pause_file=ROOT/'PRIORITY_PAUSE.json'
    if pause_file.exists():
        pause=json.loads(pause_file.read_text())
        if pause.get('state')=='resumed_after_no_eos_evaluation':
            recent=[r for r in recent if r['optimizer_step']>pause['step']]
    if len(recent)<15:return None
    delta=recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds']
    return delta/(recent[-1]['optimizer_step']-recent[0]['optimizer_step']) if delta>0 else None
def render():
    OUT.mkdir(exist_ok=True);ready=json.loads((ROOT/'DATA_READY.json').read_text()) if (ROOT/'DATA_READY.json').exists() else None
    history=json.loads((REPO/'runs/qwen3_three_job_queue_20261119/DATA_READY.json').read_text())
    old=REPO/'runs/qwen3_three_job_queue_20261119/training/masked';base_rate=rate(events(old)[0]) or 5.8
    new={a:events(ROOT/'training'/a) for a in ('plain','masked')};live_rate=rate(new['plain'][0]);summaries=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));remaining=0
    for ax,arm in zip(axes,('plain','masked')):
        train,dev=new[arm];values=list(train.values());step=values[-1]['optimizer_step'] if values else 0
        ratio=ready['workload'][arm]['padded_input_tokens']/history['workload']['masked']['padded_input_tokens'] if ready else .82
        estimate=rate(train) or live_rate or base_rate*ratio
        done=(ROOT/f'training/{arm}/COMPLETED.json').exists();eval_done=(ROOT/f'EVAL_COMPLETED.{arm}.json').exists()
        hours=(0 if done else (TOTAL-step)*estimate/3600*1.1)+(0 if eval_done else .3)
        remaining+=hours;loss=np.array([r['loss'] for r in values]);avg=[float(loss[max(0,i-49):i+1].mean()) for i in range(len(loss))]
        if train:
            ax.plot(list(train),loss,color='#9bbac2',alpha=.5,lw=.5,label='Train raw');ax.plot(list(train),avg,color='#087f8c',lw=1.5,label='Train mean (50)')
        if dev:ax.plot(sorted(dev),[dev[s]['mean'] for s in sorted(dev)],'o-',color='#c56d27',ms=4,label='Full v2 validation')
        if train or dev:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Queued',ha='center',transform=ax.transAxes)
        ax.set_title(('nzd v2 control PT' if arm=='plain' else 'nzd v2 augmented PT')+f' | {step}/{TOTAL}',loc='left',fontsize=12,fontweight='bold')
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Cross-entropy');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        for kind,rows in [('train',[{'step':s,'loss':r['loss'],'mean50':m} for (s,r),m in zip(train.items(),avg)]),('val',[{'step':s,**dev[s]} for s in sorted(dev)])]:
            if rows:
                with (OUT/f'{arm}_{kind}.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summaries.append({'arm':arm,'step':step,'state':'completed' if eval_done else 'evaluating' if done else 'training' if train else 'queued',
            'latest_train_mean50':avg[-1] if avg else None,'latest_val':dev[max(dev)]['mean'] if dev else None,
            'seconds_per_step':estimate,'remaining_hours':hours})
    now=datetime.now(timezone.utc);zone=ZoneInfo('America/Los_Angeles');complete=(ROOT/'COMPLETED.json').exists()
    pause=json.loads((ROOT/'PRIORITY_PAUSE.json').read_text()) if (ROOT/'PRIORITY_PAUSE.json').exists() else {}
    priority_paused=pause.get('state')=='paused_for_priority_evaluation'
    if priority_paused:
        for r in summaries:
            if r['state']=='training':r['state']='paused_for_no_eos_evaluation'
    result={'updated_local':now.astimezone(zone).isoformat(),'models':summaries,'remaining_hours':remaining,
        'remaining_range_hours':[remaining*.8,remaining*1.35],'estimated_finish_local':(now+timedelta(hours=remaining)).astimezone(zone).isoformat(),
        'all_completed':complete,'includes_dev_and_evaluation':True,'data_ready':ready is not None,'priority_paused':priority_paused,
        'eta_basis':'Recent measured steps when available; otherwise last Qwen3 v4 masked rate scaled by paired padded tokens; 10% allowance for full 2133-row dev evaluations.'}
    if priority_paused:result['estimated_finish_local']=None
    atomic(ROOT/'ETA.json',result)
    fig.suptitle('Original Qwen3-0.6B PT | exact nzd v2 training and dev data',fontsize=14,fontweight='bold')
    fig.text(.05,.018,'70722 paired train rows, 2133 paired dev rows. Independent raw-Qwen3 initialization; no BrickNet adapter. '+now.astimezone(zone).strftime('%m-%d %H:%M %Z'),fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.045,1,.93])
    for ext in ('png','pdf'):
        tmp=OUT/f'loss_curves.{ext}.tmp';fig.savefig(tmp,format=ext,dpi=180);tmp.replace(OUT/f'loss_curves.{ext}')
    plt.close(fig);print(json.dumps(result),flush=True);return complete
def main():
    lock=(ROOT/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if render():break
        time.sleep(60)
if __name__=='__main__':main()
