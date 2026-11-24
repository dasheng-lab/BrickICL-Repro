"""Estimate serial wall-clock time and save private loss curves. No web publishing."""
import fcntl,json,time,csv
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT,atomic

REPO=ROOT.parents[1];REG=REPO/'runs/qwen3_fullg_reg_20261119';BASE=REPO/'runs/qwen3_direct_pt_20261117/training/Full_G'
JOBS=[('Full_G_reg',REG/'training/Full_G_reg',REG/'EVAL_COMPLETED.json'),
      ('plain',ROOT/'training/plain',ROOT/'EVAL_COMPLETED.plain.json'),('masked',ROOT/'training/masked',ROOT/'EVAL_COMPLETED.masked.json')]
LABELS={'Full_G_reg':'Full-G regularized','plain':'nzd control PT','masked':'nzd augmented PT'}
def events(folder):
    tr={};dev={};p=folder/'metrics.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            s=r['optimizer_step']
            if r['event']=='train':
                tr={k:v for k,v in tr.items() if k<s};dev={k:v for k,v in dev.items() if k<s};tr[s]=r
            elif r['event'] in ('dev','baseline_dev'):dev[s]=r['dev']
    return tr,dev
def rate(tr):
    recent=list(tr.values())[-101:]
    if len(recent)<10:return None
    elapsed=recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds']
    return elapsed/(recent[-1]['optimizer_step']-recent[0]['optimizer_step']) if elapsed>0 else None
def render():
    now=datetime.now(timezone.utc);zone=ZoneInfo('America/Los_Angeles');out=ROOT/'loss_curves';out.mkdir(exist_ok=True)
    all_events={key:events(folder) for key,folder,_ in JOBS}
    fullg_rate=rate(all_events['Full_G_reg'][0]) or 14.0
    historical=rate(events(BASE)[0]) or 6.76
    workload=json.loads((ROOT/'DATA_READY.json').read_text()) if (ROOT/'DATA_READY.json').exists() else None
    ratios={a:workload['workload'][a]['padded_input_tokens']/workload['fullg_reference_padded_tokens'] if workload else 1.0 for a in ('plain','masked')}
    assumed={key:fullg_rate*(ratios[key] if key!='Full_G_reg' else 1) for key,_,_ in JOBS}
    observed_plain=rate(all_events['plain'][0])
    if observed_plain:assumed['masked']=observed_plain*ratios['masked']/ratios['plain']
    summaries=[];cumulative=0.;fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,(key,folder,eval_marker) in zip(axes,JOBS):
        tr,dev=all_events[key];steps=list(tr);step=steps[-1] if steps else 0
        trained=(folder/'COMPLETED.json').exists();evaluated=eval_marker.exists();seconds=rate(tr) or assumed[key]
        train_hours=0 if trained else (2188-step)*seconds/3600*1.03
        eval_hours=0 if evaluated else 850*max(1.,fullg_rate/historical)/3600
        cumulative+=train_hours+eval_hours
        state='completed' if evaluated else 'evaluation' if trained else 'training' if tr else 'queued'
        summaries.append({'job':key,'label':LABELS[key],'state':state,'step':step,'total_steps':2188,'seconds_per_step':seconds,
            'padded_token_ratio_to_fullg':ratios.get(key,1.),'training_remaining_hours':train_hours,'evaluation_remaining_hours':eval_hours,
            'estimated_finish_local':(now+timedelta(hours=cumulative)).astimezone(zone).isoformat()})
        losses=np.array([r['loss'] for r in tr.values()]);av=[float(losses[max(0,i-49):i+1].mean()) for i in range(len(losses))]
        if tr:
            ax.plot(steps,losses,lw=.5,alpha=.35,color='#9bbac2');ax.plot(steps,av,lw=1.5,color='#087f8c',label='Train mean (50)')
        if dev:ax.plot(sorted(dev),[dev[s]['mean'] for s in sorted(dev)],'o-',ms=3.5,color='#c56d27',label='Dev')
        if tr or dev:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Queued',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(f'{LABELS[key]} | {step}/2188',loc='left',fontsize=11,fontweight='bold');ax.set_xlabel('Optimizer step');ax.set_ylabel('Cross-entropy')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        for kind,rows in [('train',[dict(step=s,loss=r['loss'],mean_50=a) for (s,r),a in zip(tr.items(),av)]),('dev',[dict(step=s,**dev[s]) for s in sorted(dev)])]:
            if rows:
                with (out/f'{key}_{kind}.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'updated_local':now.astimezone(zone).isoformat(),'timezone':'America/Los_Angeles','jobs':summaries,
            'data_ready':workload is not None,'queue_remaining_hours':cumulative,'remaining_range_hours':[.75*cumulative,1.30*cumulative],
            'estimated_finish_local':(now+timedelta(hours=cumulative)).astimezone(zone).isoformat(),
            'finish_range_local':[(now+timedelta(hours=cumulative*x)).astimezone(zone).isoformat() for x in (.75,1.30)],
            'eta_basis':'Recent up to 100 Full-G updates, padded input-token ratios for queued arms, observed evaluation timing scaled by shared-GPU slowdown; 3% validation/save allowance.',
            'all_completed':(ROOT/'COMPLETED.json').exists(),'official_scheduler':False,'persistent_serial_queue':True}
    atomic(ROOT/'ETA.json',report)
    fig.suptitle('Three-job serial queue: training / validation loss',fontsize=15,fontweight='bold')
    fig.text(.04,.018,'Cluster files only. '+now.astimezone(zone).strftime('%Y-%m-%d %H:%M %Z'),fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.045,1,.93]);tmp=out/'loss_curves.png.tmp';fig.savefig(tmp,format='png',dpi=180);tmp.replace(out/'loss_curves.png');plt.close(fig)
    print(json.dumps({'updated':report['updated_local'],'steps':{r['job']:r['step'] for r in summaries},'remaining_hours':cumulative,'data_ready':report['data_ready']}),flush=True)
    return report['all_completed']
def main():
    lock=(ROOT/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if render():break
        time.sleep(60)
if __name__=='__main__':main()
