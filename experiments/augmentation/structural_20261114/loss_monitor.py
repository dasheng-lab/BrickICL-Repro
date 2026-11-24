"""Read-only training-log observer. Persist curves, CSVs and ETA every minute."""
import argparse,csv,fcntl,io,json,math,os,shutil,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'loss_curves'
LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer/training_structural')
ARMS={'PT_M':'PT M-only','SFT_M':'SFT M-only','PT_G':'PT Full-G','SFT_G':'SFT Full-G'}
ZONE=ZoneInfo('America/Los_Angeles')

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)

def read_events(path):
    train={};dev={};last=0;elapsed=0;segment=0;errors=[]
    if not path.exists():return train,dev,errors
    lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try:r=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1:errors.append(i+1)
            continue
        step=r.get('optimizer_step',0)
        if r['event']=='train':
            if step<=last or r['elapsed_seconds']<elapsed:
                train={k:v for k,v in train.items() if k<step}
                dev={k:v for k,v in dev.items() if k<step}
                segment+=1
            r['segment']=segment;train[step]=r;last=step;elapsed=r['elapsed_seconds']
        elif r['event'] in ('dev','baseline_dev'):dev[step]=r['dev']
    return train,dev,errors

def csv_file(path,columns,rows):
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=columns)
    writer.writeheader();writer.writerows(rows)
    tmp=path.with_suffix('.tmp');tmp.write_text(buf.getvalue());tmp.replace(path)

def render():
    OUT.mkdir(exist_ok=True)
    now=datetime.now(timezone.utc)
    workload=json.loads((ROOT/'training_workload.json').read_text())
    reports={};series={}
    for arm in ARMS:
        folder=ROOT/'training'/arm
        train,dev,errors=read_events(folder/'metrics.jsonl')
        values=[train[s] for s in sorted(train)]
        losses=np.asarray([r['loss'] for r in values],float)
        c=np.concatenate(([0.],np.cumsum(losses)))
        means=[float((c[i+1]-c[max(0,i-49)])/(i+1-max(0,i-49))) for i in range(len(values))]
        csv_file(OUT/f'{arm}_train.csv',
            ['step','train_loss','mean_50_steps','learning_rate','grad_norm','elapsed_seconds','peak_memory_gib','segment'],
            [dict(step=r['optimizer_step'],train_loss=r['loss'],mean_50_steps=m,learning_rate=r['lr'],
                  grad_norm=r['grad_norm'],elapsed_seconds=r['elapsed_seconds'],peak_memory_gib=r['peak_memory_gib'],segment=r['segment'])
             for r,m in zip(values,means)])
        csv_file(OUT/f'{arm}_dev.csv',['step','dev_mean','dev_v2','dev_v3'],
                 [dict(step=s,dev_mean=r['mean'],dev_v2=r['v2'],dev_v3=r['v3']) for s,r in sorted(dev.items())])
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'state':'queued'}
        if (folder/'COMPLETED.json').exists():state['state']='completed'
        elif state.get('pid') and not Path(f'/proc/{state["pid"]}').exists():state['state']='stopped'
        step=values[-1]['optimizer_step'] if values else 0
        total=values[-1]['total_steps'] if values else 2188
        # Same-objective measured historical duration, scaled by the actual padded input-token workload.
        reference=workload[arm]
        full_seconds=reference['historical_seconds']*reference['ratios']['input_tokens']
        method='historical H20 run scaled by padded input tokens'
        rate=None
        if len(values)>=20 and state['state']=='training':
            recent=[r for r in values if r['segment']==values[-1]['segment']][-201:]
            if len(recent)>=20:
                rate=(recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds'])/(recent[-1]['optimizer_step']-recent[0]['optimizer_step'])
                remaining=max(0,total-step)*rate
                method='last up to 200 optimizer steps, including any validation intervals'
            else:remaining=full_seconds*(1-step/total)
        else:remaining=0 if state['state']=='completed' else full_seconds*(1-step/total)
        low,high=(.9,1.15) if rate else (.85,1.2)
        reports[arm]={'label':ARMS[arm],'state':state['state'],'step':step,'total_steps':total,
            'latest_train_loss':float(losses[-1]) if len(losses) else None,
            'train_mean_50_steps':means[-1] if means else None,
            'latest_dev_step':max(dev) if dev else None,'latest_dev_loss':dev[max(dev)]['mean'] if dev else None,
            'best_dev_loss':min((r['mean'] for r in dev.values()),default=None),
            'seconds_per_step':rate,'estimated_remaining_hours':remaining/3600,
            'estimated_remaining_range_hours':[remaining*low/3600,remaining*high/3600],
            'eta_method':method,'malformed_log_lines':errors}
        series[arm]=(values,means,dev)
    remaining=sum(r['estimated_remaining_hours'] for r in reports.values())
    bounds=[sum(r['estimated_remaining_range_hours'][i] for r in reports.values()) for i in (0,1)]
    complete=all(r['state']=='completed' for r in reports.values())
    stopped=any(r['state']=='stopped' for r in reports.values())
    summary={'updated_at_utc':now.isoformat(),'updated_at_local':now.astimezone(ZONE).isoformat(),
        'timezone':'America/Los_Angeles','arms':reports,'all_completed':complete,'attention_needed':stopped,
        'estimated_queue_remaining_hours':remaining,'estimated_queue_remaining_range_hours':bounds,
        'estimated_finish_local':(now+timedelta(hours=remaining)).astimezone(ZONE).isoformat(),
        'estimated_finish_range_local':[(now+timedelta(hours=h)).astimezone(ZONE).isoformat() for h in bounds],
        'eta_note':'Planning estimate for sequential jobs on one H20. Queued jobs use historical timing; estimates update after each begins.'}
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    for ax,(arm,label) in zip(axes.flat,ARMS.items()):
        values,means,dev=series[arm];r=reports[arm]
        ax.set_title(f'{label}  |  {r["step"]}/{r["total_steps"]}',loc='left',fontsize=12,fontweight='bold')
        if values:
            steps=[x['optimizer_step'] for x in values]
            ax.plot(steps,[x['loss'] for x in values],color='#9bbac2',alpha=.6,linewidth=.6,label='Train raw')
            ax.plot(steps,means,color='#087f8c',linewidth=1.6,label='Train mean (50 steps)')
            if dev:ax.plot(sorted(dev),[dev[s]['mean'] for s in sorted(dev)],'o-',color='#c56d27',linewidth=1.2,markersize=4,label='Dev mean')
            ax.set_xlim(0,max(25,steps[-1])*1.02)
            ax.legend(fontsize=8,loc='upper right')
        else:
            ax.text(.5,.5,'Queued',ha='center',va='center',transform=ax.transAxes,color='#64748b',fontsize=16)
            ax.set_xlim(0,2188);ax.set_ylim(0,1)
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Loss (nats/token, sample mean)')
        ax.grid(True,alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('BrickICL structural post-training loss',fontsize=16,fontweight='bold',x=.055,ha='left')
    fig.text(.055,.025,'Updated '+now.astimezone(ZONE).strftime('%Y-%m-%d %H:%M %Z')+
             '  |  Dev evaluation every 250 steps and at completion',fontsize=9,color='#475569')
    fig.tight_layout(rect=[0,.045,1,.94])
    temp=OUT/'loss_curves.png.tmp';fig.savefig(temp,format='png',dpi=160);plt.close(fig)
    temp.replace(OUT/'loss_curves.png')
    write_json(OUT/'summary.json',summary)
    LIVE.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix not in ('.png','.csv','.json'):continue
        tmp=LIVE/(p.name+'.tmp');shutil.copyfile(p,tmp);tmp.replace(LIVE/p.name)
    if (ROOT/'loss_index.html').exists():
        tmp=LIVE/'index.html.tmp';shutil.copyfile(ROOT/'loss_index.html',tmp);tmp.replace(LIVE/'index.html')
    return complete,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    with (ROOT/'loss_monitor.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('A loss monitor is already running');return
        while True:
            complete,summary=render()
            print(json.dumps({'updated':summary['updated_at_local'],'steps':{a:r['step'] for a,r in summary['arms'].items()},
                              'remaining_hours':round(summary['estimated_queue_remaining_hours'],2),'completed':complete}),flush=True)
            if complete or not args.watch or (ROOT/'SUPERSEDED.json').exists():break
            time.sleep(60)
if __name__=='__main__':main()
