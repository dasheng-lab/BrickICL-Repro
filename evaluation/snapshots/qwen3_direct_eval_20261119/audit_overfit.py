"""Read-only data diversity / within-objective loss audit; no new training."""
import json
from collections import Counter
from statistics import mean,median
from pathlib import Path

REPO=Path('/home/sunshq114514/brickICL');OLD=REPO/'runs/structural_20261114';TRAIN=REPO/'runs/qwen3_direct_pt_20261117'
report={'structured':{},'official':{},'loss':{},'nzd_read_access':{}}
all_rows=[]
for version in ('v2','v3'):
    rows=[json.loads(s) for s in (OLD/f'cache/PT_M/{version}/train.index.jsonl').open()]
    all_rows.extend(rows)
    report['structured'][version]={'rows':len(rows),'sources':len({r['source_key'] for r in rows}),
        'families':len({r['source_family'] for r in rows}),'edit_types':dict(Counter(r['edit']['kind'] for r in rows)),
        'mean_target_tokens':mean(r['target_length'] for r in rows),'median_target_tokens':median(r['target_length'] for r in rows)}
families=Counter(r['source_family'] for r in all_rows)
report['structured']['total']={'rows':len(all_rows),'sources':len({r['source_key'] for r in all_rows}),
    'families':len(families),'rows_per_family_mean':mean(families.values()),'rows_per_family_max':max(families.values()),
    'edit_types':dict(Counter(r['edit']['kind'] for r in all_rows))}
for scope in ('M','G'):
    rows=[json.loads(s) for v in ('v2','v3') for s in (OLD/f'cache/PT_{scope}/{v}/train.index.jsonl').open()]
    report['structured'][scope+'_supervision']={'input_tokens':sum(r['length'] for r in rows),
        'target_tokens':sum(r['target_length'] for r in rows),'mean_target_tokens':mean(r['target_length'] for r in rows)}
families=set();count=0
for line in (TRAIN/'official_selected.jsonl').open():
    r=json.loads(line);families.add(r['source_key']);count+=1
report['official']={'rows':count,'sources':len(families)}
for arm in ('M_only','Full_G','BrickNet'):
    events=[json.loads(s) for s in (TRAIN/f'training/{arm}/metrics.jsonl').open()]
    train={r['optimizer_step']:r['loss'] for r in events if r['event']=='train'}
    dev={r['optimizer_step']:r['dev']['mean'] for r in events if r['event'] in ('dev','baseline_dev')}
    best=json.loads((TRAIN/f'training/{arm}/best.json').read_text())['optimizer_step']
    report['loss'][arm]={'best_step':best,'best_dev':dev[best],'last_dev':dev[2188],
        'dev_relative_increase':dev[2188]/dev[best]-1,
        'train_mean_50_at_best':mean(train[s] for s in range(best-49,best+1)),
        'train_mean_50_at_last':mean(train[s] for s in range(2139,2189))}
for folder in ('train_v4_mix','dev_v4'):
    for arm in ('masked','plain'):
        p=Path('/home/nzd0420/brickICL/runs/nzd-supertry')/folder/(arm+'.jsonl')
        try:
            with p.open() as f:r=json.loads(next(f))
            report['nzd_read_access'][str(p)]={'readable':True,'first_row_keys':list(r),'source':r.get('source'),'origin':r.get('origin')}
        except (OSError,ValueError) as e:report['nzd_read_access'][str(p)]={'readable':False,'error':str(e)}
out=REPO/'runs/qwen3_direct_eval_20261119/OVERFIT_AUDIT.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
