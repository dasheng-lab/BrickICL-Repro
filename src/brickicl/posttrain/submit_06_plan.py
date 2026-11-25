"""Freeze a shared reference-last schedule and two authorized 0.6B training configs."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('runs/reference_last_06'))
    parser.add_argument('--out',type=Path,default=Path('runs/bricknet_06_reference_last'))
    args = parser.parse_args(argv)
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    data=args.data.resolve()
    ready=json.loads((data/'READY.json').read_text());assert ready['status']=='ready'
    indices={};families={}
    for name in ('v2','v3'):
        indices[name]=[json.loads(s) for s in (data/name/'train.index.jsonl').open()]
        groups={}
        for i,r in enumerate(indices[name]):groups.setdefault(r['source_family'],{}).setdefault(r['condition'],[]).append(i)
        families[name]=groups
    rng=random.Random(42);datasets=['v2']*35000+['v3']*35000;rng.shuffle(datasets)
    schedule=[];counts=Counter();tokens=0
    for name in datasets:
        f=rng.choice(sorted(families[name]));c=rng.choice(sorted(families[name][f]));i=rng.choice(families[name][f][c])
        schedule.append({'dataset':name,'row_index':i,'length':indices[name][i]['length']})
        counts[name]+=1;counts[f'{name}:{c}']+=1;tokens+=indices[name][i]['length']
    # Preserve selected draws while reducing padding only within short shuffled windows.
    batches=[]
    for start in range(0,len(schedule),256):
        block=sorted(schedule[start:start+256],key=lambda r:r['length'])
        mini=[block[i:i+4] for i in range(0,len(block),4)];rng.shuffle(mini);batches.extend(mini)
    schedule_path=root/'batches.jsonl'
    with schedule_path.open('w') as f:
        for b in batches:f.write(json.dumps(b)+'\n')
    schedule_hash=hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    dev=[]
    for name in ('v2','v3'):
        rs=[json.loads(s) for s in (data/name/'dev.index.jsonl').open()]
        grouped={}
        for i,r in enumerate(rs):grouped.setdefault(r['source_family'],[]).append(i)
        pools=[list(v) for _,v in sorted(grouped.items())];chooser=random.Random(2026)
        for p in pools:chooser.shuffle(p)
        selected=[]
        while len(selected)<min(128,len(rs)):
            for p in pools:
                if p and len(selected)<128:selected.append(p.pop())
        dev.extend({'dataset':name,'row_index':i} for i in selected)
    (root/'dev_selection.json').write_text(json.dumps(dev,indent=2)+'\n')
    for mode in ('PT','SFT'):
        config={'mode':mode,'size':'0.6B','data_dir':str(data),'output_dir':str((root/mode).resolve()),
            'schedule':str(schedule_path.resolve()),'schedule_sha256':schedule_hash,'dev_selection':str((root/'dev_selection.json').resolve()),
            'ready_sha256':hashlib.sha256((data/'READY.json').read_bytes()).hexdigest(),
            'train_input_sha256':{name:ready['artifact_sha256'][f'{name}/train.u32'] for name in ('v2','v3')},
            'cache_sha256':{k:v for k,v in ready['artifact_sha256'].items() if
                            any(f'/{s}.' in k for s in ('train','dev')) and (k.endswith('.u32') or k.endswith('.index.jsonl'))},
            'code_sha256':{name:hashlib.sha256((Path('src/brickicl/posttrain')/name).read_bytes()).hexdigest()
                           for name in ('train_bricknet06.py','bricknet_runtime.py','data.py')},
            'micro_batch':4,'gradient_accumulation':8,'draws':70000,'epochs':1,
            'lr':1e-4,'weight_decay':.01,'warmup_ratio':.03,'max_grad_norm':1.,'eval_steps':250,'save_steps':250,
            'seed':42,'gradient_checkpointing':False,'max_length':6400,
            'initial_adapters':['kulits/BrickNet-0.6B-PT']+(['kulits/BrickNet-0.6B-SFT'] if mode=='SFT' else []),
            'loss':'mean per example over completion tokens including EOS; prompt/pad ignored',
            'reference_policy':'one complete visible reference immediately before target, prime=0',
            'caption':False,'test_access':'not loaded by trainer','status':'authorized_queued'}
        (root/f'{mode}.json').write_text(json.dumps(config,indent=2)+'\n')
    summary={'rows':70000,'micro_batches':len(batches),'optimizer_steps':(len(batches)+7)//8,'counts':dict(counts),
             'real_tokens':tokens,'schedule_sha256':schedule_hash,'dev_rows':len(dev),'test_rows_used':0}
    (root/'submission_plan.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
