import hashlib
import json
import math
import random
from collections import Counter,defaultdict
from pathlib import Path
import shutil

ROOT=Path('runs/dfs_posttrain_20261114').resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bucket(n):
    if n<=32:return '4-32'
    if n<=64:return '33-64'
    if n<100:return '65-99'
    return '100'

def main():
    source=Path('brickicl/posttrain')
    # Snapshot the executed training sources to keep ongoing runs reproducible.
    for name in ('bricknet_runtime.py','data.py'):
        shutil.copyfile(source/name,ROOT/name)
    text=(source/'train_fullg06.py').read_text()
    text=text.replace('from brickicl.posttrain.bricknet_runtime import','from bricknet_runtime import')
    text=text.replace('from brickicl.posttrain.data import CompletionCollator','from data import CompletionCollator')
    (ROOT/'train.py').write_text(text)
    summary={}
    for mode in ('PT','SFT'):
        paths=('pt','sft') if mode=='PT' else ('sft',)
        hist=Counter()
        for split in paths:
            hist.update(json.loads((ROOT/f'official_{split}_audit.json').read_text())['length_histogram'])
        weights=Counter()
        for n,count in hist.items():weights[bucket(int(n))]+=count
        # One row plan per initialization, shared between M-only and Full-G.
        records={v:[json.loads(s) for s in (ROOT/f'cache/{mode}_G/{v}/train.index.jsonl').open()] for v in ('v2','v3')}
        grouped={}
        for v,rs in records.items():
            pool={}
            for i,r in enumerate(rs):
                pool.setdefault(bucket(r['selected_parts']),{}).setdefault(r['source_family'],{}).setdefault(r['condition'],[]).append(i)
            grouped[v]=pool
        rng=random.Random(42)
        versions=['v2']*35000+['v3']*35000;rng.shuffle(versions)
        refs=[];counts=Counter();unique=set()
        for v in versions:
            bins=sorted(grouped[v])
            b=rng.choices(bins,weights=[weights[x] for x in bins],k=1)[0]
            fam=rng.choice(sorted(grouped[v][b]))
            cond=rng.choice(sorted(grouped[v][b][fam]))
            i=rng.choice(grouped[v][b][fam][cond])
            refs.append({'dataset':v,'row_index':i})
            unique.add((v,i));counts[f'{v}/{b}']+=1
        # Length buckets use a common ordering for M and G.
        batches=[]
        for j in range(0,len(refs),256):
            buf=sorted(refs[j:j+256],key=lambda r:records[r['dataset']][r['row_index']]['length'])
            local=[buf[k:k+4] for k in range(0,len(buf),4)];rng.shuffle(local);batches.extend(local)
        plan=ROOT/f'{mode}_schedule.jsonl'
        plan.write_text(''.join(json.dumps(b)+'\n' for b in batches))
        dev=[]
        for v in ('v2','v3'):
            n=sum(1 for _ in (ROOT/f'cache/{mode}_G/{v}/dev.index.jsonl').open())
            assert n>0
            dev.extend({'dataset':v,'row_index':i} for i in random.Random(42).sample(range(n),min(128,n)))
        devfile=ROOT/f'{mode}_dev_selection.json'
        devfile.write_text(json.dumps(dev))
        for scope in ('M','G'):
            arm=f'{mode}_{scope}';cache=ROOT/'cache'/arm
            ready=json.loads((cache/'READY.json').read_text())
            output=ROOT/'training'/arm;output.mkdir(parents=True,exist_ok=True)
            cfg={'mode':mode,'objective':scope,'output_dir':str(output),'data_dir':str(cache),
                'schedule':str(plan),'schedule_sha256':sha(plan),'ready_sha256':sha(cache/'READY.json'),
                'train_input_sha256':{v:sha(cache/v/'train.u32') for v in ('v2','v3')},
                'cache_sha256':ready['artifact_sha256'],'code_sha256':{n:sha(ROOT/n) for n in ('train.py','bricknet_runtime.py','data.py')},
                'dev_selection':str(devfile),'seed':42,'draws':70000,'micro_batch':4,'gradient_accumulation':8,
                'lr':1e-4,'weight_decay':.01,'max_grad_norm':1.,'warmup_ratio':.03,'eval_steps':250,
                'gradient_checkpointing':scope=='G','max_length':6400,
                'initial_adapters':['kulits/BrickNet-0.6B-PT']+(['kulits/BrickNet-0.6B-SFT'] if mode=='SFT' else []),
                'caption': 'grounded caption before both partial and whole-G inputs' if mode=='SFT' else None,
                'order':'strict DFS' if scope=='G' else 'DFS context/reference then DFS missing parts',
                'lora':{'rank':16,'alpha':32,'dropout':.05},
                'sampling':'50/50 v2/v3; shipped same-initialization length-bin weights renormalized to available bins; uniform family/condition/row within bin',
                'length_bins':dict(weights),'length_reference_files':list(paths),
                'length_reference_scope':'all shipped rows in these files, all saved sampling rounds'}
            (output/'config.json').write_text(json.dumps(cfg,indent=2))
        summary[mode]={'draws':70000,'unique_rows':len(unique),'draw_lengths':dict(counts),'official_bin_counts':dict(weights),'reference_files':list(paths),
                       'available_bins':{v:sorted(grouped[v]) for v in grouped},'optimizer_steps':math.ceil(len(batches)/8)}
    (ROOT/'TRAINING_READY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
