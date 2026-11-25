"""Create mode-specific full-G experiment configs without changing the old M-only runs."""
import hashlib,json,random
from collections import Counter
from pathlib import Path
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('runs/fullg_06'))
    parser.add_argument('--out',type=Path,default=Path('runs/bricknet_06_fullg'))
    parser.add_argument('--m-only-run',type=Path,default=Path('runs/bricknet_06_reference_last'))
    parser.add_argument('--m-only-data',type=Path,default=Path('runs/reference_last_06'))
    args = parser.parse_args(argv)
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    if any(root.glob('*/metrics.jsonl')):raise RuntimeError('experiment already started')
    summary={}
    for mode in ['PT','SFT']:
        data=args.data/mode;data=data.resolve()
        ready=json.loads((data/'READY.json').read_text());assert ready['status']=='ready'
        index={name:[json.loads(s) for s in (data/name/'train.index.jsonl').open()] for name in ['v2','v3']}
        pools={}
        for name,rs in index.items():
            groups={}
            for i,r in enumerate(rs):groups.setdefault(r['source_family'],{}).setdefault(r['condition'],[]).append(i)
            pools[name]=groups
        rng=random.Random(42);names=['v2']*35000+['v3']*35000;rng.shuffle(names)
        sampled=[];counts=Counter();tokens=0;supervised=0
        for name in names:
            family=rng.choice(sorted(pools[name]));condition=rng.choice(sorted(pools[name][family]));i=rng.choice(pools[name][family][condition])
            r=index[name][i];sampled.append({'dataset':name,'row_index':i,'length':r['length']})
            counts[name]+=1;counts[f'{name}:{condition}']+=1;tokens+=r['length'];supervised+=r['target_length']
        batches=[]
        for start in range(0,len(sampled),256):
            block=sorted(sampled[start:start+256],key=lambda r:r['length'])
            mini=[block[i:i+4] for i in range(0,len(block),4)];rng.shuffle(mini);batches.extend(mini)
        schedule=root/f'{mode}.batches.jsonl';schedule.write_text(''.join(json.dumps(b)+'\n' for b in batches))
        dev=[]
        for name in ['v2','v3']:
            rs=[json.loads(s) for s in (data/name/'dev.index.jsonl').open()];groups={}
            for i,r in enumerate(rs):groups.setdefault(r['source_family'],[]).append(i)
            groups=list(groups.values());selector=random.Random(2026)
            for g in groups:selector.shuffle(g)
            selected=[]
            while len(selected)<min(128,len(rs)):
                for g in groups:
                    if g and len(selected)<128:selected.append(g.pop())
            dev.extend({'dataset':name,'row_index':i} for i in selected)
        if mode=='PT':
            dev=json.loads((args.m_only_run/'dev_selection.json').read_text())
            old_ready=json.loads((args.m_only_data/'READY.json').read_text())
            for name in ['v2','v3']:
                assert ready['artifact_sha256'][f'{name}/train.u32']==old_ready['artifact_sha256'][f'{name}/train.u32']
            assert schedule.read_bytes()==(args.m_only_run/'batches.jsonl').read_bytes()
        devpath=root/f'{mode}.dev.json';devpath.write_text(json.dumps(dev,indent=2)+'\n')
        cfg={'mode':mode,'size':'0.6B','experiment':'full_G_generation','data_dir':str(data),'output_dir':str((root/mode).resolve()),
             'schedule':str(schedule.resolve()),'schedule_sha256':hashlib.sha256(schedule.read_bytes()).hexdigest(),
             'dev_selection':str(devpath.resolve()),'ready_sha256':hashlib.sha256((data/'READY.json').read_bytes()).hexdigest(),
             'train_input_sha256':{n:ready['artifact_sha256'][f'{n}/train.u32'] for n in ['v2','v3']},
             'cache_sha256':{k:v for k,v in ready['artifact_sha256'].items() if any('/'+s+'.' in k for s in ['train','dev']) and (k.endswith('.u32') or k.endswith('.index.jsonl'))},
             'code_sha256':{n:hashlib.sha256((Path('src/brickicl/posttrain')/n).read_bytes()).hexdigest() for n in ['train_fullg06.py','bricknet_runtime.py','data.py']},
             'micro_batch':4,'gradient_accumulation':8,'draws':70000,'epochs':1,'lr':1e-4,'weight_decay':.01,
             'warmup_ratio':.03,'max_grad_norm':1.,'eval_steps':250,'save_steps':250,'seed':42,
             'gradient_checkpointing':True,'max_length':6400,
             'initial_adapters':['kulits/BrickNet-0.6B-PT']+(['kulits/BrickNet-0.6B-SFT'] if mode=='SFT' else []),
             'loss':'per-example mean CE over all G output tokens plus EOS, not just missing M; PT given a only; SFT caption fully masked',
             'caption':mode=='SFT','given_bricks':0,'dataset_mix':[.5,.5],
             'data_difference':'SFT restricted to author-caption-covered SFT pool; edits explicitly described in text; PT uses both source pools',
             'test_access':'test not loaded by trainer','status':'authorized_prepared'}
        (root/f'{mode}.json').write_text(json.dumps(cfg,indent=2)+'\n')
        summary[mode]={'counts':dict(counts),'all_tokens':tokens,'supervised_tokens':supervised,
                       'micro_batches':len(batches),'optimizer_steps':(len(batches)+7)//8,'dev_rows':len(dev)}
    (root/'submission_plan.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
