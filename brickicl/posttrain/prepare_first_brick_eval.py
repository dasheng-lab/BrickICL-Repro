"""Build matched first-brick/no-first-brick tests without requiring prior evaluation outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from bricknet import parse_sample, score_text
from transformers import AutoTokenizer
from brickicl.posttrain.paths import ARMS, experiment_root, expected_adapters

DEFAULT_SOURCES=(15,48,57,58,85,95,105,113,114,119,175,183,252,258,290,433,458,475)
CONDITIONS=('original','stretched','ablated')

def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]

def paired_inputs(raw_rows,sources=DEFAULT_SOURCES):
    """Move the identical first target action into the prompt; never alter full references."""
    selected={}
    for row in raw_rows:
        if row['source'] in sources and row['pair']==f"{row['source']}-0-0":
            key=(row['source'],row['condition'])
            if key in selected:raise ValueError(f'duplicate source/condition: {key}')
            selected[key]=row
    result={0:[],1:[]}
    for source in sorted(sources):
        for condition in CONDITIONS:
            r=selected[source,condition]
            if r['r']!=4:raise ValueError('This fixed protocol requires r=4')
            head=''.join(r['target_short'].splitlines(True)[:2])
            if not head or not r['target_long'].startswith(head):raise ValueError('Long/short target head differs')
            expected='long' if condition=='stretched' else 'short'
            common={'id':f'val:{source}:{condition}','source_key':f'val:{source}','source_family':f'val:{source}',
                    'condition':condition,'expected':expected,'r':4,'seed':source,
                    'demo':r['prefix']+r['target_'+expected]}
            for prime in (0,1):
                row=common|{'suite':f'p{prime}','prime':prime,'prefix':r['prefix']+(head if prime else ''),
                    'n_prefix':r['n_prefix']+prime,'target_short':r['target_short'][len(head):] if prime else r['target_short'],
                    'target_long':r['target_long'][len(head):] if prime else r['target_long'],
                    'k_short':r['k_short']-prime,'k_long':r['k_long']-prime}
                parsed=parse_sample(row['prefix'])
                assert parsed.error is None and len(parsed.tree.parts)==row['n_prefix']
                assert row['k_short']>0 and row['k_long']>0
                for kind in ('short','long'):
                    full=row['prefix']+row['target_'+kind]
                    assert full==r['prefix']+r['target_'+kind] and parse_sample(full).error is None
                row['prefix_sha256']=hashlib.sha256(row['prefix'].encode()).hexdigest()
                result[prime].append(row)
    return result

def generation_plan(primed_rows,tokenizer):
    """Reproduce the old p1 grouping and seeds, and use that identical plan for p0."""
    groups={}
    for r in primed_rows:groups.setdefault(64*(r['k_long']+4)+32,[]).append(r)
    plan=[]
    for cap,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:len(tokenizer.encode(r['prefix'],add_special_tokens=False)))
        for start in range(0,len(rows),8):
            ids=[r['id'] for r in rows[start:start+8]]
            seed=1729+int(hashlib.sha256('|'.join(ids).encode()).hexdigest()[:8],16)%10000000
            plan.append({'seed':seed,'max_new_tokens':cap,'ids':ids})
    return plan

def reusable(cache_inputs,cache_outputs,inputs,plan,adapters):
    """Fail closed if input text, weights, draw plan or sample completeness differs."""
    byid={r['id']:r for r in inputs};old_inputs={r['id']:r for r in cache_inputs}
    if len(cache_outputs)!=len(byid) or len({r['id'] for r in cache_outputs})!=len(byid):return False
    if set(old_inputs)!=set(byid) or {r['id'] for r in cache_outputs}!=set(byid):return False
    for key,r in byid.items():
        if any(old_inputs[key].get(k)!=r[k] for k in ['prefix','target_short','target_long','n_prefix','k_short','k_long','prime','r']):return False
    planned={key:g for g in plan for key in g['ids']}
    return all(r.get('adapters')==adapters and len(r.get('samples',[]))==4 and r.get('prime')==1
               and r['seed']==planned[r['id']]['seed'] and r['max_new_tokens']==planned[r['id']]['max_new_tokens']
               for r in cache_outputs)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/stretch/stretch.contiguous.r4.jsonl'))
    p.add_argument('--out',type=Path,default=Path('runs/history_first_brick06'))
    p.add_argument('--reuse-existing',action='store_true',help='Reuse p1 only after exact text/weights/seed/cap checks')
    p.add_argument('--legacy-eval',type=Path)
    p.add_argument('--fullg-eval',type=Path)
    args=p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):raise RuntimeError('Use a new output directory; preparation never overwrites a run')
    rows=paired_inputs(read_rows(args.input))
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    plan=generation_plan(rows[1],tok)
    args.out.mkdir(parents=True,exist_ok=True)
    def save(name,value):
        (args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for prime,values in rows.items():
        for r in values:r['prefix_collisions']=score_text(r['prefix'])[2]
        (args.out/f'p{prime}_triads.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values),encoding='utf-8')
    save('generation_plan.json',plan)
    legacy=args.legacy_eval or experiment_root()/'runs/posttrain_eval06'
    fullg=args.fullg_eval or experiment_root()/'runs/fullg_main_eval06'
    reuse={}
    for arm in ARMS:
        adapters=expected_adapters(arm);folder=fullg if arm.startswith('fullg') else legacy
        path=folder/f'legacy.{arm}.jsonl';input_path=folder/'legacy_triads.jsonl'
        reused=False
        if args.reuse_existing and path.exists() and input_path.exists():
            cached=read_rows(path)
            reused=reusable(read_rows(input_path),cached,rows[1],plan,adapters)
            if reused:(args.out/f'p1.{arm}.jsonl').write_text(''.join(json.dumps(r|{'suite':'p1','reused_from':str(path)})+'\n' for r in cached))
        reuse[arm]={'reused':reused,'path':str(path) if reused else None,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if reused else None}
        save(f'provenance.{arm}.json',{'arm':arm,'adapters':adapters,'samples':4,'temperature':1.,'top_k':20,'top_p':.95,
              'selection':'dev-best checkpoints only; test not used for selection','p1_reuse':reuse[arm]})
    # Captions are display-only; never part of the six model prompts.
    import lzma,os
    data=Path(os.environ.get('BRICKNET_DATA',str(experiment_root()/'baseline/bricknet/data')))
    with lzma.open(data/'dataset/captions_val.jsonl.xz','rt') as stream:captions={int(r['id']):r['caption'] for r in map(json.loads,stream)}
    save('sources.json',[{'source':s,'key':f'val:{s}','caption':captions[s],'label':f'#{s}'} for s in DEFAULT_SOURCES])
    save('protocol.json',{'sources':list(DEFAULT_SOURCES),'source_count':18,'conditions':list(CONDITIONS),'r':4,
         'prime_settings':[0,1],'arms':list(ARMS),'samples_per_input':4,'temperature':1.,'top_k':20,'top_p':.95,
         'provided_caption':False,'provided_full_demo':False,'p1_reuse':reuse,
         'checkpoint_selection':'dev-best, not selected using test scores',
         'token_cap':'same absolute p1-derived cap in p0 and p1','seed_and_batch_plan':'same batch order, composition and seeds',
         'reference_identity':'p0 prefix + target == p1 prefix + remaining target',
         'input_sha256':{f'p{i}':hashlib.sha256((args.out/f'p{i}_triads.jsonl').read_bytes()).hexdigest() for i in (0,1)},
         'limits':'Previously explored cases, not fresh generalization. Ablated short reference is diagnostic.'})
    print(json.dumps({'sources':18,'cases_per_prime':54,'reused_p1_models':sum(x['reused'] for x in reuse.values()),'batch_groups':len(plan)}))

if __name__=='__main__':main()
