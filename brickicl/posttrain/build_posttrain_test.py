"""Freeze evaluation inputs before viewing outputs: heldout triads plus historical regression."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,score_text
from transformers import AutoTokenizer
from brickicl.posttrain.reference_last import connected_subsets
from brickicl.sampling.masked_walk import masked_walk
from brickicl.sampling.edits import stack_joint,stretch
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/posttrain_eval06'))
    args = parser.parse_args(argv)
    OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)

    def anchor(g,c):
        c=set(c);outside=set()
        for e in g.edges:
            a,b=int(e['a']),int(e['b'])
            if a in c and b not in c:outside.add(b)
            if b in c and a not in c:outside.add(a)
        assert len(outside)==1
        return next(iter(outside))

    def make_triads(row,dataset):
        parsed=parse_sample(row['prompt']+row['completion']);assert parsed.error is None
        g=tree_to_graph(parsed.tree)
        targets=connected_subsets(row['target_positions'],g.edges)
        refs=connected_subsets(row['evidence_positions'],g.edges)
        # Keep one target copy for an interpretable local completion probe.
        if len(targets)!=1:return None,'multiple_targets'
        copies=refs+targets;joints=[stack_joint(g,c,anchor(g,c)) for c in copies]
        if not all(joints):return None,'no_stack_joint'
        r=2 # predeclared small intervention; not selected using model results
        target=targets[0];reference=refs[0];nref=len(refs)
        gt,(added_t,)=stretch(g,[joints[-1]],r)
        go,added_o=stretch(g,joints[:-1],r)
        gall,added_all=stretch(g,joints,r)
        variants={
          'original':(g,gt,target+added_t,[],reference),
          'stretched':(go,gall,target+added_all[-1],[],reference+added_o[0]),
          'ablated':(g,gt,target+added_t,[v for c in refs for v in c],[])}
        result=[]
        for condition,(shortg,longg,longm,exclude,late) in variants.items():
            kwargs={'exclude':exclude,'seed':row['sampling_seed'],'late':[late] if late else ()}
            sh=masked_walk(shortg,target,**kwargs);lo=masked_walk(longg,longm,**kwargs)
            assert sh.prefix==lo.prefix
            if any(score_text(x.prefix+x.target)[2] for x in [sh,lo]):return None,'reference_collision'
            expected='long' if condition=='stretched' else 'short'
            demo=sh.prefix+(lo.target if expected=='long' else sh.target)
            if len(tok.encode(demo+tok.eos_token+sh.prefix+lo.target,add_special_tokens=False))>6400:
                return None,'double_prompt_over_limit'
            result.append({'suite':'heldout','id':f'{row["source_key"]}:{condition}',
                 'source_key':row['source_key'],'source_family':row['source_family'],'source_dataset':dataset,
                 'condition':condition,'expected':expected,'r':r,'prime':0,
                 'prefix':sh.prefix,'target_short':sh.target,'target_long':lo.target,'demo':demo,
                 'n_prefix':sh.walk.n_prefix,'k_short':len(target),'k_long':len(longm),
                 'seed':int(hashlib.sha256(row['source_key'].encode()).hexdigest()[:8],16)})
        return result,None

    candidates={}
    for dataset in ['v3','v2']:
        path=Path('runs/reference_last_06')/dataset/'test.jsonl'
        for line in path.open():
            r=json.loads(line)
            if r['condition']=='original' and r.get('palette_id',0)==0 and r.get('r',0)==0:
                candidates.setdefault(r['source_key'],(r,dataset))
    rows=[];skipped=Counter()
    for source,(r,dataset) in sorted(candidates.items()):
        try:triads,reason=make_triads(r,dataset)
        except (ValueError,AssertionError,KeyError,IndexError) as e:triads,reason=None,type(e).__name__
        if triads:rows+=triads
        else:skipped[reason]+=1
    with (OUT/'heldout_triads.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    sources=sorted({r['source_key'] for r in rows},key=lambda s:hashlib.sha256(('viewer:'+s).encode()).digest())
    assert len(sources)>=4,('need four sources',len(sources),skipped)
    (OUT/'viewer_selection.json').write_text(json.dumps({'sources':sources[:4],
        'selection':'fixed SHA256 ordering of eligible heldout sources before model outputs; generation sample 0'},indent=2)+'\n')

    legacy=[]
    for line in Path('runs/twocopy/twocopy.matched.r4.p1.jsonl').open():
        r=json.loads(line)
        legacy.append({'suite':'legacy','id':f'val:{r["source"]}:{r["condition"]}','source_key':f'val:{r["source"]}',
           'source_family':f'val:{r["source"]}','condition':r['condition'],'expected':r['expected'],'r':r['r'],'prime':1,
           'prefix':r['query_prefix'],'target_short':r['target_short'],'target_long':r['target_long'],
           'demo':r['demo_text'],'n_prefix':r['n_prefix'],'k_short':r['k_short'],'k_long':r['k_long'],
           'seed':r['source']})
    with (OUT/'legacy_triads.jsonl').open('w') as f:
        for r in legacy:f.write(json.dumps(r)+'\n')
    info={'heldout_sources':len(sources),'heldout_rows':len(rows),'skipped':dict(skipped),'viewer_sources':sources[:4],
          'legacy_rows':len(legacy),'heldout_prime':0,'legacy_prime':1,'heldout_r':2,'legacy_r':4,
          'ablated_semantics':'all other repeat evidence removed; original short reference is a diagnostic convention, not uniquely implied',
          'no_test_selection_by_outputs':True}
    (OUT/'protocol.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2),flush=True)

if __name__ == '__main__':
    main()
