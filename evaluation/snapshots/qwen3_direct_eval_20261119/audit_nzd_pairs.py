"""Read both nzd v4 arms, hash the exact sources and check row pairing."""
import hashlib,json,time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

ROOT=Path('/home/nzd0420/brickICL/runs/nzd-supertry')
OUT=Path('/home/sunshq114514/brickICL/runs/qwen3_direct_eval_20261119/NZD_DATA_AUDIT.json')
report={};started=time.time();source_sets={}
for pool in ('train_v4_mix','dev_v4'):
    paths={a:ROOT/pool/(a+'.jsonl') for a in ('plain','masked')}
    digests={a:hashlib.sha256() for a in paths};counts=Counter();sources=set();examples={};rounds=Counter()
    with paths['plain'].open('rb') as pf,paths['masked'].open('rb') as mf:
        for i,(pb,mb) in enumerate(zip_longest(pf,mf)):
            assert pb is not None and mb is not None,(pool,i,'unequal lines')
            digests['plain'].update(pb);digests['masked'].update(mb)
            p=json.loads(pb);m=json.loads(mb)
            for field in ('source','split','sample_index','nodes'):
                assert p.get(field)==m.get(field),(pool,i,field)
            assert p.get('origin')==m.get('origin'),(pool,i,'origin')
            if 'order' in p and 'order' in m:assert sorted(p['order'])==sorted(m['order']),(pool,i,'part set')
            origin=p.get('origin','generated_pair');counts[origin]+=1;rounds['plain:'+str(p.get('round'))]+=1;rounds['masked:'+str(m.get('round'))]+=1
            sources.add((p.get('split'),str(p['source'])))
            if origin=='shipped':assert p['path']==m['path'],(pool,i,'shared replay differs')
            if p['path']==m['path']:counts['identical_paths']+=1
            counts['rows']+=1
            examples.setdefault(origin,{'row_index':i,'source':p['source'],'split':p.get('split'),'nodes':p['nodes'],
                                        'keys':list(p),'k':p.get('k'),'jump':m.get('jump')})
            if (i+1)%100000==0:print(pool,i+1,flush=True)
    source_sets[pool]=sources
    report[pool]={'files':{a:{'path':str(path),'sha256':digests[a].hexdigest(),'bytes':path.stat().st_size} for a,path in paths.items()},
                  'counts':dict(counts),'source_count':len(sources),'rounds':dict(rounds),'examples':examples,'row_pairing_verified':True}
overlap=source_sets['train_v4_mix']&source_sets['dev_v4']
report['source_overlap_count']=len(overlap);report['source_overlap_examples']=sorted(overlap,key=str)[:20]
report['elapsed_seconds']=time.time()-started
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
