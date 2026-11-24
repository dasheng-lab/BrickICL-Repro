"""Freeze two matched 70k subsets of nzd v4; exclude complete held-out source sets."""
import fcntl,json,random,shutil,time,traceback
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from prepare import ROOT,sha,atomic

REPO=ROOT.parents[1];NZD=Path('/home/nzd0420/brickICL/runs/nzd-supertry')
BASE=REPO/'runs/qwen3_direct_pt_20261117';OLD=REPO/'runs/structural_20261114'
def source(r):return f'{r["split"]}:{r["source"]}'
def priority(text):return __import__('hashlib').sha256(('qwen3-nzd-20261119:'+text).encode()).hexdigest()

def main():
    lock=(ROOT/'data.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'DATA_READY.json').exists():print('Data already ready');return
    started=time.time();devs={a:[json.loads(s) for s in (NZD/f'dev_v4/{a}.jsonl').open()] for a in ('plain','masked')}
    assert len(devs['plain'])==len(devs['masked'])==1866
    dev_sources={source(r) for r in devs['plain']}
    held_struct=set();train_sources=set();blocked_families=set();known={}
    for v in ('v2','v3'):
        for split in ('train','dev','test'):
            for s in (OLD/f'cache/PT_G/{v}/{split}.index.jsonl').open():
                r=json.loads(s);known[r['source_key']]=r['source_family']
                if split!='train':held_struct.add(r['source_key']);blocked_families.add(r['source_family'])
    blocked=dev_sources|held_struct|{k for k,v in known.items() if v in blocked_families}
    rng=random.Random(20261119);representatives={};counts=Counter();excluded=Counter();source_hashes={a:__import__('hashlib').sha256() for a in ('plain','masked')}
    offsets=[0,0]
    with (NZD/'train_v4_mix/plain.jsonl').open('rb') as pfile,(NZD/'train_v4_mix/masked.jsonl').open('rb') as mfile:
        for i,(pb,mb) in enumerate(zip_longest(pfile,mfile)):
            assert pb is not None and mb is not None
            source_hashes['plain'].update(pb);source_hashes['masked'].update(mb)
            p=json.loads(pb);m=json.loads(mb)
            assert all(p.get(k)==m.get(k) for k in ('source','split','sample_index','nodes','origin')),(i,'pair mismatch')
            if 'order' in p:assert sorted(p['order'])==sorted(m['order'])
            key=source(p);origin=p.get('origin','generated_pair')
            if origin=='shipped':assert p['path']==m['path']
            counts['raw_rows']+=1
            if key in blocked:excluded[origin]+=1
            else:
                bucket=(origin,p['split']);counts['eligible:'+':'.join(bucket)]+=1
                ident=(origin,key);rank=rng.random()
                record={'line':i,'plain_offset':offsets[0],'masked_offset':offsets[1],'source_key':key,
                        'origin':origin,'split':p['split'],'nodes':p['nodes'],'priority':rank}
                if ident not in representatives or rank<representatives[ident]['priority']:representatives[ident]=record
            offsets[0]+=len(pb);offsets[1]+=len(mb)
            if (i+1)%100000==0:
                atomic(ROOT/'data_status.json',{'state':'scanning','rows':i+1,'excluded_rows':sum(excluded.values())});print('SCAN',i+1,flush=True)
    assert counts['raw_rows']==469662
    pair_total=sum(v for k,v in counts.items() if k.startswith('eligible:generated_pair:'))
    n_sft=round(46667*counts['eligible:generated_pair:sft']/pair_total);n_pt=46667-n_sft
    chosen=[]
    for pool,n in [('sft',n_sft),('pt',n_pt)]:
        candidates=[r for r in representatives.values() if r['origin']=='generated_pair' and r['split']==pool]
        candidates.sort(key=lambda r:priority('pair:'+r['source_key']));assert len(candidates)>=n
        chosen.extend(candidates[:n])
    used={r['source_key'] for r in chosen}
    replay=[r for r in representatives.values() if r['origin']=='shipped' and r['source_key'] not in used]
    replay.sort(key=lambda r:priority('replay:'+r['source_key']));assert len(replay)>=23333
    chosen.extend(replay[:23333]);random.Random(0).shuffle(chosen)
    assert len(chosen)==len({r['source_key'] for r in chosen})==70000
    assert not {r['source_key'] for r in chosen}&blocked
    (ROOT/'selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    # One fixed validation representative per source. The full 541-source pool is excluded from training.
    by_source={}
    for i,(p,m) in enumerate(zip(devs['plain'],devs['masked'])):
        assert source(p)==source(m) and p['nodes']==m['nodes']
        key=source(p)
        if key not in by_source or priority(f'dev-row:{i}')<priority(f'dev-row:{by_source[key]}'):by_source[key]=i
    dev_indices=[by_source[k] for k in sorted(by_source,key=lambda s:priority('dev:'+s))[:256]]
    assert len(dev_indices)==256;random.Random(0).shuffle(dev_indices)
    base_prov=json.loads((BASE/'BASE_PROVENANCE.json').read_text());shutil.copy2(BASE/'BASE_PROVENANCE.json',ROOT/'BASE_PROVENANCE.json')
    tok=AutoTokenizer.from_pretrained(base_prov['model_directory'],local_files_only=True)
    indices={};source_streams={a:(NZD/f'train_v4_mix/{a}.jsonl').open('rb') for a in ('plain','masked')}
    try:
        for arm in ('plain','masked'):
            for split in ('train','dev'):
                refs=chosen if split=='train' else dev_indices
                for v,items in [('v2',refs[:len(refs)//2]),('v3',refs[len(refs)//2:])]:
                    dest=ROOT/f'cache/{arm}/{v}';dest.mkdir(parents=True,exist_ok=True);offset=0;records=[]
                    with (dest/f'{split}.u32').open('wb') as binary,(dest/f'{split}.index.jsonl').open('w') as index:
                        for start in range(0,len(items),128):
                            block=items[start:start+128];rows=[]
                            for ref in block:
                                if split=='train':
                                    stream=source_streams[arm];stream.seek(ref[arm+'_offset']);r=json.loads(stream.readline())
                                    assert source(r)==ref['source_key']
                                else:r=devs[arm][ref]
                                assert r['nodes']<=100
                                rows.append(r)
                            texts=[r['path'].strip(' ') for r in rows]
                            encoded=tok(texts,add_special_tokens=False)['input_ids']
                            for j,(r,ids) in enumerate(zip(rows,encoded)):
                                assert ids[0]==tok.encode('a',add_special_tokens=False)[0]
                                ids=ids+[tok.eos_token_id];assert len(ids)<=6400
                                meta={'id':f'{split}-{v}-{start+j}','source_key':source(r),'origin':r.get('origin','generated_pair'),
                                      'nodes':r['nodes'],'offset':offset,'length':len(ids),'prefix_length':1,'target_length':len(ids)-1}
                                np.asarray(ids,dtype='<u4').tofile(binary);index.write(json.dumps(meta)+'\n');records.append(meta);offset+=len(ids)
                        indices[arm,v,split]=records
                    print('TOKENIZED',arm,v,split,len(records),flush=True)
    finally:
        for stream in source_streams.values():stream.close()
    refs=[{'dataset':v,'row_index':i} for v in ('v2','v3') for i in range(35000)]
    lengths=[max(indices[a,r['dataset'],'train'][r['row_index']]['length'] for a in ('plain','masked')) for r in refs]
    order=list(LengthGroupedSampler(batch_size=4,lengths=lengths,generator=torch.Generator().manual_seed(0)))
    assert set(order)==set(range(70000))
    batches=[[refs[k] for k in order[i:i+4]] for i in range(0,70000,4)]
    schedule=ROOT/'schedule.jsonl';schedule.write_text(''.join(json.dumps(b)+'\n' for b in batches))
    devfile=ROOT/'dev_selection.json';atomic(devfile,[{'dataset':v,'row_index':i} for v in ('v2','v3') for i in range(128)])
    workload={}
    for arm in ('plain','masked'):
        cache=ROOT/'cache'/arm
        artifact={str(p.relative_to(cache)):sha(p) for v in ('v2','v3') for p in sorted((cache/v).iterdir())}
        atomic(cache/'READY.json',{'status':'ready','cross_split_overlap':0,'artifact_sha256':artifact,
            'split_check':'source keys; all 541 nzd dev sources and structural dev/test known families excluded',
            'cache_shards':'v2/v3 are storage partitions, not structural-v2/v3 dataset versions'})
        cfg=json.loads((BASE/'training/Full_G/config.json').read_text())
        cfg.update(arm=arm,output_dir=str(ROOT/'training'/arm),data_dir=str(cache),schedule=str(schedule),schedule_sha256=sha(schedule),
                   ready_sha256=sha(cache/'READY.json'),train_input_sha256={v:sha(cache/v/'train.u32') for v in ('v2','v3')},
                   cache_sha256=artifact,code_sha256={n:sha(ROOT/n) for n in ('train.py','bricknet_runtime.py','data.py')},
                   dev_selection=str(devfile),order='DFS' if arm=='plain' else 'same tree, missing subtree moved last',
                   sampling='source-diverse paired nzd v4 subset; 46667 generated pairs + 23333 identical official rows; shared row order',
                   sampler_lengths='max paired plain/masked token length',official_replay_mix=True,
                   experiment='Raw Qwen3 nzd control/augmentation, full-sequence CE; no initial BrickNet adapter')
        atomic(ROOT/'training'/arm/'config.json',cfg)
        padded=sum(max(indices[arm,r['dataset'],'train'][r['row_index']]['length'] for r in b)*len(b) for b in batches)
        workload[arm]={'padded_input_tokens':padded,'input_tokens':sum(r['length'] for v in ('v2','v3') for r in indices[arm,v,'train']),
                       'supervised_tokens':sum(r['target_length'] for v in ('v2','v3') for r in indices[arm,v,'train'])}
    old_indices={v:[json.loads(s) for s in (OLD/f'cache/PT_G/{v}/train.index.jsonl').open()] for v in ('v2','v3')}
    old_batches=[json.loads(s) for s in (OLD/'PT_schedule.jsonl').open()]
    old_padded=sum(max(old_indices[r['dataset']][r['row_index']]['length'] for r in b)*len(b) for b in old_batches)
    report={'state':'ready','rows_per_arm':70000,'steps_per_arm':2188,'unique_train_sources':70000,'validation_pairs':256,
            'nzd_dev_sources_excluded':len(dev_sources),'blocked_sources':len(blocked),'excluded_rows':dict(excluded),
            'eligible_counts':dict(counts),'selection_counts':dict(Counter(r['origin']+':'+r['split'] for r in chosen)),
            'source_sha256':{a:h.hexdigest() for a,h in source_hashes.items()},'dev_source_sha256':{a:sha(NZD/f'dev_v4/{a}.jsonl') for a in ('plain','masked')},
            'train_dev_source_overlap':0,'selection_sha256':sha(ROOT/'selection.jsonl'),
            'workload':workload,'fullg_reference_padded_tokens':old_padded,'elapsed_seconds':time.time()-started}
    atomic(ROOT/'DATA_READY.json',report);atomic(ROOT/'data_status.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'DATA_FAILED.json',{'traceback':traceback.format_exc()});raise
