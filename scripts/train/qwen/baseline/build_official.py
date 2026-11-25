"""Uniform reservoir over official first-round PT+SFT paths; cluster CPU only."""
import fcntl,io,json,os,random,re,subprocess,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from prepare import ROOT,OLD,sha,atomic,config

DATA=ROOT.parents[3]/'data/bricknet/dataset'

def main():
    lock=(ROOT/'official_build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'OFFICIAL_READY.json').exists():return
    excluded=set()
    for v in ('v2','v3'):
        for split in ('dev','test'):
            for line in (OLD/f'cache/PT_G/{v}/{split}.index.jsonl').open():excluded.add(json.loads(line)['source_key'])
    selected=ROOT/'official_selected.jsonl';counts=Counter();first_round={};start=time.time()
    if not selected.exists():
        rng=random.Random(20261117);reservoir=[];seen=0
        for pool in ('pt','sft'):
            source=DATA/f'paths_{pool}.jsonl.xz';tag=None
            proc=subprocess.Popen(['xz','-dc','-T2',str(source)],stdout=subprocess.PIPE)
            with io.TextIOWrapper(proc.stdout,encoding='utf-8') as stream:
                for index,line in enumerate(stream):
                    if tag is None:
                        head=json.loads(line);first_round[pool]=head['round'];tag=f'"round": {json.dumps(head["round"])}'
                    counts[pool+'_lines']+=1
                    if tag not in line:continue
                    row=json.loads(line)
                    if row['round']!=first_round[pool]:continue
                    counts[pool+'_first_round']+=1
                    key=f'{pool}:{row["source"]}'
                    if key in excluded:counts[pool+'_excluded_holdout']+=1;continue
                    seen+=1;counts[pool+'_eligible']+=1
                    slot=len(reservoir) if len(reservoir)<70000 else rng.randrange(seen)
                    if slot<70000:
                        path=row['path'].strip(' ');lines=path.rstrip('\n').split('\n')
                        clipped=len(lines)>199
                        if clipped:path='\n'.join(lines[:199])+'\n'
                        rec={'id':f'official-{pool}-{index}','source_key':key,'path':path,'round':row['round'],
                             'sample_index':row.get('sample_index'),'source_line':index,'cropped_to_100':clipped}
                        if slot==len(reservoir):reservoir.append(rec)
                        else:reservoir[slot]=rec
                    if counts[pool+'_first_round']%100000==0:
                        atomic(ROOT/'official_build_status.json',{'state':'sampling','pool':pool,'eligible_rows':seen,'counts':dict(counts),'elapsed_seconds':time.time()-start})
                        print('SAMPLING',pool,dict(counts),flush=True)
            if proc.wait()!=0:raise RuntimeError(f'xz failed: {source}')
        assert len(reservoir)==70000 and all(r['source_key'] not in excluded for r in reservoir)
        tmp=selected.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in reservoir));tmp.replace(selected)
        atomic(ROOT/'OFFICIAL_SAMPLING.json',{'method':'uniform reservoir over eligible first-round rows, official PT+SFT files','seed':20261117,
            'counts':dict(counts),'first_round':first_round,'source_file_stat':{p:{'path':str(DATA/f'paths_{p}.jsonl.xz'),'bytes':(DATA/f'paths_{p}.jsonl.xz').stat().st_size} for p in ('pt','sft')},
            'selected_pool_counts':dict(Counter(r['source_key'].split(':')[0] for r in reservoir)),
            'selected_rows':70000,'cropped_to_100':sum(r['cropped_to_100'] for r in reservoir),
            'excluded_source_keys':sorted(excluded),'selected_sha256':sha(selected),'elapsed_seconds':time.time()-start})
    rows=[json.loads(s) for s in selected.open()];assert len(rows)==70000
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    cache=ROOT/'official_cache';refs=[];lengths=[]
    for v,subset in [('v2',rows[:35000]),('v3',rows[35000:])]:
        dest=cache/v;dest.mkdir(parents=True,exist_ok=True);offset=0
        with (dest/'train.u32').open('wb') as binary,(dest/'train.index.jsonl').open('w') as idx:
            for begin in range(0,len(subset),128):
                chunk=subset[begin:begin+128];encoded=tok([r['path'] for r in chunk],add_special_tokens=False)['input_ids']
                for j,(r,ids) in enumerate(zip(chunk,encoded)):
                    assert ids[0]==tok.encode('a',add_special_tokens=False)[0]
                    ids=ids+[tok.eos_token_id]
                    if len(ids)>6400:raise ValueError(f'Official path exceeds 6400 tokens; no silent token truncation: {r["id"]}')
                    np.asarray(ids,dtype='<u4').tofile(binary)
                    meta={k:value for k,value in r.items() if k!='path'}
                    meta.update(offset=offset,length=len(ids),prefix_length=1,target_length=len(ids)-1)
                    idx.write(json.dumps(meta)+'\n');offset+=len(ids)
                    refs.append({'dataset':v,'row_index':begin+j});lengths.append(len(ids))
        # Common structural holdouts keep the full-generation loss directly comparable.
        for split in ('dev','test'):
            for ext in ('u32','index.jsonl'):
                target=dest/f'{split}.{ext}'
                if not target.exists():target.symlink_to(OLD/f'cache/PT_G/{v}/{split}.{ext}')
        print('TOKENIZED',v,offset,'tokens',flush=True)
    artifacts={str(f.relative_to(cache)):sha(f) for v in ('v2','v3') for f in sorted((cache/v).iterdir())}
    atomic(cache/'READY.json',{'status':'ready','cross_split_overlap':0,'artifact_sha256':artifacts,
        'sampling_manifest':str(ROOT/'OFFICIAL_SAMPLING.json'),'v2_v3_note':'arbitrary equal cache shards for official rows, not structural v2/v3 categories'})
    order=list(LengthGroupedSampler(batch_size=4,lengths=lengths,generator=torch.Generator().manual_seed(0)))
    assert len(set(order))==70000
    schedule=ROOT/'official_schedule.jsonl';schedule.write_text(''.join(json.dumps([refs[k] for k in order[i:i+4]])+'\n' for i in range(0,len(order),4)))
    cfg=config('BrickNet',cache,schedule,OLD/'PT_dev_selection.json',OLD/'training/PT_G/config.json')
    cfg.update(order='official shipped DFS, first round; optional prefix crop to 100 parts',sampling='uniform without replacement over first-round official PT+SFT paths; seed 20261117',
               sampler_lengths='official tokenized path lengths',data_shards='v2/v3 are arbitrary 35k/35k cache shards')
    atomic(ROOT/'training/BrickNet/config.json',cfg)
    atomic(ROOT/'OFFICIAL_READY.json',{'rows':70000,'token_count':sum(lengths),'max_tokens':max(lengths),'elapsed_seconds':time.time()-start})
    print('OFFICIAL_READY',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'OFFICIAL_FAILED.json',{'traceback':traceback.format_exc()});raise
