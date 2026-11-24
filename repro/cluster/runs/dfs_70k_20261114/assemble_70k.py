"""Build exactly one epoch of distinct v2/v3 records, with fewer v2 and captioned SFT expansion."""
import hashlib,json,shutil
from collections import Counter,defaultdict,deque
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from caption_rules import ROOT,load_bases,caption_for

OLD=Path('runs/dfs_posttrain_20261114')
CURATED=Path('runs/dfs_curated_20261114')
POOLS={'v2':OLD/'corpus/v2/train.jsonl','v3_old':OLD/'corpus/v3/train.jsonl','v3_extra':ROOT/'extra_v3.jsonl'}
QUOTAS={'PT':{'v2':20000,'v3':50000},'SFT':{'v2':10000,'v3':60000}}
def hash_text(s):return hashlib.sha256(s.encode()).hexdigest()
def sha_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (ROOT/'DATA_READY.json').exists():return
    bases=load_bases();metadata={m:{v:[] for v in ('v2','v3')} for m in QUOTAS}
    for pool,path in POOLS.items():
        version='v2' if pool=='v2' else 'v3'
        for i,line in enumerate(path.open()):
            row=json.loads(line);cap,quality=caption_for(row,bases)
            assert row['split']=='train' and row['selected_parts']<=100
            for mode in QUOTAS:
                if (mode=='PT' and pool=='v3_extra') or (mode=='SFT' and not cap):continue
                head=cap+'\n' if mode=='SFT' else ''
                metadata[mode][version].append({'pool':pool,'i':i,'id':row['id'],'family':row['source_family'],
                    'source_key':row['source_key'],'condition':row['condition'],'r':row['r'],'palette':row['palette_id'],
                    'target':tuple(row['target_copy_indices']),'parts':row['selected_parts'],
                    'full_key':hash_text(head+row['full_dfs_path']),
                    'masked_key':hash_text(head+row['prompt']+row['completion']+str(row['n_prefix'])),
                    'rank':hash_text('70k:'+row['id']),'caption_quality':quality})
    chosen={};report={}
    old_v3_sources={r['source_key'] for r in metadata['PT']['v3'] if r['pool']=='v3_old'}
    old_v3_families={r['family'] for r in metadata['PT']['v3'] if r['pool']=='v3_old'}
    for mode,quotas in QUOTAS.items():
        blocked_g=set();blocked_m=set();report[mode]={}
        for version,quota in quotas.items():
            groups=defaultdict(lambda:defaultdict(list))
            for r in metadata[mode][version]:
                groups[r['family']][(r['condition'],r['r'],r['palette'],r['target'],r['parts'])].append(r)
            queues={}
            for family,variants in groups.items():
                buckets=[deque(sorted(rs,key=lambda r:r['rank'])) for key,rs in sorted(variants.items())]
                queue=[]
                while any(buckets):
                    for b in buckets:
                        if b:queue.append(b.popleft())
                queues[family]=deque(queue)
            family_order=sorted(queues,key=lambda f:hash_text('family:'+f))
            selected=[];duplicates=0
            while len(selected)<quota:
                before=len(selected)
                for family in family_order:
                    queue=queues[family]
                    while queue:
                        r=queue.popleft()
                        if r['full_key'] in blocked_g or r['masked_key'] in blocked_m:
                            duplicates+=1;continue
                        selected.append(r);blocked_g.add(r['full_key']);blocked_m.add(r['masked_key'])
                        break
                    if len(selected)==quota:break
                if len(selected)==before:break
            assert len(selected)==quota,(mode,version,'unique eligible rows',len(selected),'need',quota)
            chosen[mode,version]=selected
            report[mode][version]={'rows':len(selected),'candidate_rows':len(metadata[mode][version]),
                'families':len({r['family'] for r in selected}),'sources':len({r['source_key'] for r in selected}),
                'new_official_paths':sum(r['pool']=='v3_extra' for r in selected),
                'sources_new_to_old_v3_pool':len({r['source_key'] for r in selected}-old_v3_sources) if version=='v3' else None,
                'families_new_to_old_v3_pool':len({r['family'] for r in selected}-old_v3_families) if version=='v3' else None,
                'duplicate_serializations_skipped':duplicates}
    lookup=defaultdict(list)
    for (mode,version),rs in chosen.items():
        for r in rs:lookup[r['pool'],r['i']].append((mode,version,r))
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    handles={};offsets=Counter();counts=Counter();supervised=Counter();families=defaultdict(set);examples=[]
    for mode in QUOTAS:
        for scope in ('M','G'):
            for version in ('v2','v3'):
                d=ROOT/f'cache/{mode}_{scope}/{version}';d.mkdir(exist_ok=True,parents=True)
                handles[mode,scope,version]=((d/'train.u32').open('wb'),(d/'train.index.jsonl').open('w'))
    captions={v:(ROOT/f'captions_{v}_train.jsonl').open('w') for v in ('v2','v3')}
    manifest=(ROOT/'selection_manifest.jsonl').open('w')
    try:
        for pool,path in POOLS.items():
            for i,line in enumerate(path.open()):
                if (pool,i) not in lookup:continue
                row=json.loads(line);cap,quality=caption_for(row,bases)
                prefix=tok.encode(row['prompt'],add_special_tokens=False)
                target=tok.encode(row['completion'],add_special_tokens=False)
                whole=tok.encode(row['full_dfs_path'],add_special_tokens=False)
                assert tok.encode(row['prompt']+row['completion'],add_special_tokens=False)==prefix+target
                for mode,version,meta in lookup[pool,i]:
                    head=tok.encode(cap,add_special_tokens=False)+tok.encode('\n',add_special_tokens=False) if mode=='SFT' else []
                    if mode=='SFT':
                        captions[version].write(json.dumps({'id':row['id'],'source_key':row['source_key'],'caption':cap,'quality':quality,
                                                          'pool':pool,'pool_row_index':i})+'\n')
                        if len(examples)<12:examples.append({'id':row['id'],'source_key':row['source_key'],'version':version,'caption':cap,'quality':quality})
                    for scope in ('M','G'):
                        ids=head+(prefix+target if scope=='M' else whole)+[tok.eos_token_id]
                        start=len(head)+len(prefix) if scope=='M' else (len(head) if mode=='SFT' else 1)
                        assert len(ids)<=6400 and ids[-1]==tok.eos_token_id and 0<start<len(ids)
                        assert tok.decode(ids[start:-1])==(row['completion'] if scope=='M' else row['full_dfs_path'][1:] if mode=='PT' else row['full_dfs_path'])
                        key=(mode,scope,version);binary,index=handles[key]
                        rec={k:row[k] for k in ('id','source_key','source_family','condition','k','selected_parts')}
                        rec.update(offset=offsets[key],length=len(ids),prefix_length=start,target_length=len(ids)-start,
                                   caption_method=quality if mode=='SFT' else 'path_only',pool=pool,pool_row_index=i)
                        binary.write(np.asarray(ids,dtype='<u4').tobytes());index.write(json.dumps(rec)+'\n')
                        offsets[key]+=len(ids);counts[f'{mode}_{scope}/{version}']+=1
                        supervised[f'{mode}_{scope}/{version}']+=len(ids)-start
                    families[mode].add(row['source_family'])
                    manifest.write(json.dumps(meta|{'mode':mode,'version':version,'cropped':row['cropped_to_path_budget']})+'\n')
        for mode in QUOTAS:
            for scope in ('M','G'):
                for version in ('v2','v3'):
                    for split in ('dev','test'):
                        d=ROOT/f'cache/{mode}_{scope}/{version}'
                        src=CURATED/f'cache/{mode}_{scope}/{version}'
                        for ext in ('u32','index.jsonl'):shutil.copyfile(src/f'{split}.{ext}',d/f'{split}.{ext}')
                        f={json.loads(l)['source_family'] for l in (d/f'{split}.index.jsonl').open()}
                        assert not f&families[mode]
    finally:
        for b,idx in handles.values():b.close();idx.close()
        for f in captions.values():f.close()
        manifest.close()
    for mode in QUOTAS:
        for scope in ('M','G'):
            cache=ROOT/f'cache/{mode}_{scope}'
            (cache/'READY.json').write_text(json.dumps({'status':'ready','cross_split_overlap':0,
                'artifact_sha256':{str(p.relative_to(cache)):sha_file(p) for p in cache.glob('*/*') if p.is_file()}},indent=2))
    report.update(total_per_arm=70000,effective_batch=32,max_iter_steps=2188,unique_serializations_each_arm=True,
                  supervised_token_totals=dict(supervised),
                  max_parts=100,max_tokens=6400,draw_without_replacement=True,
                  caption_quality='automatic source screening plus specified manual revisions; not full visual verification',
                  selection_source_sha256={k:sha_file(v) for k,v in POOLS.items()})
    (ROOT/'caption_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    (ROOT/'DATA_READY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
