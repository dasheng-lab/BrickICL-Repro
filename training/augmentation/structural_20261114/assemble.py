"""Build four new caches using geometry-unique normal/modified records; no deletion or old caches."""
import hashlib,json,math
from collections import Counter,defaultdict,deque
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT=Path('runs/structural_20261114')
PREFERRED={'PT':{'v2':20000,'v3':50000},'SFT':{'v2':10000,'v3':60000}}
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fair_order(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[r['family']][(r['condition'],r['edit_kind'],r['added'])].append(r)
    queues={}
    for f,variants in groups.items():
        bins=[deque(sorted(rs,key=lambda r:digest(r['id']))) for _,rs in sorted(variants.items())]
        q=[]
        while any(bins):
            for b in bins:
                if b:q.append(b.popleft())
        queues[f]=deque(q)
    keys=sorted(queues,key=lambda f:digest('structural:'+f))
    while any(queues.values()):
        for f in keys:
            if queues[f]:yield queues[f].popleft()
def choose(rows,n,blocked):
    selected=[]
    for r in fair_order(rows):
        if r['geometry'] in blocked:continue
        selected.append(r);blocked.add(r['geometry'])
        if len(selected)==n:break
    return selected
def main():
    if (ROOT/'DATA_READY.json').exists():return
    metadata=defaultdict(list)
    source_hashes={}
    for pool in ('sft','pt'):
        assert (ROOT/pool/'READY.json').exists()
        for version in ('v2','v3'):
            for split in ('dev','test','train'):
                path=ROOT/pool/f'{version}_{split}.jsonl'
                source_hashes[str(path.relative_to(ROOT))]=sha(path)
                for i,line in enumerate(path.open()):
                    r=json.loads(line)
                    assert r['condition'] in ('original','modified') and r['deletion_count']==0
                    assert r['serialization_count']==1 and r['selected_parts']<=100
                    metadata[version,split].append({'pool':pool,'version':version,'split':split,'i':i,'id':r['id'],
                        'family':r['source_family'],'source':r['source_key'],'geometry':r['geometry_id'],
                        'caption':r['has_semantic_caption'],'condition':r['condition'],'edit_kind':r['edit']['kind'],
                        'added':r['edit'].get('added_per_copy',0),'k':r['k'],'parts':r['selected_parts']})
    forbidden=set();held={}
    for split in ('dev','test'):
        for version in ('v2','v3'):
            # Prefer a captioned representative when the same geometry occurs in both source pools.
            rows=sorted(metadata[version,split],key=lambda r:not r['caption'])
            unique={}
            for r in rows:
                if r['geometry'] not in forbidden:unique.setdefault(r['geometry'],r)
            held[version,split]=list(unique.values())
            forbidden.update(unique)
    selection={};stats={};lookup=defaultdict(list)
    for mode,preferred in PREFERRED.items():
        blocked=set(forbidden);stats[mode]={}
        available={v:[r for r in metadata[v,'train'] if mode=='PT' or r['caption']] for v in ('v2','v3')}
        v3=choose(available['v3'],preferred['v3'],blocked)
        v2_needed=70000-len(v3)
        assert 0<v2_needed<len(v3),(mode,'insufficient v3 structures',len(v3))
        v2=choose(available['v2'],v2_needed,blocked)
        assert len(v2)==v2_needed,(mode,'insufficient v2 structures',len(v2),v2_needed)
        for version,rows in (('v2',v2),('v3',v3)):
            selection[mode,version,'train']=rows
            stats[mode][version]={'rows':len(rows),'unique_geometries':len({r['geometry'] for r in rows}),
                'sources':len({r['source'] for r in rows}),'families':len({r['family'] for r in rows}),
                'conditions':dict(Counter(r['condition'] for r in rows)),
                'edit_types':dict(Counter(r['edit_kind'] for r in rows))}
            for split in ('dev','test'):
                selection[mode,version,split]=[r for r in held[version,split] if mode=='PT' or r['caption']]
                assert selection[mode,version,split],(mode,version,split,'empty evaluation data')
        assert sum(r['condition']=='modified' for r in v2+v3)>35000
    for (mode,version,split),rows in selection.items():
        for r in rows:lookup[r['pool'],version,split,r['i']].append(mode)
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    totals=Counter();supervised=Counter();conditions=Counter();caption_examples=[];manifest=[]
    families=defaultdict(set)
    for version in ('v2','v3'):
        for split in ('train','dev','test'):
            handles={};offsets=Counter()
            for mode in PREFERRED:
                for scope in ('M','G'):
                    arm=f'{mode}_{scope}';d=ROOT/'cache'/arm/version;d.mkdir(parents=True,exist_ok=True)
                    handles[arm]=((d/f'{split}.u32').open('wb'),(d/f'{split}.index.jsonl').open('w'))
            with (ROOT/f'captions_{version}_{split}.jsonl').open('w') as cf:
                try:
                    for pool in ('sft','pt'):
                        for i,line in enumerate((ROOT/pool/f'{version}_{split}.jsonl').open()):
                            modes=lookup.get((pool,version,split,i),[])
                            if not modes:continue
                            r=json.loads(line)
                            prefix=tok.encode(r['prompt'],add_special_tokens=False)
                            target=tok.encode(r['completion'],add_special_tokens=False)
                            full=tok.encode(r['full_dfs_path'],add_special_tokens=False)
                            assert tok.encode(r['prompt']+r['completion'],add_special_tokens=False)==prefix+target
                            caption=r['caption'].replace('Insert 1 additional copies','Insert 1 additional copy').replace('(1 parts)','(1 part)')
                            for mode in modes:
                                head=tok.encode(caption,add_special_tokens=False)+tok.encode('\n',add_special_tokens=False) if mode=='SFT' else []
                                for scope in ('M','G'):
                                    arm=f'{mode}_{scope}'
                                    ids=head+(prefix+target if scope=='M' else full)+[tok.eos_token_id]
                                    start=len(head)+len(prefix) if scope=='M' else len(head) if mode=='SFT' else 1
                                    assert len(ids)<=6400 and ids[-1]==tok.eos_token_id
                                    b,idx=handles[arm]
                                    rec={k:r[k] for k in ('id','source_key','source_family','condition','k','selected_parts','geometry_id')}
                                    rec.update(offset=offsets[arm],length=len(ids),prefix_length=start,target_length=len(ids)-start,
                                               source_pool=pool,source_row=i,edit=r['edit'])
                                    b.write(np.asarray(ids,dtype='<u4').tobytes());idx.write(json.dumps(rec)+'\n')
                                    offsets[arm]+=len(ids);totals[f'{arm}/{version}/{split}']+=1
                                    supervised[f'{arm}/{version}/{split}']+=len(ids)-start
                                families[mode,split].add(r['source_family'])
                                if split=='train':
                                    manifest.append({'mode':mode,'version':version,'id':r['id'],'geometry_id':r['geometry_id'],
                                        'source_pool':pool,'source_row':i,'condition':r['condition'],'edit':r['edit'],
                                        'source_key':r['source_key'],'family':r['source_family']})
                                conditions[mode,split,r['condition']]+=1
                                if mode=='SFT':
                                    cf.write(json.dumps({'id':r['id'],'caption':caption,'source_key':r['source_key'],'edit':r['edit']})+'\n')
                                    if split=='train' and len(caption_examples)<12:
                                        caption_examples.append({'id':r['id'],'version':version,'caption':caption,'edit':r['edit']})
                    print(version,split,{a:totals[f'{a}/{version}/{split}'] for a in handles},flush=True)
                finally:
                    for b,idx in handles.values():b.close();idx.close()
    for mode in PREFERRED:
        for a,b in (('train','dev'),('train','test'),('dev','test')):assert not families[mode,a]&families[mode,b]
        for scope in ('M','G'):
            cache=ROOT/'cache'/f'{mode}_{scope}'
            (cache/'READY.json').write_text(json.dumps({'status':'ready','cross_split_overlap':0,
                'artifact_sha256':{str(p.relative_to(cache)):sha(p) for p in cache.glob('*/*') if p.is_file()}},indent=2))
    (ROOT/'selection_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (ROOT/'caption_examples.json').write_text(json.dumps(caption_examples,ensure_ascii=False,indent=2))
    report={'status':'ready','selection':stats,'counts':dict(totals),'deleted_rows':0,'max_parts':100,
        'supervised_token_totals':dict(supervised),
        'augmentation':'actual repeated-subassembly stack insertions and matching terminal modules',
        'deduplication':'frame/order invariant part/pose geometry signature, color excluded',
        'one_DFS_serialization_per_geometry':True,'source_sha256':source_hashes,
        'cross_split_family_overlap':0,'cross_split_geometry_overlap':0,'rows_per_arm':70000,
        'code_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')}}
    (ROOT/'DATA_READY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
