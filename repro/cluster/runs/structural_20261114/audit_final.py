import json,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
report={};needed=defaultdict(set);records={}
for mode in ('PT','SFT'):
    ids=[];geometries=[];conditions=Counter();types=Counter()
    for version in ('v2','v3'):
        paired={}
        for scope in ('M','G'):
            path=root/f'cache/{mode}_{scope}/{version}/train.index.jsonl'
            rows=[json.loads(s) for s in path.open()]
            paired[scope]=rows
            records[mode,scope,version]=random.Random(0).sample(rows,min(4,len(rows)))
            for r in records[mode,scope,version]:needed[r['source_pool'],version].add(r['source_row'])
            for split in ('dev','test'):
                assert all(json.loads(s)['condition'] in ('original','modified') for s in (root/f'cache/{mode}_{scope}/{version}/{split}.index.jsonl').open())
        assert [r['id'] for r in paired['M']]==[r['id'] for r in paired['G']]
        for r in paired['M']:
            assert r['condition'] in ('original','modified') and r['selected_parts']<=100
            ids.append(r['id']);geometries.append(r['geometry_id']);conditions[r['condition']]+=1;types[r['edit']['kind']]+=1
    assert len(ids)==len(set(ids))==len(set(geometries))==70000
    report[mode]={'rows':len(ids),'unique_geometries':len(set(geometries)),'conditions':dict(conditions),
                  'edit_types':dict(types),'modified_fraction':conditions['modified']/len(ids),'deletion_rows':0}
raw={}
for (pool,version),indices in needed.items():
    for i,line in enumerate((root/pool/f'{version}_train.jsonl').open()):
        if i in indices:raw[pool,version,i]=json.loads(line)
checked=0
for (mode,scope,version),rs in records.items():
    array=np.memmap(root/f'cache/{mode}_{scope}/{version}/train.u32',dtype='<u4',mode='r')
    for r in rs:
        row=raw[r['source_pool'],version,r['source_row']]
        ids=array[r['offset']:r['offset']+r['length']].tolist();n=r['prefix_length']
        assert ids[-1]==tok.eos_token_id and len(ids)<=6400
        target=row['completion'] if scope=='M' else row['full_dfs_path'] if mode=='SFT' else row['full_dfs_path'][1:]
        assert tok.decode(ids[n:-1])==target
        caption=row['caption'].replace('Insert 1 additional copies','Insert 1 additional copy').replace('(1 parts)','(1 part)')
        prompt=(caption+'\n' if mode=='SFT' else '')+(row['prompt'] if scope=='M' else '' if mode=='SFT' else 'a')
        assert tok.decode(ids[:n])==prompt
        checked+=1
report['encoding_checks']=checked
report['status']='passed'
(root/'VALIDATED.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
