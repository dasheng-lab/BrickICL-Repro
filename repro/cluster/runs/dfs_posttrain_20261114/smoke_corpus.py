import json,lzma
from pathlib import Path
from collections import Counter,defaultdict
import prepare_corpus as pc
from transformers import AutoTokenizer

pc.TOK=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
captions=defaultdict(list)
for line in lzma.open(pc.DATA/'captions_sft.jsonl.xz','rt'):
    r=json.loads(line);captions[str(r['id'])].append(r['caption'])
pc.CAPS=captions
found=Counter();examples=[]
for version in ('v2','v3'):
    pool=[json.loads(s) for s in Path(f'runs/reference_last_06/{version}/dev.jsonl').read_text().splitlines()]
    cases=[]
    for attr,predicate in [
        ('short_sft',lambda r:r['source_split']=='sft' and r['n_prefix']+r['k']<=100),
        ('long',lambda r:r['n_prefix']+r['k']>100),
        ('recolor',lambda r:r.get('palette_id',0)>0),
        ('stretch',lambda r:r['r']>0),
        ('delete',lambda r:r['deleted_copy_count']>0),
        ('pt',lambda r:r['source_split']=='pt')]:
        cases.extend((attr,r) for r in pool if predicate(r))
    for name,r in cases[:100]+[(name,r) for name,r in cases if name in ('long','pt')][:60]:
        result,err=pc.transform(json.dumps(r))
        if err:found[version+'/'+err]+=1;continue
        row,encoded=result
        found[version+'/'+name]+=1
        for arm,(ids,n) in encoded.items():
            expected=row['completion'] if arm.endswith('_M') else row['full_dfs_path']
            actual=pc.TOK.decode(ids[n:-1])
            assert actual==(expected[1:] if arm=='PT_G' else expected),(arm,actual[:30],expected[:30])
            assert len(ids)<=6400 and row['selected_parts']<=100
        if len(examples)<3 or (row['source_split']=='pt' and len(examples)<5):
            examples.append({k:row[k] for k in ('source_key','selected_parts','cropped_to_path_budget','caption_method','caption')})
(pc.ROOT/'caption_smoke_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
(pc.ROOT/'CORPUS_SMOKE.json').write_text(json.dumps(dict(found),indent=2))
print(dict(found),flush=True)
