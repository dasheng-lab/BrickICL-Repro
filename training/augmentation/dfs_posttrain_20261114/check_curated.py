import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

root=Path('runs/dfs_curated_20261114')
old=Path('runs/dfs_posttrain_20261114')
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
checks=[]
for v in ('v2','v3'):
    captions=[json.loads(s) for s in (root/f'captions_{v}_train.jsonl').open()]
    selected=captions[:2]+[r for r in captions if r['quality'].startswith('manual')][:3]
    needed={r['original_row_index']:r for r in selected}
    original={}
    for i,line in enumerate((old/f'corpus/{v}/train.jsonl').open()):
        if i in needed:original[i]=json.loads(line)
    for scope in ('M','G'):
        idx={r['id']:r for r in map(json.loads,(root/f'cache/SFT_{scope}/{v}/train.index.jsonl').open())}
        data=np.memmap(root/f'cache/SFT_{scope}/{v}/train.u32',mode='r',dtype='<u4')
        for cap in selected:
            r=original[cap['original_row_index']];rec=idx[cap['id']]
            ids=data[rec['offset']:rec['offset']+rec['length']].tolist()
            n=rec['prefix_length']
            assert ids[-1]==tok.eos_token_id and len(ids)<=6400
            assert tok.decode(ids[n:-1])==(r['completion'] if scope=='M' else r['full_dfs_path'])
            assert tok.decode(ids[:n])==cap['caption']+'\n'+(r['prompt'] if scope=='M' else '')
            checks.append([v,scope,cap['id']])
report={'status':'passed','checked_caption_label_boundaries':len(checks),'rows':checks}
(root/'CAPTION_ENCODING_CHECK.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
