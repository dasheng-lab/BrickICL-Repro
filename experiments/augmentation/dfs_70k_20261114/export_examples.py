import hashlib,json
from pathlib import Path
from collections import defaultdict
root=Path(__file__).resolve().parent
meta={r['id']:r for r in map(json.loads,(root/'selection_manifest.jsonl').open()) if r['mode']=='SFT'}
groups=defaultdict(list)
for version in ('v2','v3'):
    for r in map(json.loads,(root/f'captions_{version}_train.jsonl').open()):
        r.update(version=version,condition=meta[r['id']]['condition'])
        groups[version,r['condition']].append(r)
out=[]
seen=set()
for key,rows in sorted(groups.items()):
    selected=[]
    for r in sorted(rows,key=lambda r:hashlib.sha256(('review:'+r['id']).encode()).digest()):
        if r['source_key'] in seen:continue
        selected.append(r);seen.add(r['source_key'])
        if len(selected)==2:break
    out.extend(selected)
(root/'caption_examples.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('Actual selected caption examples',len(out))
