import argparse,hashlib,itertools,json
from collections import defaultdict
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--id');args=ap.parse_args()
groups=defaultdict(list)
for version in ('v2','v3'):
    for r in map(json.loads,(root/f'captions_{version}_train.jsonl').open()):
        r['version']=version;groups[version,r['edit']['kind']].append(r)
examples=[];used=set()
for key,rs in sorted(groups.items()):
    count=0
    for r in sorted(rs,key=lambda r:hashlib.sha256(('example:'+r['id']).encode()).digest()):
        if r['source_key'] in used:continue
        examples.append(r);used.add(r['source_key']);count+=1
        if count==2:break
(root/'caption_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
target=args.id or next(r['id'] for r in examples if r['version']=='v3' and r['edit']['kind']=='tip_extension')
meta=next(r for r in map(json.loads,(root/'selection_manifest.jsonl').open()) if r['id']==target and r['mode']=='SFT')
with (root/meta['source_pool']/f'{meta["version"]}_train.jsonl').open() as f:
    row=json.loads(next(itertools.islice(f,meta['source_row'],meta['source_row']+1)))
cap=next(r for r in map(json.loads,(root/f'captions_{meta["version"]}_train.jsonl').open()) if r['id']==target)
tree=parse_sample(row['full_dfs_path'])
assert tree.error is None and row['deletion_count']==0
out=root/'example_export';out.mkdir(exist_ok=True)
(out/'model.ldr').write_text(graph_to_ldr(tree_to_graph(tree.tree)))
(out/'caption.txt').write_text(cap['caption'])
(out/'record.json').write_text(json.dumps(meta|{'parts':len(tree.tree.parts),'caption':cap['caption']},ensure_ascii=False,indent=2))
print(json.dumps({'examples':len(examples),'exported':target,'parts':len(tree.tree.parts)}))
