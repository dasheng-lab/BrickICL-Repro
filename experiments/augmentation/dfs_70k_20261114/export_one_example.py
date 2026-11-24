import argparse,itertools,json
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr

root=Path(__file__).resolve().parent
repo=root.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--id',help='SFT record ID; defaults to a selected v3 caption example')
args=parser.parse_args()
examples=json.loads((root/'caption_examples.json').read_text())
target=args.id or next(r['id'] for r in examples if r['version']=='v3')
meta=next(r for r in map(json.loads,(root/'selection_manifest.jsonl').open()) if r['id']==target and r['mode']=='SFT')
paths={'v2':repo/'runs/dfs_posttrain_20261114/corpus/v2/train.jsonl',
       'v3_old':repo/'runs/dfs_posttrain_20261114/corpus/v3/train.jsonl','v3_extra':root/'extra_v3.jsonl'}
with paths[meta['pool']].open() as stream:row=json.loads(next(itertools.islice(stream,meta['i'],meta['i']+1)))
cap=next(r for r in map(json.loads,(root/f'captions_{meta["version"]}_train.jsonl').open()) if r['id']==target)
parsed=parse_sample(row['full_dfs_path'])
assert parsed.error is None
out=root/'example_export';out.mkdir(exist_ok=True)
(out/'model.ldr').write_text(graph_to_ldr(tree_to_graph(parsed.tree)))
(out/'caption.txt').write_text(cap['caption'])
(out/'record.json').write_text(json.dumps({'selection':meta,'caption':cap,'parts':len(parsed.tree.parts)},ensure_ascii=False,indent=2))
print(json.dumps({'directory':str(out),'id':target,'parts':len(parsed.tree.parts)}))
