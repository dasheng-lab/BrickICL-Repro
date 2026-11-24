import json
from pathlib import Path
import numpy as np
from bricknet import parse_sample,tree_to_graph,decode_graph
from brickicl.sampling.build import read_graphs
from brickicl.posttrain.build_dataset import fingerprint
import factory

factory.init()
print('module part types',len(factory.PARTS),flush=True)
needed={74,229,62400}
results=[]
for i,g in enumerate(read_graphs(Path('baseline/bricknet/data/dataset/sft.npz.xz'))):
    if i not in needed:continue
    rows,skips=factory.process(('sft',i,g,fingerprint(g),'dev'))
    print('SMOKE',i,len(rows),skips,flush=True)
    assert rows and any(r['edit']['kind']!='none' for r in rows)
    for r in rows[:10]:
        p=parse_sample(r['full_dfs_path'])
        mats=np.asarray(decode_graph(tree_to_graph(p.tree)))
        key=factory.shape_id(p.tree.parts,mats)
        order=list(reversed(range(len(p.tree.parts))))
        transform=np.eye(4);transform[:3,3]=[123.,-45.,67.]
        transform[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        assert key==factory.shape_id([p.tree.parts[j] for j in order],transform[None]@mats[order])
        assert key==r['geometry_id'] and r['deletion_count']==0 and r['selected_parts']<=100
    results.extend(rows)
out=Path('runs/structural_20261114')
(out/'smoke_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
(out/'SMOKE_PASSED.json').write_text(json.dumps({'rows':len(results),'geometry_invariant':True,'no_deletion':True,
    'edit_types':sorted({r['edit']['kind'] for r in results})},indent=2))
