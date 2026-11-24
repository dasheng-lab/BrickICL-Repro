import json
from pathlib import Path
from brickicl.sampling.build import read_graphs
from brickicl.sampling.missing import pendant_copies
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
gs=read_graphs(REPO/'baseline/bricknet/data/dataset/val.npz.xz');rows=[]
for i,g in enumerate(gs):
 groups=pendant_copies(g,max_size=32,min_copies=2)
 good=[x for x in groups if x.size>=2]
 rows.append({'id':i,'n':len(g.part_ids),'groups':[(x.size,len(x.copies)) for x in good]})
print(json.dumps({'all':len(rows),'repeat_size2':sum(bool(r['groups']) for r in rows),'repeat_size2_under100':sum(any(r['n']+4*c<=100 for s,c in r['groups']) for r in rows),'rows':[r for r in rows if r['groups']]},indent=2))
