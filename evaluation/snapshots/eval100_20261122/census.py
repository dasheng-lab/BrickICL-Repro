import json,time
from pathlib import Path
from brickicl.sampling.build import read_graphs
from brickicl.sampling.missing import pendant_copies
from brickicl.sampling.edits import stack_joint
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
graphs=read_graphs(REPO/'baseline/bricknet/data/dataset/val.npz.xz')
rows=[];start=time.time()
for gid,g in enumerate(graphs):
 groups=pendant_copies(g,max_size=12,min_copies=2);stretch=[]
 for ix,grp in enumerate(groups):
  joints=[stack_joint(g,c,a) for c,a in zip(grp.copies,grp.anchors)]
  if all(joints):stretch.append({'index':ix,'size':grp.size,'copies':len(grp.copies),'nodes':len(g.part_ids),'fully_stretched_nodes':len(g.part_ids)+4*len(grp.copies)})
 rows.append({'source':gid,'nodes':len(g.part_ids),'repeat_groups':len(groups),'stretch_groups':stretch})
summary={'total':len(rows),'with_repeat':sum(bool(r['repeat_groups']) for r in rows),'with_stretch':sum(bool(r['stretch_groups']) for r in rows),'with_stretch_under100':sum(any(g['fully_stretched_nodes']<=100 for g in r['stretch_groups']) for r in rows),'total_groups_under100':sum(sum(g['fully_stretched_nodes']<=100 for g in r['stretch_groups']) for r in rows),'seconds':time.time()-start}
(ROOT/'CENSUS.json').write_text(json.dumps({'summary':summary,'sources':rows},indent=2));print(json.dumps(summary),flush=True)
