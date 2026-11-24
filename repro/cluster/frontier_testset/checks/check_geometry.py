import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from bricknet import parse_sample,tree_to_graph,decode_graph
from brickicl.eval.match import same_pose
from pathlib import Path
D=Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122/p1_inputs.jsonl')
F=Path('/home/shensx3017/brickicl_eval/frontier_testset/p1_inputs.jsonl')
a={r['id']:r for r in map(json.loads,D.open())}
b={r['id']:r for r in map(json.loads,F.open())}
assert a.keys()==b.keys()
for ident in a:
 for suffix in ('short','long'):
  def parts(row):
   t=parse_sample(row['prefix']+row['target_'+suffix]).tree
   return t.parts,np.asarray(decode_graph(tree_to_graph(t)))
  pa,ta=parts(a[ident]);pb,tb=parts(b[ident])
  assert len(pa)==len(pb)
  match=np.zeros((len(pa),len(pb)),dtype=int)
  for i,p in enumerate(pa):
   for j,q in enumerate(pb):
    if p.part_id==q.part_id and same_pose(p.part_id,ta[i],tb[j]):match[i,j]=1
  assert match[linear_sum_assignment(-match)].sum()==len(pa),(ident,suffix)
print('GEOMETRY_IDENTICAL',len(a),'paired inputs, short and long targets',flush=True)
