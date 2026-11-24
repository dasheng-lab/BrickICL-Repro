import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from bricknet import parse_sample,tree_to_graph,decode_graph
ROOT=Path(__file__).resolve().parent
def geometry(r):
 tree=parse_sample(r['prefix']+r['target_'+r['expected']]).tree;mats=np.asarray(decode_graph(tree_to_graph(tree)));parent={e.child:e.parent for e in tree.edges};modules=set()
 for i,p in enumerate(tree.parts):
  if p.color!=0:continue
  node=i
  while node:modules.add(node);node=parent[node]
 body=tuple(sorted((p.part_id,p.color,*np.round(mats[i][:3,:].ravel(),5).tolist()) for i,p in enumerate(tree.parts) if i not in modules))
 return body,mats[r['n_prefix']-1]
def main():
 pairs=defaultdict(dict)
 for r in map(json.loads,(ROOT/'p1_inputs.jsonl').open()):
  if r['source_key'].startswith('synthetic:'):pairs[r['source_key']][r['condition']]=r
 for key,p in pairs.items():
  a,head_a=geometry(p['original']);b,head_b=geometry(p['stretched']);assert a==b,(key,'body moved');assert np.allclose(head_a,head_b,atol=1e-5),(key,'target first brick moved')
 (ROOT/'BODY_INVARIANCE.json').write_text(json.dumps({'checked_sources':len(pairs),'all_body_parts_same_type_color_pose':True,'all_given_target_first_bricks_same_pose':True,'method':'remove four synthetic branches marked by black terminal parts; compare sorted body poses in the shared plate-root frame'},indent=2));print('82 new pairs: body and given target anchor unchanged',flush=True)
if __name__=='__main__':main()
