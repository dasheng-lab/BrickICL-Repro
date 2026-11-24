import hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from bricknet import parse_sample,tree_to_graph,decode_graph
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def shape(parts,mats):
 pids=[p.part_id for p in parts];counts=Counter(pids);anchor=min(counts,key=lambda p:(counts[p],p));best=None
 for a in range(len(parts)):
  if pids[a]!=anchor:continue
  rel=np.linalg.inv(mats[a])[None]@np.asarray(mats);key=tuple(sorted((int(pid),*np.rint(t[:3,:3].ravel()*100000).astype(int).tolist(),*np.rint(t[:3,3]*1000).astype(int).tolist()) for pid,t in zip(pids,rel)))
  if best is None or key<best:best=key
 return hashlib.sha256(repr(best).encode()).hexdigest()
def inv(text):return tuple(sorted(Counter(line.split(' ',1)[1].split(' | ',1)[0] for line in text.splitlines() if ' | ' in line).items()))
def main():
 begin=time.time();tests={};inventories=defaultdict(set)
 for r in map(json.loads,(ROOT/'p1_inputs.jsonl').open()):
  if not r['source_key'].startswith('synthetic:'):continue
  txt=r['prefix']+r['target_'+r['expected']];tree=parse_sample(txt).tree;key=shape(tree.parts,np.asarray(decode_graph(tree_to_graph(tree))));tests[key]=r['id'];inventories[inv(txt)].add(key)
 matches=[];n=0
 for row in map(json.loads,(REPO/'runs/structural_20261114/selection_manifest.jsonl').open()):
  if row['mode']!='PT':continue
  n+=1
  if row['geometry_id'] in tests:matches.append({'training':row['id'],'test':tests[row['geometry_id']]})
 nzd=Path('/home/nzd0420/brickICL/runs/nzd-supertry/sft_train_v2/plain.jsonl');m=0;candidates=0
 for row in map(json.loads,nzd.open()):
  m+=1;txt=row['path'];key=inv(txt)
  if key not in inventories:continue
  candidates+=1;t=parse_sample(txt).tree;k=shape(t.parts,np.asarray(decode_graph(tree_to_graph(t))))
  if k in tests:matches.append({'training':str(nzd)+':'+str(m-1),'test':tests[k]})
 report={'new_test_full_geometries':len(tests),'ourdata_selected_PT_rows':n,'nzd_v2_plain_rows':m,'nzd_v2_inventory_candidates':candidates,'exact_geometry_matches':matches,'scope':'ourdata PT selected geometries and exact nzd v2 source paths (plain/masked share geometry); not an exhaustive audit of official BrickNet pretraining or historical v4/official-path subsets','seconds':time.time()-begin}
 (ROOT/'TRAINING_OVERLAP_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));assert not matches,matches;print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
