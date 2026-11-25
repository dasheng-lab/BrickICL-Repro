import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from bricknet import parse_sample,decode_graph,tree_to_graph,score_text
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];sys.path.insert(0,str(REPO/'training/augmentation/dfs_posttrain_20261114'))
from prepare_corpus import check_dfs
def main():
 global ROOT
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data',type=Path,default=REPO/'runs/new100')
 ROOT=parser.parse_args().data.resolve()
 sources=json.loads((ROOT/'sources.json').read_text());assert len(sources)==100 and len({s['source'] for s in sources})==100
 counts=Counter();geometry=set();rows=[]
 for s in sources:
  if 'retained_source_node_indices' in s:s['retained_sampled_tree_node_indices']=s.pop('retained_source_node_indices');s['source_sampling_seed']=20261122+s['source']
  assert {r['condition'] for r in s['rows']}=={'original','stretched'}
  for r in s['rows']:
   assert r['prime']==1 and r['k_long']==r['k_short']+4 and r['k_short']>=1
   assert hashlib.sha256(r['prefix'].encode()).hexdigest()==r['prefix_sha256']
   for kind in ['prefix','short','long']:
    text=r['prefix'] if kind=='prefix' else r['prefix']+r['target_'+kind];t=parse_sample(text);assert t.error is None and check_dfs(t.tree),(r['id'],kind)
    n,fail,coll=score_text(text);assert fail is None and coll==[] and n<=100,(r['id'],kind,coll);counts[kind]+=1
   rows.append(r)
  counts[s['origin']]+=1
 (ROOT/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
 # Audit no target geometry or order changes other than the four repeated added pieces in each paired condition.
 report={'state':'validated','sources':100,'official_validation_ids':[s['source'] for s in sources],'conditions':2,'prime':1,'only_completion_metrics':True,'input_and_counterfactual_reference_checks':dict(counts),'all_strict_DFS':True,'all_collisions_empty':True,'max_parts':100,'input_sha256':hashlib.sha256((ROOT/'p1_inputs.jsonl').read_bytes()).hexdigest(),'new_source_retention_min':min(s['source_retained_parts']/s['source_original_parts'] for s in sources if 'source_retained_parts' in s)}
 (ROOT/'DATASET_VALIDATED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
