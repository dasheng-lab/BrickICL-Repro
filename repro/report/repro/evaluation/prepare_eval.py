import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def main():
 rows=[json.loads(s) for s in (ROOT/'p1_inputs.jsonl').open()];assert len(rows)==200
 old=json.loads((REPO/'runs/eval100_20261122/plan.json').read_text());models=old['models'];tok=AutoTokenizer.from_pretrained(models[0]['base'],local_files_only=True)
 rows.sort(key=lambda r:(len(tok.encode(r['prefix'],add_special_tokens=False)),r['id']))
 groups=[]
 for start in range(0,200,8):
  batch=rows[start:start+8];cap=max(512,64*max(r['k_long'] for r in batch)+256);assert max(len(tok.encode(r['prefix'],add_special_tokens=False)) for r in batch)+cap<=6400
  groups.append({'ids':[r['id'] for r in batch],'seed':20261122+start*1009,'max_new_tokens':cap})
 scoring=(REPO/'runs/structural_eval_20261115/score_results.py').read_text().replace('assert len(sources)==18 and all(','assert len(sources)==100 and all(').replace('integers(18,size=(2000,18))','integers(len(sources),size=(2000,len(sources)))')
 (ROOT/'reference_scoring.py').write_text(scoring)
 plan={'models':models,'sources':100,'source_pool':'official val only','conditions':['original','stretched'],'prime':1,'samples_per_condition':4,'only_completion':True,'independent_generation':False,'strict_DFS':True,'temperature':1.,'top_k':20,'top_p':.95,'batch_plan':groups,'input_sha256':hashlib.sha256((ROOT/'p1_inputs.jsonl').read_bytes()).hexdigest(),'scoring_sha256':hashlib.sha256(scoring.encode()).hexdigest(),'checkpoint_selection':'previous development-best unchanged; no test selection','note':'Previous rejected synthetic run is stopped and not reused; even old18 are reserialized using strict DFS and rerun.'}
 (ROOT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));print('15 models, 100 official-val objects, 800 prime1 completions/model, no independent generation',flush=True)
if __name__=='__main__':main()
