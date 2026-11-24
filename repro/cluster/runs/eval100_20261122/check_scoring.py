import hashlib,json,math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from reference_scoring import score_row,summarize_completion
ROOT=Path(__file__).resolve().parent
def main():
 rows=[json.loads(s) for s in (ROOT/'p1_inputs.jsonl').open()];jobs=[]
 for r in rows:
  rec={k:r[k] for k in ['id','source_key','condition','n_prefix','k_short','k_long','expected','prime']}
  rec['samples']=[{'text':r['target_'+r['expected']],'eos_emitted':True,'hit_token_cap':False} for _ in range(4)];jobs.append((r,rec))
 with ProcessPoolExecutor(max_workers=4) as pool:scored=list(pool.map(score_row,jobs,chunksize=2))
 m=summarize_completion(scored);assert m['samples']==800 and all(v==1 for v in m['f1'].values()) and abs(m['delta']-2*math.log(9))<1e-10
 assert all(s['generation_collision_free'] for r in scored for s in r['samples'])
 (ROOT/'SCORING_SELF_CHECK.json').write_text(json.dumps({'ideal_targets':m['samples'],'f1':m['f1'],'expected_max_delta':2*math.log(9),'delta':m['delta'],'collision_free':True,'input_sha256':hashlib.sha256((ROOT/'p1_inputs.jsonl').read_bytes()).hexdigest()},indent=2));print('All 800 gold continuations score F1=1, expected max delta, and no collisions',flush=True)
if __name__=='__main__':main()
