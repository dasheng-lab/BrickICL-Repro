import json,hashlib,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];REPORT=ROOT/'report';DATA=REPORT/'data'
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.open() if s.strip()]
def compact(rows):
 return [{k:r[k] for k in ['id','source_key','condition','n_prefix','k_short','k_long','expected','prime'] if k in r}|{'samples':[{k:s.get(k) for k in ['pose_f1','pose_f1_untruncated','node_f1','long_feature','parse_ok','n_generated','target_parts','exact','eos_emitted','generation_collision_free']} for s in r['samples']]} for r in rows]
def main():
 if not REPORT.exists():shutil.copytree(REPO/'runs/final_report_20261123/author_bundle',REPORT)
 new=REPO/'runs/eval100_official_dfs_20261122';assert j(new/'LIVE_VERIFIED.json')['state']=='verified';plan=j(new/'plan.json');previous=j(DATA/'completion_100.json');updated={};changed=[];hashes={}
 for spec in plan['models']:
  arm=spec['arm'];p=new/'results'/arm;raw=j(p/'metrics.json');v=j(p/'VERIFIED.json');assert v['input_sha256']==plan['input_sha256'] and v['completion_samples']==800
  assert sha(p/'p1.scored.jsonl')==v['output_sha256']['p1.scored.jsonl'];hashes[arm]=sha(p/'p1.scored.jsonl')
  updated[arm]={'model':spec,'metrics':raw['completion']['1'],'cases':compact(read(p/'p1.scored.jsonl'))}
  if previous.get(arm,{}).get('metrics')!=updated[arm]['metrics']:changed.append(arm)
 assert len(updated)==15
 old=j(DATA/'completion_18.json');p=REPO/'runs/grpo_v21_18_eval_20261122/results/grpo_pt';old['grpo_pt']={'metrics':j(p/'metrics.json')['completion']['1'],'cases':compact(read(p/'p1.scored.jsonl'))}
 phys=j(DATA/'physical.json');p=REPO/'runs/grpo_free_20261122';assert j(p/'LIVE_VERIFIED.json')['state']=='verified'
 for mode,file in [('natural','natural.scored.jsonl'),('no_eos','no_eos.scored.jsonl')]:phys['grpo_pt'][mode]=[{k:r.get(k) for k in ['id','seed','valid_length','collision_free_length','eos_emitted','invalid','n_actions','capped']} for r in read(p/file)]
 for name,value in [('completion_100.json',updated),('completion_18.json',old),('physical.json',phys)]: (DATA/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
 audit={'final_verification_unix':time.time(),'new100_models':15,'new100_sources':100,'new100_samples_per_model':800,'old18_samples_per_model':144,'changed_primary_metrics_since_draft':changed,'all_current_score_hashes':hashes,'new100_live_verified':j(new/'LIVE_VERIFIED.json'),'grpo_independent_live_verified':j(p/'LIVE_VERIFIED.json'),'unverified_fullg_regularized_GRPO_included':False,'grpo_checkpoint':updated['grpo_pt']['model']['checkpoint']}
 (DATA/'final_result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'verified_models':15,'changed_metric_rows':changed,'new_fullg_GRPO_included':False}),flush=True)
if __name__=='__main__':main()
