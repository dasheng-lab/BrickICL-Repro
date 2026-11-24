import hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=ROOT/'report/data';OUT.mkdir(parents=True,exist_ok=True)
def j(p):return json.loads(Path(p).read_text())
def read(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def compact(rs):
 return [{k:r[k] for k in ['id','source_key','condition','n_prefix','k_short','k_long','expected','prime'] if k in r}|{'samples':[{k:s.get(k) for k in ['pose_f1','pose_f1_untruncated','node_f1','long_feature','parse_ok','n_generated','target_parts','exact','eos_emitted','generation_collision_free']} for s in r['samples']]} for r in rs]
def lengths(path):return [{k:r.get(k) for k in ['id','seed','valid_length','collision_free_length','eos_emitted','invalid','n_actions','capped']} for r in read(path)]
def main():
 new=REPO/'runs/eval100_official_dfs_20261122';plan=j(new/'plan.json');models=plan['models'];data={};traces={};old={};physical={};train={}
 fox=REPO/'runs/viewer_parse_fox_fix_20261121';foxplan=j(fox/'FOX_PROTOCOL.json')
 for spec in models:
  arm=spec['arm'];d=new/'results'/arm
  if (d/'metrics.json').exists():
   assert j(d/'VERIFIED.json')['input_sha256']==plan['input_sha256'];data[arm]={'model':spec,'metrics':j(d/'metrics.json')['completion']['1'],'cases':compact(read(d/'p1.scored.jsonl'))};traces[str(d/'p1.scored.jsonl')]=sha(d/'p1.scored.jsonl')
  if arm in foxplan['models']:
   olddir=fox/'results'/arm;old[arm]={'metrics':j(fox/'metrics.json')[arm]['completion']['1'],'cases':compact(read(olddir/'p1.scored.jsonl'))}
   natural=Path(foxplan['old_sources'][arm]['metrics']).parent/('free.scored.jsonl' if arm not in ['frozen_pt','post_pt','fullg_pt','plain','masked'] else f'{arm}.free100.scored.jsonl')
   forced=REPO/'runs/no_eos_eval_20261120/results'/arm/'free_no_eos.scored.jsonl'
  else:
   run=next(x for x in j(REPO/'runs/latest_results_20261122/delivery/metrics.json') if x['model']['arm']==arm);ev=Path(run['evaluation_root']);old[arm]={'metrics':run['completion']['1'],'cases':compact(read(ev/'results'/arm/'p1.scored.jsonl'))};natural=ev/'results'/arm/'free.scored.jsonl';forced=ev/'no_eos/results'/arm/'free_no_eos.scored.jsonl'
  if arm=='grpo_pt':
   ev=REPO/'runs/grpo_v21_18_eval_20261122/results/grpo_pt';old[arm]={'metrics':j(ev/'metrics.json')['completion']['1'],'cases':compact(read(ev/'p1.scored.jsonl'))};natural=REPO/'runs/grpo_free_20261122/natural.scored.jsonl';forced=REPO/'runs/grpo_free_20261122/no_eos.scored.jsonl'
  assert natural.exists() and forced.exists(),(arm,natural,forced)
  physical[arm]={'natural':lengths(natural),'no_eos':lengths(forced),'natural_source':str(natural),'no_eos_source':str(forced)}
  if spec.get('checkpoint') and arm!='grpo_pt':
   directory=Path(spec['checkpoint']).parent
   config=directory/'config.json'
   if not config.exists():config=directory/'run_config.json'
   logs=directory/'metrics.jsonl'
   if not logs.exists():logs=directory/'log.jsonl'
   train[arm]={'directory':str(directory),'config':j(config) if config.exists() else None,'best':j(directory/'best.json') if (directory/'best.json').exists() else None,'completed':j(directory/'COMPLETED.json') if (directory/'COMPLETED.json').exists() else None,'logs':read(logs) if logs.exists() else [],'available_files':[x.name for x in directory.iterdir() if x.is_file()]}
 for arm in ['frozen_sft','post_sft','fullg_sft']:
  old[arm]={'model':foxplan['models'][arm],'metrics':j(fox/'metrics.json')[arm]['completion']['1'],'cases':compact(read(fox/'results'/arm/'p1.scored.jsonl'))}
  path=REPO/'runs/structural_eval_20261115'/f'{arm}.free100.scored.jsonl';physical[arm]={'natural':lengths(path),'no_eos':[]}
  if arm!='frozen_sft':
   directory=Path(foxplan['models'][arm]['checkpoint']).parent;train[arm]={'config':j(directory/'config.json'),'best':j(directory/'best.json'),'completed':j(directory/'COMPLETED.json'),'logs':read(directory/'metrics.jsonl')}
 rl=Path('/home/jywu2289/RL');run=rl/'runs/v2.1/train/grpo';cfg=j(run/'run_config.json')
 reward_path=rl/'common/rewards.py';driver=rl/'step9_grpo/grpo.py'
 write('grpo_training.json',{'config':cfg,'summary':j(run/'summary.json'),'train_log':read(run/'log.jsonl'),'dev_log':read(run/'dev.jsonl'),'reward_sha256_matches_training':sha(reward_path)==cfg['code']['common/rewards.py'],'current_driver_sha256_matches_training':sha(driver)==cfg['code']['step9_grpo/grpo.py'],'current_reward_source':reward_path.read_text(),'current_driver_source':driver.read_text()})
 write('completion_100.json',data);write('completion_18.json',old);write('physical.json',physical);write('training.json',train);write('models.json',models)
 rows=read(new/'p1_inputs.jsonl');write('dataset_summary.json',{'status':j(new/'DATASET_STATUS.json'),'validation':j(new/'DATASET_VALIDATED.json'),'plan':plan,'sources':[{k:v for k,v in s.items() if k!='rows'} for s in j(new/'sources.json')],'lengths':[{'source':r['source_key'],'condition':r['condition'],'n_prefix':r['n_prefix'],'k_target':r['k_'+r['expected']]} for r in rows]})
 write('provenance.json',{'snapshot_unix':time.time(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'git_remote':'https://github.com/nzd0420/BrickICL','completed_100_models':len(data),'raw_score_sha256':traces,'input100_sha256':sha(new/'p1_inputs.jsonl'),'note':'Local cluster contains uncommitted work; git commit alone does not identify all run code. Frozen benchmark scripts are copied separately.'})
 archive=ROOT/'report/repro/evaluation';archive.mkdir(parents=True,exist_ok=True)
 for name in ['build_dataset.py','validate_dataset.py','reference_scoring.py','run_eval.py','prepare_eval.py','p1_inputs.jsonl']:
  shutil.copy2(new/name,archive/name)
 print(json.dumps({'new100_models':len(data),'old18_models':len(old),'physical_models':len(physical),'training_models':len(train)}),flush=True)
 print('PRIMARY100',json.dumps({a:{k:data[a]['metrics'][k] for k in ['delta','f1']} for a in data},ensure_ascii=False),flush=True)
 print('TRAIN_FILES',json.dumps({a:t.get('available_files',[]) for a,t in train.items() if not t['logs']}),flush=True)
if __name__=='__main__':main()
