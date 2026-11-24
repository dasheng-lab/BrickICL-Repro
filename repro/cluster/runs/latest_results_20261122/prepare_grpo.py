import json,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];DEST=REPO/'runs/grpo_v21_eval_20261122';OLD=REPO/'runs/grpo_eval_20261120'
DEST.mkdir(exist_ok=True)
for name in ['common.py','direct_loader.py','generate.py','score.py','verify.py','prepare.py']:
 shutil.copy2(OLD/name,DEST/name)
p=DEST/'prepare.py';s=p.read_text().replace("RL/runs/r1'","RL/runs/v2.1'").replace('==300','==400').replace("'step':300","'step':400").replace("'dev_best_step':300","'dev_best_step':400").replace('checkpoint-300','checkpoint-400').replace('step300','step400').replace('step 300','step 400')
# Readability must be checked before hashing protected weights. prepare.py writes ACCESS_CHECK.json on failure.
p.write_text(s)
launch='''#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
RUN=runs/grpo_v21_eval_20261122
python "$RUN/prepare.py"
python -u "$RUN/generate.py"
python -u "$RUN/score.py"
python "$RUN/verify.py"
python runs/viewer_parse_fox_fix_20261121/future_patch.py "$RUN"
python "$RUN/verify.py"
python runs/no_eos_eval_20261120/evaluate_future.py "$RUN"
'''
(DEST/'run_eval.sh').write_text(launch)
src=Path('/home/jywu2289/RL/runs/v2.1');cfg=json.loads((src/'train/grpo/run_config.json').read_text());best=json.loads((src/'eval/dev/grpo/best.json').read_text())
assert best['best']['step']==400 and cfg['args']['steps']==400 and cfg['args']['first_part']
audit={'source_run':str(src),'best':best['best'],'training_args':cfg['args'],'weights_readable':os.access(src/'train/grpo/checkpoint-400/adapter_model.safetensors',os.R_OK),'state':'ready' if os.access(src/'train/grpo/checkpoint-400/adapter_model.safetensors',os.R_OK) else 'blocked_read_permission','prepared_eval':str(DEST),'not_evaluated_yet':True}
(DEST/'SOURCE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(audit,ensure_ascii=False),flush=True)
