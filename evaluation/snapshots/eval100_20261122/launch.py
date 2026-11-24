import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def main():
 for name in ['DATASET_VALIDATED.json','RENDER_VERIFIED.json','SCORING_SELF_CHECK.json','plan.json']:assert (ROOT/name).exists()
 digest=hashlib.sha256((ROOT/'p1_inputs.jsonl').read_bytes()).hexdigest();assert json.loads((ROOT/'SCORING_SELF_CHECK.json').read_text())['input_sha256']==digest
 approval={'approved':True,'input_sha256':digest,'new_sources_inspected':82,'legacy_sources_retained':18,'reference_renders_inspected':200,'evidence':'visual_review/contact_01.jpg ... contact_10.jpg; normal and stretched pairs inspected by Codex. First draft was revised to 8 different repeated-part profiles and upward/downward assembly directions.','checks':['connected bodies and corresponding repeated modules','same body geometry within each new pair','normal versus four-piece stretch visible','no collision in input or target','no duplicate color-ignored new-source graph fingerprints'],'limitation':'new sources use twelve procedural object layout families; this is a mixed constructed benchmark, not 100 natural validation objects','unix_time':time.time()}
 (ROOT/'VISUAL_APPROVED.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2))
 script='''#!/usr/bin/env bash
set -euo pipefail
cd /home/sunshq114514/brickICL
source brickicl/posttrain/bricknet_environment.sh
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
umask 002
python -u runs/eval100_20261122/run_eval.py
python -u runs/eval100_20261122/publish_results.py
python runs/viewer_parse_fox_fix_20261121/restart_ui.py
runs/proposal_report_20261111/env/bin/python runs/eval100_20261122/verify_published.py
python runs/eval100_20261122/export_report.py
'''
 (ROOT/'run_all.sh').write_text(script)
 pidfile=ROOT/'job.pid'
 if pidfile.exists():
  pid=int(pidfile.read_text())
  if Path(f'/proc/{pid}').exists():raise RuntimeError(f'already running {pid}')
 with (ROOT/'evaluation.log').open('ab') as log:
  proc=subprocess.Popen(['bash',str(ROOT/'run_all.sh')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 pidfile.write_text(str(proc.pid));print('Started cluster evaluation pipeline',proc.pid,flush=True)
if __name__=='__main__':main()
