"""One dev-selected nzd arm, using the existing fixed completion/physical protocol."""
import json,shutil,subprocess,sys
from pathlib import Path
from prepare import ROOT,sha,atomic
REPO=ROOT.parents[1];SOURCE=REPO/'runs/qwen3_direct_eval_20261119'

def main(arm):
    assert arm in ('plain','masked')
    out=REPO/'runs'/f'qwen3_nzd_v2_{arm}_eval_20261120';out.mkdir(exist_ok=True)
    if (out/'COMPLETED.json').exists():return
    for name in ('common.py','generate.py','score.py','verify.py','direct_loader.py','reference_scoring.py'):
        text=(SOURCE/name).read_text()
        if name=='common.py':text=text.replace("TRAIN=REPO/'runs/qwen3_direct_pt_20261117/training'","TRAIN=REPO/'runs/qwen3_nzd_v2_20261120/training'")
        if name=='direct_loader.py':text=text.replace("base=TRAIN.parent/'verified_base'","base=Path(provenance['model_directory'])").replace('import json','import json\nfrom pathlib import Path',1)
        if name in ('generate.py','score.py'):text=text.replace("'models':3","'models':1")
        if name=='verify.py':text=text.replace("'expected_saved_checkpoints':3,'all_completed':len(checked)==3","'expected_saved_checkpoints':1,'all_completed':len(checked)==1")
        compile(text,str(out/name),'exec');(out/name).write_text(text)
    for name in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl'):shutil.copy2(SOURCE/name,out/name)
    best=json.loads((ROOT/f'training/{arm}/best.json').read_text());cp=Path(best['path'])
    prov=json.loads((ROOT/'BASE_PROVENANCE.json').read_text());key='nzd_v2_'+arm
    spec={'key':key,'arm':key,'label':'Qwen3 nzd v2 '+arm,'step':best['optimizer_step'],'dev_best_step':best['optimizer_step'],
          'checkpoint':str(cp),'weights_sha256':sha(cp/'adapter_model.safetensors'),'base':prov['model_directory'],'adapters':[str(cp)]}
    (out/'results'/key).mkdir(parents=True,exist_ok=True)
    atomic(out/'plan.json',{'checkpoints':[spec],'protocol':json.loads((out/'protocol.json').read_text()),
       'inputs_sha256':{n:sha(out/n) for n in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl')},
       'implementation_sha256':{n:sha(out/n) for n in ('reference_scoring.py','direct_loader.py')},'selection':'dev-best only'})
    for name in ('generate.py','score.py','verify.py'):subprocess.run([sys.executable,'-u',str(out/name)],cwd=REPO,check=True)
    subprocess.run([sys.executable,'-u',str(REPO/'runs/viewer_parse_fox_fix_20261121/future_patch.py'),str(out)],cwd=REPO,check=True)
    subprocess.run([sys.executable,'-u',str(REPO/'runs/no_eos_eval_20261120/evaluate_future.py'),str(out)],cwd=REPO,check=True)
    atomic(out/'COMPLETED.json',{'state':'completed','completion_samples':288,'independent_samples':64})
    atomic(ROOT/f'EVAL_COMPLETED.{arm}.json',{'state':'completed','evaluation':str(out)})
if __name__=='__main__':main(sys.argv[1])
