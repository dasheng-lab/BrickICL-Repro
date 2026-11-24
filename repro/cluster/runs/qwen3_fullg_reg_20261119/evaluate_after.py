"""Evaluate the dev-best regularized model once; store results without changing websites."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from prepare import ROOT,BASELINE,atomic

REPO=ROOT.parents[1];SOURCE=REPO/'runs/qwen3_direct_eval_20261119';EVAL=REPO/'runs/qwen3_fullg_reg_eval_20261119'
def main():
    EVAL.mkdir(exist_ok=True)
    if (EVAL/'COMPLETED.json').exists():return
    # Reuse the exact sampler and scorer validated for the three baseline models.
    for name in ('common.py','generate.py','score.py','verify.py','direct_loader.py','reference_scoring.py'):
        text=(SOURCE/name).read_text()
        if name=='common.py':text=text.replace("TRAIN=REPO/'runs/qwen3_direct_pt_20261117/training'","TRAIN=REPO/'runs/qwen3_fullg_reg_20261119/training'")
        if name=='direct_loader.py':text=text.replace("base=TRAIN.parent/'verified_base'","base=Path(provenance['model_directory'])").replace('import json','import json\nfrom pathlib import Path',1)
        if name=='generate.py':text=text.replace("'models':3","'models':1")
        if name=='score.py':text=text.replace("'models':3","'models':1")
        if name=='verify.py':text=text.replace("'expected_saved_checkpoints':3,'all_completed':len(checked)==3","'expected_saved_checkpoints':1,'all_completed':len(checked)==1")
        compile(text,str(EVAL/name),'exec')
        (EVAL/name).write_text(text)
    for name in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl'):shutil.copy2(SOURCE/name,EVAL/name)
    atomic(ROOT/'EVAL_PREPARED.json',{'evaluation':str(EVAL),'state':'waiting_for_training','protocol_source':str(SOURCE)})
    while not (ROOT/'COMPLETED.json').exists():
        if (ROOT/'FAILED.json').exists():raise RuntimeError('Training failed')
        time.sleep(30)
    best=json.loads((ROOT/'training/Full_G_reg/best.json').read_text());cp=Path(best['path'])
    from prepare import sha
    base=json.loads((ROOT/'BASE_PROVENANCE.json').read_text())
    spec={'key':'regularized_g','arm':'regularized_g','label':'Qwen3 direct Full-G · dropout .10 / decay .05',
          'step':best['optimizer_step'],'dev_best_step':best['optimizer_step'],'checkpoint':str(cp),
          'weights_sha256':sha(cp/'adapter_model.safetensors'),'adapters':[str(cp)],'base':base['model_directory']}
    (EVAL/'results'/spec['key']).mkdir(parents=True,exist_ok=True)
    atomic(EVAL/'plan.json',{'checkpoints':[spec],'protocol':json.loads((EVAL/'protocol.json').read_text()),
       'inputs_sha256':{n:sha(EVAL/n) for n in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl')},
       'implementation_sha256':{n:sha(EVAL/n) for n in ('reference_scoring.py','direct_loader.py')},'selection':'dev-best, not test-selected'})
    for name in ('generate.py','score.py','verify.py'):
        subprocess.run([sys.executable,'-u',str(EVAL/name)],cwd=REPO,check=True)
    new=json.loads((EVAL/'results/regularized_g/metrics.json').read_text())
    baseline=json.loads((SOURCE/'results/direct_g/metrics.json').read_text())
    comparison={'baseline':baseline,'regularized':new,'test_used_for_checkpoint_selection':False}
    atomic(EVAL/'comparison.json',comparison);atomic(ROOT/'EVAL_COMPLETED.json',{'state':'completed','evaluation':str(EVAL)})
    atomic(EVAL/'COMPLETED.json',{'state':'completed','completion_samples':288,'independent_samples':64})
    print('Regularization evaluation completed',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'EVAL_FAILED.json',{'traceback':traceback.format_exc()});raise
