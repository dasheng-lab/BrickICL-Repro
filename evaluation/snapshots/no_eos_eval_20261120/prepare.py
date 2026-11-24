import json,os,shutil,time
from pathlib import Path
from common import *

def main():
    if (ROOT/'plan.json').exists():return
    old=REPO/'runs/pt_results_reorder_20261120'
    rows=json.loads((old/'delivery/metrics_with_initialization.json').read_text());assert len(rows)==12
    metrics={}
    for run in ('structural_eval_20261115','qwen3_direct_eval_20261119','qwen3_queue_results_20261120','grpo_eval_20261120'):
        metrics.update(json.loads((REPO/'runs'/run/'metrics.json').read_text()))
    base=REPO/'runs/qwen3_direct_pt_20261117/verified_base'
    prov=json.loads((base.parent/'BASE_PROVENANCE.json').read_text());assert sha(base/'model.safetensors')==prov['expected_model_sha256']
    specs=[]
    for row in rows:
        m=metrics[row['arm']]['model'];spec={'key':row['arm'],'label':row['model'],'group':row['group'],
            'base':str(base),'adapters':m['adapters'],'published_row':row,'local_weight_sha256':{},'allow_eos_physical':metrics[row['arm']]['physical']}
        for a in spec['adapters']:
            if a.startswith('/'):spec['local_weight_sha256'][a]=sha(Path(a)/'adapter_model.safetensors')
        specs.append(spec);(ROOT/'results'/spec['key']).mkdir(parents=True,exist_ok=True)
    ref=REPO/'runs/structural_eval_20261115/score_results.py';shutil.copy2(ref,ROOT/'reference_scoring.py')
    paused=REPO/'runs/qwen3_nzd_v2_20261120';state=json.loads((paused/'training/plain/status.json').read_text());pid=state['pid']
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'qwen3_nzd_v2_20261120/train.py' in cmd
    starttime=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
    pause={'state':'paused_for_priority_evaluation','pid':pid,'proc_start_ticks':starttime,'command':cmd,'step':state['optimizer_step'],
           'evaluation':str(ROOT),'resume_after':'all no-EOS metrics verified and Gradio columns published','unix_time':time.time()}
    atomic(ROOT/'PAUSED_TRAINING.json',pause);atomic(paused/'PRIORITY_PAUSE.json',pause)
    atomic(ROOT/'plan.json',{'models':specs,'base_provenance':prov,'samples_per_model':64,'batch_size':32,'batch_seeds':[0,32],
        'max_new_tokens':4096,'max_scored_parts':100,'temperature':1.0,'top_k':20,'top_p':.95,'prompt':'a',
        'forbid_eos':'Suppress union of tokenizer EOS and model generation-config EOS IDs, disable forced EOS and EOS stopping. All samples run to the same 4096-token cap.',
        'scoring_sha256':sha(ROOT/'reference_scoring.py'),'source_table_sha256':sha(old/'delivery/metrics_with_initialization.json'),
        'initial_scope':'12 currently published PT methods, each dev-best; future newly trained models need this protocol too',
        'new_regularization_training_authorized':False})
    print('Prepared 12 PT models x64 independent samples; EOS suppression',flush=True)
if __name__=='__main__':main()
