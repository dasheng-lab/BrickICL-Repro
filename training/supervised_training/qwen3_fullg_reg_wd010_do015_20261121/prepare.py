import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT.parent/'qwen3_direct_pt_20261117'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
    baseline=json.loads((BASELINE/'training/Full_G/config.json').read_text());cfg=json.loads(json.dumps(baseline))
    cfg.update(output_dir=str(ROOT/'training/Full_G_reg'),arm='Full_G_reg',weight_decay=.10,
               experiment='Full-G regularization pilot: only dropout and AdamW weight decay change',
               code_sha256={n:sha(ROOT/n) for n in ('train.py','bricknet_runtime.py','data.py')})
    cfg['lora']['dropout']=.15
    changed={k:{'baseline':baseline[k],'regularized':cfg[k]} for k in baseline if baseline[k]!=cfg[k]}
    assert set(changed)=={'output_dir','arm','weight_decay','lora','experiment','code_sha256'}
    shutil.copy2(BASELINE/'BASE_PROVENANCE.json',ROOT/'BASE_PROVENANCE.json')
    atomic(ROOT/'training/Full_G_reg/config.json',cfg)
    atomic(ROOT/'COMPARISON.json',{'baseline_run':str(BASELINE/'training/Full_G'),'regularized_run':cfg['output_dir'],
        'changed_fields':changed,'hyperparameter_changes':{'lora_dropout':[.05,.15],'adamw_weight_decay':[.01,.10]},
        'previous_regularized_run':str(ROOT.parent/'qwen3_fullg_reg_20261119'),
        'changes_from_previous_regularized':{'lora_dropout':[.10,.15],'adamw_weight_decay':[.05,.10]},
        'same_data_sha256':cfg['cache_sha256'],'same_schedule_sha256':cfg['schedule_sha256'],
        'same_dev_selection_sha256':sha(cfg['dev_selection']),'same_initial_base':True,'same_adapter_initialization_seed':42,
        'regularization_implementation':'LoRA branch dropout and decoupled AdamW weight decay; no explicit L2 added to reported CE.',
        'selection':'Dev-best checkpoint, never test-selected; run same 2188-step budget to compare overfitting curves.',
        'interpretation':'Combined regularization pilot; does not separately identify dropout vs weight-decay contribution.'})
    print('Prepared Full-G: weight_decay=.10, LoRA dropout=.15; same raw Qwen3 base, dataset and schedule',flush=True)
if __name__=='__main__':main()
