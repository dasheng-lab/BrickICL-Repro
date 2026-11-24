"""CPU-only configuration and dependency validation; no model load or GPU work."""
import json
from pathlib import Path
from prepare import ROOT,BASELINE,sha,atomic

cfg=json.loads((ROOT/'training/Full_G_reg/config.json').read_text())
assert cfg['initial_adapters']==[] and cfg['initialization']=='RAW_QWEN3'
assert cfg['lora']=={'rank':64,'alpha':128,'dropout':.15} and cfg['weight_decay']==.10
assert cfg['lr']==2e-4 and cfg['draws']==70000 and cfg['micro_batch']*cfg['gradient_accumulation']==32
base=json.loads((BASELINE/'training/Full_G/config.json').read_text())
for key in ('data_dir','cache_sha256','schedule_sha256','dev_selection','seed','gradient_accumulation','lr','warmup_ratio','eval_steps','max_length'):
    assert cfg[key]==base[key],key
for name,digest in cfg['code_sha256'].items():assert sha(ROOT/name)==digest,name
for p in ROOT.glob('*.py'):compile(p.read_text(),str(p),'exec')
dependency=ROOT.parent/'qwen3_nzd_v2_20261120'
ready=json.loads((dependency/'DATA_READY.json').read_text())
assert ready['train_rows']==70722 and ready['dev_rows']==2133 and ready['steps']==2211
for arm in ('plain','masked'):
    config=json.loads((dependency/f'training/{arm}/config.json').read_text())
    for split in ('train','dev'):
        record=ready['source_files'][arm][split];assert record['matches_bricknet_run']
        assert sha(Path(record['path']))==record['sha256']
    assert config['initial_adapters']==[] and config['draws']==70722
record={'state':'validated','regularization':{'weight_decay':.10,'lora_dropout':.15},
        'same_fullg_data_and_schedule':True,'raw_qwen3_initialization':True,
        'dependency_train_rows_per_arm':70722,'dependency_dev_rows_per_arm':2133,
        'dependency_hashes_match_bricknet_v2':True,'gpu_preflight_deferred_until_dependency_complete':True,
        'will_evaluate_natural_and_no_eos':True,'new_job_started_training':(ROOT/'training/Full_G_reg/MODEL_INITIALIZATION.json').exists()}
atomic(ROOT/'SETUP_VALIDATED.json',record);print(json.dumps(record),flush=True)
