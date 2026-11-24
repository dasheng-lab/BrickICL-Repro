import json,os,shutil
from pathlib import Path
from common import *

SOURCE=Path('/home/jywu2289/RL/runs/v2.1')
START=Path('/home/nzd0420/brickICL/runs/nzd-supertry/train_v2/masked/checkpoint-2000')
def main():
    best=json.loads((SOURCE/'eval/dev/grpo/best.json').read_text())
    chosen=max(best['table'],key=lambda r:r['reward'])
    assert chosen['step']==best['best']['step']==400
    cp=SOURCE/'train/grpo'/f'checkpoint-{chosen["step"]}'
    expected_start=['kulits/BrickNet-0.6B-PT',str(START)]
    merge=json.loads(Path('/home/jywu2289/RL/runs/start_merged/masked/merge_config.json').read_text())
    assert merge['args']['model']=='Qwen/Qwen3-0.6B' and merge['args']['lora']==expected_start
    cfg=json.loads((cp/'adapter_config.json').read_text())
    assert cfg['base_model_name_or_path']=='/home/jywu2289/RL/runs/start_merged/masked'
    assert cfg['r']==32 and cfg['lora_alpha']==32
    for name in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl'):
        shutil.copy2(PREVIOUS/name,ROOT/name)
    shutil.copy2(PREVIOUS/'score_results.py',ROOT/'reference_scoring.py')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(START/'adapter_model.safetensors')==protocol['models']['masked']['weights_sha256']
    access={'state':'ready','checkpoint':str(cp),'best_dev':chosen,'adapters':expected_start+[str(cp)],
            'uid':os.getuid(),'weights_mode':oct((cp/'adapter_model.safetensors').stat().st_mode & 0o777),
            'source_summary':str(SOURCE/'train/grpo/summary.json'),'completion_samples':288,'independent_samples':64}
    try:
        with (cp/'adapter_model.safetensors').open('rb') as f:f.read(8)
    except PermissionError:
        access['state']='blocked_read_permission';atomic(ROOT/'ACCESS_CHECK.json',access)
        print(json.dumps(access,ensure_ascii=False,indent=2),flush=True)
        raise SystemExit(2)
    atomic(ROOT/'ACCESS_CHECK.json',access)
    base=REPO/'runs/qwen3_direct_pt_20261117/verified_base'
    spec={'key':'grpo_pt','arm':'grpo_pt','label':'BrickNet PT + nzd 增强 + GRPO','step':400,'dev_best_step':400,
          'checkpoint':str(cp),'weights_sha256':sha(cp/'adapter_model.safetensors'),'adapters':access['adapters'],
          'base':str(base),'registry_key':'grpo-PT-checkpoint-400','dev_best':chosen,
          'intermediate_weights_sha256':sha(START/'adapter_model.safetensors')}
    folder(spec).mkdir(parents=True,exist_ok=True);atomic(folder(spec)/'model.json',spec)
    atomic(ROOT/'plan.json',{'checkpoints':[spec],'protocol':protocol,
       'inputs_sha256':{n:sha(ROOT/n) for n in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl')},
       'implementation_sha256':{n:sha(ROOT/n) for n in ('reference_scoring.py','direct_loader.py')},
       'selection':'jywu2289 dev reward, checkpoint-400; no test selection',
       'comparison':'Use frozen_pt and masked from structural_eval_20261115, identical completion and independent-generation protocol.'})
    print('Prepared GRPO step400, 288 completion +64 independent samples',flush=True)
if __name__=='__main__':main()
