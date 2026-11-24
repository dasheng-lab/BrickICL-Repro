import json,shutil
from common import *

def main():
    if (ROOT/'plan.json').exists():validate_protocol(plan());return
    assert set(json.loads((TRAIN.parent/'COMPLETED.json').read_text()))==set(ARMS.values())
    for name in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl'):
        shutil.copy2(PREVIOUS/name,ROOT/name)
    shutil.copy2(PREVIOUS/'score_results.py',ROOT/'reference_scoring.py')
    protocol=json.loads((ROOT/'protocol.json').read_text());specs=[]
    for arm,training in ARMS.items():
        run=TRAIN/training;best=json.loads((run/'best.json').read_text());cp=Path(best['path'])
        assert json.loads((run/'COMPLETED.json').read_text())['state']=='completed'
        assert json.loads((cp/'provenance.json').read_text())['initial_adapters']==[]
        spec={'key':arm,'arm':arm,'label':LABELS[arm],'step':best['optimizer_step'],'dev_best_step':best['optimizer_step'],
            'checkpoint':str(cp),'adapters':[str(cp)],'base':str(TRAIN.parent/'verified_base'),'weights_sha256':sha(cp/'adapter_model.safetensors'),
            'training_arm':training,'dev_best':best,'initialization':'raw Qwen3 only; no BrickNet adapter'}
        specs.append(spec);folder(spec).mkdir(parents=True,exist_ok=True);atomic(folder(spec)/'model.json',spec)
    atomic(ROOT/'plan.json',{'checkpoints':specs,'protocol':protocol,'models':{s['arm']:s for s in specs},
        'inputs_sha256':{n:sha(ROOT/n) for n in ('protocol.json','p0_inputs.jsonl','p1_inputs.jsonl','captions.jsonl')},
        'implementation_sha256':{n:sha(ROOT/n) for n in ('reference_scoring.py','direct_loader.py')},'created_unix':time.time()})
    print('Prepared three dev-best direct-Qwen3 models',flush=True)
if __name__=='__main__':main()
