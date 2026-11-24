import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'structural_20261114'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def config(arm,cache,schedule,dev,template):
    cfg=json.loads(template.read_text())
    cfg.update(arm=arm,mode='PT',initialization='RAW_QWEN3',initial_adapters=[],
        output_dir=str(ROOT/'training'/arm),data_dir=str(cache),schedule=str(schedule),schedule_sha256=sha(schedule),
        ready_sha256=sha(cache/'READY.json'),train_input_sha256={v:sha(cache/v/'train.u32') for v in ('v2','v3')},
        cache_sha256=json.loads((cache/'READY.json').read_text())['artifact_sha256'],
        code_sha256={n:sha(ROOT/n) for n in ('train.py','bricknet_runtime.py','data.py')},dev_selection=str(dev),
        gradient_checkpointing=True,lr=2e-4,lora={'rank':64,'alpha':128,'dropout':.05},
        draws=70000,epochs=1,caption=None,experiment='direct Qwen3 PT; 70k examples per dataset; 3 independent initializations')
    atomic(ROOT/'training'/arm/'config.json',cfg)
    return cfg
def main():
    for arm,old in [('M_only','PT_M'),('Full_G','PT_G')]:
        config(arm,OLD/'cache'/old,OLD/'PT_schedule.jsonl',OLD/'PT_dev_selection.json',OLD/'training'/old/'config.json')
    atomic(ROOT/'EXPERIMENT.json',{'base_model':'Qwen/Qwen3-0.6B','initial_adapters':[],
        'arms':['M_only','Full_G','BrickNet'],'lora':{'rank':64,'alpha':128,'dropout':.05},
        'micro_batch':4,'accumulation':8,'effective_batch':32,'max_length':6400,'max_parts':100,
        'examples_per_arm':70000,'steps_per_arm':2188,'learning_rate':2e-4,'warmup_ratio':.03,
        'eval_save_steps':250,'seed':42,'sampler_seed':0,
        'data_policy':'M/G reuse frozen paired structural data. BrickNet uniformly samples shipped first-round paths from official PT+SFT path files, excluding structural dev/test sources; no captions.',
        'comparison_note':'Equal examples and updates, not equal supervised tokens. M-only masks visible context; other arms supervise complete path after initial a.',
        'dev_policy':'M-only uses fixed M dev; Full-G and BrickNet use identical fixed Full-G dev. Each arm dev selects checkpoint; cross-objective losses are not directly comparable.'})
    print('Prepared M-only and Full-G direct-Qwen3 configs',flush=True)
if __name__=='__main__':main()
