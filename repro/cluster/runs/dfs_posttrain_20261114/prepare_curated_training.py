import hashlib,json,math,shutil
from pathlib import Path
import torch
from transformers.trainer_pt_utils import LengthGroupedSampler

ROOT=Path('runs/dfs_curated_20261114').resolve()
OLD=Path('runs/dfs_posttrain_20261114').resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (ROOT/'DATA_READY.json').exists()
    for name in ('train.py','bricknet_runtime.py','data.py'):
        shutil.copyfile(OLD/name,ROOT/name)
    summary={}
    for mode in ('PT','SFT'):
        records={}
        for scope in ('M','G'):
            records[scope]={v:[json.loads(s) for s in (ROOT/f'cache/{mode}_{scope}/{v}/train.index.jsonl').open()] for v in ('v2','v3')}
        refs=[];lengths=[]
        for v in ('v2','v3'):
            assert [r['id'] for r in records['M'][v]]==[r['id'] for r in records['G'][v]]
            for i,r in enumerate(records['M'][v]):
                refs.append({'dataset':v,'row_index':i})
                lengths.append(max(r['length'],records['G'][v][i]['length']))
        sampler=LengthGroupedSampler(batch_size=4,lengths=lengths,generator=torch.Generator().manual_seed(0))
        order=list(sampler)
        assert len(order)==len(refs) and set(order)==set(range(len(refs)))
        batches=[[refs[k] for k in order[j:j+4]] for j in range(0,len(order),4)]
        schedule=ROOT/f'{mode}_schedule.jsonl'
        schedule.write_text(''.join(json.dumps(b)+'\n' for b in batches))
        dev=[]
        for v in ('v2','v3'):
            ds={s:[json.loads(l) for l in (ROOT/f'cache/{mode}_{s}/{v}/dev.index.jsonl').open()] for s in ('M','G')}
            assert [r['id'] for r in ds['M']]==[r['id'] for r in ds['G']] and ds['M']
            # A fixed dev subset preserves the previous four-arm model-selection budget.
            generator=torch.Generator().manual_seed(0)
            chosen=torch.randperm(len(ds['M']),generator=generator).tolist()[:128]
            dev.extend({'dataset':v,'row_index':i} for i in chosen)
        devfile=ROOT/f'{mode}_dev_selection.json';devfile.write_text(json.dumps(dev))
        for scope in ('M','G'):
            arm=f'{mode}_{scope}';cache=ROOT/'cache'/arm
            cfg=json.loads((OLD/f'training/{arm}/config.json').read_text())
            cfg.update(output_dir=str(ROOT/'training'/arm),data_dir=str(cache),draws=len(refs),
                schedule=str(schedule),schedule_sha256=sha(schedule),ready_sha256=sha(cache/'READY.json'),
                train_input_sha256={v:sha(cache/v/'train.u32') for v in ('v2','v3')},
                cache_sha256=json.loads((cache/'READY.json').read_text())['artifact_sha256'],
                code_sha256={n:sha(ROOT/n) for n in ('train.py','bricknet_runtime.py','data.py')},
                dev_selection=str(devfile),sampling='One epoch; transformers LengthGroupedSampler; no replacement; sampler seed 0; paired M/G row order',
                sampler_seed=0,epochs=1,sampler_lengths='maximum of paired M/G token lengths',
                official_replay_mix=False,
                caption='screened official description or manually visually reviewed description plus verified structural specification' if mode=='SFT' else None)
            for key in ('length_bins','length_reference_files','length_reference_scope'):cfg.pop(key,None)
            folder=ROOT/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
            (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        summary[mode]={'v2':len(records['M']['v2']),'v3':len(records['M']['v3']),
            'draws':len(refs),'unique_rows':len(set((r['dataset'],r['row_index']) for r in refs)),
            'optimizer_steps':math.ceil(len(refs)/32),'row_coverage':1.0,'seed':0,
            'sampler':'transformers.trainer_pt_utils.LengthGroupedSampler','shared_M_G_schedule':True,
            'extra_official_replay':False}
    (ROOT/'TRAINING_READY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
