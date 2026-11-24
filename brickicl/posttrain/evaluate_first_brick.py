"""Six matched models, two prime settings, fixed 18-source triples and common draw plan."""
import gc,hashlib,json,os,time
from pathlib import Path
import torch
from peft import PeftModel
from brickicl.posttrain.paths import best_checkpoint
from brickicl.posttrain import evaluate_posttrain as ev
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/history_first_brick06'))
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args(argv)
    ROOT=args.run_dir;os.umask(0o022)
    def read(p):return [json.loads(s) for s in p.open()]
    protocol=json.loads((ROOT/'protocol.json').read_text());plan=json.loads((ROOT/'generation_plan.json').read_text())
    for p in (0,1):
        assert hashlib.sha256((ROOT/f'p{p}_triads.jsonl').read_bytes()).hexdigest()==protocol['input_sha256'][f'p{p}'], 'Input data changed'
    inputs={p:{r['id']:r for r in read(ROOT/f'p{p}_triads.jsonl')} for p in (0,1)}
    def complete(p,arm):
        path=ROOT/f'p{p}.{arm}.jsonl'
        if not path.exists():return False
        rows=read(path)
        return len(rows)==54 and {r['id'] for r in rows}==set(inputs[p]) and all(len(r['samples'])==4 for r in rows)

    if args.validate_only:
        assert all(complete(p,a) for p in (0,1) for a in protocol['arms']), 'Incomplete evaluation'
        print('All six models and both prime settings are complete'); return
    for arm in protocol['arms']:
        if all(complete(p,arm) for p in (0,1)):continue
        print('LOAD',arm,flush=True)
        if arm.startswith('fullg'):
            mode='PT' if arm.endswith('_pt') else 'SFT'
            model,tok,adapters=ev.load_model('frozen_pt' if mode=='PT' else 'frozen_sft')
            checkpoint=str(best_checkpoint(mode,'Full-G'))
            model=PeftModel.from_pretrained(model,checkpoint,local_files_only=True).merge_and_unload();adapters=adapters+[checkpoint]
        else:model,tok,adapters=ev.load_model(arm)
        model.eval();model.config.use_cache=True
        provenance=json.loads((ROOT/f'provenance.{arm}.json').read_text());assert adapters==provenance['adapters'],arm
        for prime in (0,1):
            if complete(prime,arm):continue
            dest=ROOT/f'p{prime}.{arm}.jsonl';existing=read(dest) if dest.exists() else []
            assert len(existing)==len({r['id'] for r in existing})
            old={r['id']:r for r in existing};done=set()
            for group in plan:
                if all(k in old for k in group['ids']):done.update(group['ids'])
            if len(done)!=len(existing):dest.write_text(''.join(json.dumps(r)+'\n' for r in existing if r['id'] in done))
            rows=list(inputs[prime].values())
            probs=ev.logprobs(model,tok,[(r['id']+':'+k,r['prefix'],r['target_'+k]) for r in rows for k in ['short','long']])
            with dest.open('a',buffering=1) as stream:
                for gi,group in enumerate(plan):
                    if all(k in done for k in group['ids']):continue
                    chunk=[inputs[prime][k] for k in group['ids']]
                    batch=tok([r['prefix'] for r in chunk for _ in range(4)],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
                    seed=group['seed'];cap=group['max_new_tokens'];torch.manual_seed(seed);start=time.time()
                    with torch.inference_mode():
                        seq=model.generate(**batch,max_new_tokens=cap,do_sample=True,temperature=1.,top_k=20,top_p=.95,
                            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                    ids=seq[:,batch.input_ids.shape[1]:].cpu().tolist()
                    for j,r in enumerate(chunk):
                        samples=[]
                        for tokens in ids[j*4:(j+1)*4]:
                            eos=tok.eos_token_id in tokens;actual=tokens[:tokens.index(tok.eos_token_id)] if eos else tokens
                            sample=ev.score_completion(r,tok.decode(actual,skip_special_tokens=True),eos,not eos and len(tokens)>=cap)
                            coll=sample['collisions'];new_coll=None if coll is None else [i for i in coll if i>=r['n_prefix']]
                            sample.update(generated_collision_indices=new_coll,generation_collision_free=sample['parse_ok'] and new_coll==[])
                            samples.append(sample)
                        a,b=probs[r['id']+':short'],probs[r['id']+':long']
                        out={k:r[k] for k in ['id','source_key','source_family','condition','expected','prime','r','n_prefix','k_short','k_long']}
                        out.update(arm=arm,suite=f'p{prime}',adapters=adapters,samples=samples,lp_short=a,lp_long=b,log_odds=b['logp']-a['logp'],
                            seed=seed,max_new_tokens=cap,plan_group=gi,prefix_sha256=r['prefix_sha256'],prefix_collisions=r['prefix_collisions'])
                        stream.write(json.dumps(out)+'\n')
                    print(arm,f'p{prime}',f'group {gi+1}/{len(plan)}',f'{len(chunk)*4} samples',round(time.time()-start,1),'seconds',flush=True)
                    del seq,batch;gc.collect();torch.cuda.empty_cache()
            assert complete(prime,arm)
        del model;gc.collect();torch.cuda.empty_cache()
    assert all(complete(p,a) for p in (0,1) for a in protocol['arms'])
    (ROOT/'COMPLETED.json').write_text(json.dumps({'state':'completed','sources':18,'models':6,'conditions':3,'samples_per_input':4,
        'total_generations':2592,'reused_p1_generations':216*sum(v['reused'] for v in protocol['p1_reuse'].values()),'finished_unix':time.time()},indent=2)+'\n')
    print('UNIFIED FIRST-BRICK EVALUATION COMPLETED',flush=True)

if __name__ == '__main__':
    main()
