import fcntl,gc,json,os,time,traceback
from common import *
import torch
from direct_loader import load_model

def status(spec,stage,**kw):
    atomic(ROOT/'inference_status.json',{'state':'running','pid':os.getpid(),'key':spec['key'],'stage':stage,'updated_unix':time.time(),**kw})
def main():
    lock=(ROOT/'inference.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=plan();validate_protocol(p);protocol=p['protocol']
    inputs={prime:{r['id']:r for r in read(ROOT/f'p{prime}_inputs.jsonl')} for prime in (0,1)}
    captions=[r['caption'] for r in read(ROOT/'captions.jsonl')]
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    for spec in p['checkpoints']:
        dest=folder(spec)
        if (dest/'GENERATED.json').exists():continue
        assert sha(Path(spec['checkpoint'])/'adapter_model.safetensors')==spec['weights_sha256']
        begin=time.time();status(spec,'load');print('LOAD',spec['key'],flush=True)
        model,tok=load_model(spec['adapters']);shards=dest/'shards';shards.mkdir(exist_ok=True)
        for prime in (0,1):
            allrows=[]
            for gi,group in enumerate(protocol['batch_plan']):
                shard=shards/f'p{prime}_{gi:02d}.json'
                if shard.exists():records=json.loads(shard.read_text())
                else:
                    status(spec,f'p{prime} group {gi+1}/10',checkpoint_started_unix=begin)
                    rows=[inputs[prime][key] for key in group['ids']]
                    batch=tok([r['prefix'] for r in rows for _ in range(4)],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                    torch.manual_seed(group['seed']);start=time.time()
                    with torch.inference_mode():
                        outputs=model.generate(**batch,max_new_tokens=group['max_new_tokens'],do_sample=True,temperature=1.,top_k=20,top_p=.95,
                            eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
                    generated=outputs[:,batch.input_ids.shape[1]:].cpu().tolist();records=[]
                    for j,row in enumerate(rows):
                        samples=[]
                        for ids in generated[j*4:(j+1)*4]:
                            eos=tok.eos_token_id in ids;ids=ids[:ids.index(tok.eos_token_id)] if eos else ids
                            samples.append({'text':tok.decode(ids,skip_special_tokens=True),'eos_emitted':eos,'hit_token_cap':not eos,'tokens':len(ids)})
                        rec={k:row[k] for k in ('id','source_key','condition','n_prefix','k_short','k_long','expected','prime')}
                        rec.update(arm=spec['key'],adapters=spec['adapters'],samples=samples,plan_group=gi,seed=group['seed'],
                                   max_new_tokens=group['max_new_tokens'],prefix_sha256=hashlib.sha256(row['prefix'].encode()).hexdigest())
                        records.append(rec)
                    atomic(shard,records)
                    print(spec['key'],f'p{prime}',gi+1,round(time.time()-start,1),'seconds',flush=True)
                    del outputs,batch;gc.collect();torch.cuda.empty_cache()
                assert [r['id'] for r in records]==group['ids'] and all(len(r['samples'])==4 for r in records)
                assert all(r['adapters']==spec['adapters'] for r in records)
                allrows.extend(records)
            assert len(allrows)==36;jsonl(dest/f'p{prime}.jsonl',allrows)
        sft=spec['arm'].startswith('SFT')
        prompts=[tok.encode(c,add_special_tokens=False)+tok.encode('\n',add_special_tokens=False) for c in captions] if sft else [tok.encode('a',add_special_tokens=False)]*64
        records=[]
        for start in (0,32):
            shard=shards/f'free_{start:02d}.json'
            if shard.exists():rows=json.loads(shard.read_text())
            else:
                status(spec,f'independent {start+32}/64',checkpoint_started_unix=begin)
                t=time.time();prompts_batch=prompts[start:start+32];width=max(map(len,prompts_batch))
                ids=torch.tensor([[tok.eos_token_id]*(width-len(x))+x for x in prompts_batch],device='cuda')
                mask=(ids!=tok.eos_token_id).long();torch.manual_seed(start)
                with torch.inference_mode():
                    seq=model.generate(ids,attention_mask=mask,max_new_tokens=4096,do_sample=True,temperature=1.,top_k=20,top_p=.95)
                rows=[]
                for i,tokens in enumerate(seq[:,width:].cpu().tolist()):
                    eos=tok.eos_token_id in tokens;text=tok.decode(tokens,skip_special_tokens=True)
                    if not sft:text='a'+text
                    rows.append({'id':start+i,'sample':0,'text':text,'eos_emitted':eos,'arm':spec['key'],
                                 'caption':captions[start+i] if sft else None,'seed':start,'adapters':spec['adapters']})
                atomic(shard,rows);print(spec['key'],'free',start+32,round(time.time()-t,1),'seconds',flush=True)
                del ids,mask,seq;gc.collect();torch.cuda.empty_cache()
            assert [r['id'] for r in rows]==list(range(start,start+32));records.extend(rows)
        jsonl(dest/'free.jsonl',records)
        atomic(dest/'GENERATED.json',{'state':'generated','model':spec,'completion_samples':288,'independent_samples':64,
            'elapsed_seconds':time.time()-begin,'unix_time':time.time(),
            'files_sha256':{name:sha(dest/name) for name in ('p0.jsonl','p1.jsonl','free.jsonl')}})
        print('GENERATED',spec['key'],round(time.time()-begin,1),flush=True)
        del model,tok;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/'INFERENCE_COMPLETED.json',{'state':'completed','models':1,'unix_time':time.time()})
    atomic(ROOT/'inference_status.json',{'state':'completed','updated_unix':time.time()})

if __name__=='__main__':
    try:main()
    except Exception:
        atomic(ROOT/'INFERENCE_FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
