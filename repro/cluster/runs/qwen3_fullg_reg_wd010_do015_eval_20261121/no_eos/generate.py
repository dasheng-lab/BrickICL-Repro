import copy,fcntl,gc,json,os,time,traceback
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,LogitsProcessor
from peft import PeftModel
from common import ROOT,sha,atomic,jsonl

class BlockEOS(LogitsProcessor):
    def __init__(self,ids):self.ids=ids
    def __call__(self,input_ids,scores):
        scores[:,self.ids]=-float('inf');return scores

def main():
    lock=(ROOT/'inference.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text());torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    assert sha(ROOT/'reference_scoring.py')==plan['scoring_sha256']
    for spec in plan['models']:
        dest=ROOT/'results'/spec['key']
        if (dest/'GENERATED.json').exists():continue
        for adapter,digest in spec['local_weight_sha256'].items():assert sha(Path(adapter)/'adapter_model.safetensors')==digest
        begin=time.time();print('LOAD',spec['key'],flush=True)
        model=AutoModelForCausalLM.from_pretrained(spec['base'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
        for adapter in spec['adapters']:model=PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()
        model.eval();model.config.use_cache=True
        tok=AutoTokenizer.from_pretrained(spec['base'],local_files_only=True,padding_side='left');tok.pad_token=tok.eos_token
        eos=model.generation_config.eos_token_id
        forbidden=sorted(set(([eos] if isinstance(eos,int) else list(eos or []))+[tok.eos_token_id]))
        gcfg=copy.deepcopy(model.generation_config);gcfg.eos_token_id=None;gcfg.forced_eos_token_id=None;gcfg.pad_token_id=tok.eos_token_id
        prompt=tok.encode('a',add_special_tokens=False);recs=[]
        for start in (0,32):
            shard=dest/f'batch_{start}.json'
            if shard.exists():rows=json.loads(shard.read_text())
            else:
                atomic(ROOT/'status.json',{'state':'generating','model':spec['key'],'batch':start,'pid':os.getpid(),'updated_unix':time.time()})
                ids=torch.tensor([prompt]*32,device='cuda');mask=torch.ones_like(ids);torch.manual_seed(start);t=time.time()
                with torch.inference_mode():
                    result=model.generate(ids,attention_mask=mask,generation_config=gcfg,
                        max_new_tokens=4096,do_sample=True,temperature=1.,top_k=20,top_p=.95,logits_processor=[BlockEOS(forbidden)])
                generated=result[:,ids.shape[1]:].cpu().tolist();rows=[]
                for i,tokens in enumerate(generated):
                    assert len(tokens)==4096 and not set(forbidden)&set(tokens)
                    rows.append({'id':start+i,'sample':0,'text':'a'+tok.decode(tokens,skip_special_tokens=True),
                        'generated_token_ids':tokens,'tokens':len(tokens),'eos_emitted':False,'forbidden_eos_ids':forbidden,
                        'arm':spec['key'],'seed':start,'base':spec['base'],'adapters':spec['adapters'],'stop_reason':'token_limit'})
                atomic(shard,rows);print(spec['key'],'batch',start,'seconds',round(time.time()-t,1),flush=True)
                del ids,mask,result;gc.collect();torch.cuda.empty_cache()
            assert len(rows)==32 and [r['id'] for r in rows]==list(range(start,start+32));recs.extend(rows)
        jsonl(dest/'free_no_eos.jsonl',recs)
        atomic(dest/'GENERATED.json',{'model':spec,'samples':64,'raw_sha256':sha(dest/'free_no_eos.jsonl'),
            'forbidden_eos_ids':forbidden,'elapsed_seconds':time.time()-begin})
        print('GENERATED',spec['key'],round(time.time()-begin,1),flush=True)
        del model,tok;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/'INFERENCE_COMPLETED.json',{'models':1,'samples':64,'state':'completed'})
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'INFERENCE_FAILED.json',{'traceback':traceback.format_exc()});raise
