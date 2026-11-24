"""Raw Qwen3 PT on the exact nzd v2 plain/masked training and validation files."""
import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from bricknet_runtime import load_bricknet_with_fresh_adapter,completion_loss
from data import CompletionCollator

def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

class Corpus:
    def __init__(self,root,split,eos):
        self.eos=eos;self.indices={};self.tokens={}
        for name in ('v2',):
            self.indices[name]=[json.loads(line) for line in (root/name/f'{split}.index.jsonl').open()]
            self.tokens[name]=np.memmap(root/name/f'{split}.u32',mode='r',dtype='<u4')
    def item(self,ref):
        name=ref['dataset'];r=self.indices[name][ref['row_index']]
        ids=self.tokens[name][r['offset']:r['offset']+r['length']].astype(np.int64).tolist()
        n=r['prefix_length'];assert ids[-1]==self.eos and 0<n<len(ids)
        return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*n+ids[n:]}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETED.json').exists():print('already completed',flush=True);return
    schedule_path=Path(cfg['schedule'])
    assert hashlib.sha256(schedule_path.read_bytes()).hexdigest()==cfg['schedule_sha256']
    data=Path(cfg['data_dir']);assert hashlib.sha256((data/'READY.json').read_bytes()).hexdigest()==cfg['ready_sha256']
    ready=json.loads((data/'READY.json').read_text());assert ready['status']=='ready' and ready['cross_split_overlap']==0
    for name,expected in cfg['train_input_sha256'].items():
        assert hashlib.sha256((data/name/'train.u32').read_bytes()).hexdigest()==expected
    for name,expected in cfg['cache_sha256'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==expected,name
    for name,expected in cfg['code_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==expected,name
    batches=[json.loads(s) for s in schedule_path.open()];dev_refs=json.loads(Path(cfg['dev_selection']).read_text())
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    atomic_json(out/'status.json',{'state':'loading','mode':cfg['mode'],'pid':os.getpid(),'optimizer_step':0})
    atomic_json(out/'config.json',cfg)
    assert cfg['initial_adapters']==[] and cfg['initialization']=='RAW_QWEN3'
    assert cfg['lora']=={'rank':64,'alpha':128,'dropout':.05}
    from prepare import sha
    base=json.loads((Path(__file__).parent/'BASE_PROVENANCE.json').read_text())
    assert base['verified'] and sha(Path(base['model_directory'])/'model.safetensors')==base['expected_model_sha256']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    model,tok=load_bricknet_with_fresh_adapter('0.6B',cfg['gradient_checkpointing'],cfg['initialization'])
    model.train();trainable=[p for p in model.parameters() if p.requires_grad]
    atomic_json(out/'MODEL_INITIALIZATION.json',{'base_model':'Qwen/Qwen3-0.6B','initial_adapters':[],
        'trainable_parameters':sum(p.numel() for p in trainable),'total_parameters':sum(p.numel() for p in model.parameters()),
        'lora':cfg['lora'],'initialization':'RAW_QWEN3','base_provenance':base})
    optimizer=torch.optim.AdamW(trainable,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    steps=math.ceil(len(batches)/cfg['gradient_accumulation'])
    scheduler=get_cosine_schedule_with_warmup(optimizer,math.ceil(steps*cfg['warmup_ratio']),steps)
    train=Corpus(data,'train',tok.eos_token_id);dev=Corpus(data,'dev',tok.eos_token_id);collator=CompletionCollator(tok)
    start_micro=0;step=0;best=float('inf');start_time=time.time()
    if args.resume and (out/'latest.json').exists():
        latest=json.loads((out/'latest.json').read_text());cp=Path(latest['path'])
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        set_peft_model_state_dict(model,load_peft_weights(str(cp),device='cuda'))
        state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False)
        assert state['schedule_sha256']==cfg['schedule_sha256']
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['random_rng']);np.random.set_state(state['numpy_rng'])
        start_micro=state['next_micro'];step=state['step'];best=state['best']
    elif (out/'metrics.jsonl').exists():raise RuntimeError('existing run; use --resume')
    def batch(refs,corpus):return {k:v.cuda() for k,v in collator([corpus.item(r) for r in refs]).items()}
    def evaluate():
        model.eval();totals={'v2':[]}
        with torch.no_grad():
            for name in totals:
                selected=[r for r in dev_refs if r['dataset']==name]
                selected.sort(key=lambda r:dev.indices[name][r['row_index']]['length'])
                for i in range(0,len(selected),cfg['micro_batch']):
                    refs=selected[i:i+cfg['micro_batch']];loss=completion_loss(model,batch(refs,dev)).item()
                    if not math.isfinite(loss):raise RuntimeError('nonfinite dev loss')
                    totals[name].extend([loss]*len(refs))
        model.train()
        values={name:sum(x)/len(x) for name,x in totals.items()};values['mean']=values['v2']
        return values
    def checkpoint(next_micro,score):
        nonlocal best
        is_best=score['mean']<best
        if is_best:best=score['mean']
        cp=out/f'checkpoint-{step:06d}';cp.mkdir(exist_ok=True)
        model.save_pretrained(cp,safe_serialization=True)
        state={'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'next_micro':next_micro,
               'best':best,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
               'numpy_rng':np.random.get_state(),'random_rng':random.getstate(),'schedule_sha256':cfg['schedule_sha256']}
        torch.save(state,cp/'training_state.pt')
        atomic_json(cp/'provenance.json',{'initial_adapters':cfg['initial_adapters'],'base_model':'Qwen/Qwen3-0.6B',
                    'load_instruction':'Load raw Qwen/Qwen3-0.6B, then this adapter only. Do not load BrickNet PT/SFT adapters.',
                    'dev':score,'optimizer_step':step,'schedule_sha256':cfg['schedule_sha256'],'base_provenance':base})
        atomic_json(out/'latest.json',{'path':str(cp),'optimizer_step':step,'next_micro':next_micro})
        if is_best:atomic_json(out/'best.json',{'path':str(cp),'optimizer_step':step,'dev_mean':best})
    with (out/'metrics.jsonl').open('a',buffering=1) as metrics:
        if step==0:
            baseline=evaluate();metrics.write(json.dumps({'event':'baseline_dev','optimizer_step':0,'dev':baseline})+'\n')
            print('baseline_dev',baseline,flush=True)
        optimizer.zero_grad(set_to_none=True);loss_sum=0.;examples=0
        for j in range(start_micro,len(batches)):
            group_start=(j//cfg['gradient_accumulation'])*cfg['gradient_accumulation']
            group_end=min(group_start+cfg['gradient_accumulation'],len(batches))
            group_examples=sum(len(batches[x]) for x in range(group_start,group_end))
            refs=batches[j];loss=completion_loss(model,batch(refs,train))
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite training loss at micro {j}')
            value=loss.item();(loss*len(refs)/group_examples).backward();loss_sum+=value*len(refs);examples+=len(refs)
            if j+1!=group_end:continue
            norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            record={'event':'train','optimizer_step':step,'total_steps':steps,'next_micro':j+1,
                    'loss':loss_sum/examples,'grad_norm':float(norm),'lr':scheduler.get_last_lr()[0],
                    'elapsed_seconds':time.time()-start_time,'examples_completed':sum(len(b) for b in batches[:j+1]),
                    'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
            metrics.write(json.dumps(record)+'\n');atomic_json(out/'status.json',dict(record,state='training',mode=cfg['mode'],pid=os.getpid()))
            if step==1 or step%10==0:print(json.dumps(record),flush=True)
            loss_sum=0.;examples=0
            if step%cfg['eval_steps']==0 or j+1==len(batches):
                score=evaluate();checkpoint(j+1,score)
                metrics.write(json.dumps({'event':'dev','optimizer_step':step,'dev':score})+'\n');print('dev',step,score,flush=True)
        completion={'state':'completed','mode':cfg['mode'],'optimizer_steps':step,'draws':cfg['draws'],
                    'best_dev':best,'elapsed_seconds':time.time()-start_time,'test_used':False}
        atomic_json(out/'COMPLETED.json',completion);atomic_json(out/'status.json',completion);print(json.dumps(completion),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
