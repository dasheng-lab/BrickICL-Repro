"""Paired 0.6B evaluation of frozen PT/SFT, double-copy PT, and both post-trained adapters."""
import argparse
import gc
import hashlib
import json
import lzma
import os
import time
from collections import defaultdict
from pathlib import Path
from brickicl.posttrain.paths import best_checkpoint
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from bricknet import score_text
from brickicl.eval.match import realize,score
from brickicl.eval.length_probe import stacked_part,leg_stack

ROOT=Path('runs/posttrain_eval06')
ARMS=['frozen_pt','frozen_sft','frozen_pt_double','post_pt','post_sft','legacy_sft_caption']

def load_model(arm):
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    adapters=['kulits/BrickNet-0.6B-PT']
    if arm in ['frozen_sft','post_sft','legacy_sft_caption']:adapters.append('kulits/BrickNet-0.6B-SFT')
    if arm in ['post_pt','post_sft']:
        mode='PT' if arm=='post_pt' else 'SFT'
        adapters.append(str(best_checkpoint(mode,'M-only')))
    for a in adapters:model=PeftModel.from_pretrained(model,a,local_files_only=True).merge_and_unload()
    model.eval();model.config.use_cache=True
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True,padding_side='left')
    tok.pad_token=tok.eos_token;model.generation_config.pad_token_id=tok.eos_token_id
    return model,tok,adapters

@torch.inference_mode()
def logprobs(model,tok,items):
    out={}
    encoded=[]
    for key,prefix,target in items:
        p=tok.encode(prefix,add_special_tokens=False);t=tok.encode(target,add_special_tokens=False)
        assert tok.encode(prefix+target,add_special_tokens=False)==p+t
        encoded.append((key,p,t+[tok.eos_token_id]))
    encoded.sort(key=lambda x:len(x[1])+len(x[2]))
    for start in range(0,len(encoded),4):
        rs=encoded[start:start+4];length=max(len(p)+len(t) for _,p,t in rs)
        ids=torch.full((len(rs),length),tok.eos_token_id,device='cuda',dtype=torch.long)
        mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
        for j,(_,p,t) in enumerate(rs):
            x=p+t;ids[j,:len(x)]=torch.tensor(x,device='cuda');mask[j,:len(x)]=1
            labels[j,len(p):len(x)]=torch.tensor(t,device='cuda')
        shifted=labels[:,1:];keep=(shifted!=-100).any(0).nonzero().flatten()
        logits=model(input_ids=ids[:,:-1],attention_mask=mask[:,:-1],logits_to_keep=keep,use_cache=False).logits
        y=shifted[:,keep]
        ce=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=-100,reduction='none').view(y.shape)
        for j,(key,p,t) in enumerate(rs):
            nll=float(ce[j].sum());out[key]={'logp':-nll,'n_tokens':len(t),'nll':nll/len(t)}
    return out

def prompt_for(row,arm,tok,captions):
    if arm=='frozen_pt_double':return row['demo']+tok.eos_token+row['prefix']
    if arm=='legacy_sft_caption':return captions[int(row['source_key'].split(':')[1])]+'\n'+row['prefix']
    return row['prefix']

def score_completion(row,text,eos,cap):
    refs={k:realize(row['prefix']+row['target_'+k]) for k in ['short','long']}
    values={}
    for kind in ['short','long']:
        s=score(row['prefix'],row['target_'+kind],text,row['n_prefix'],refs[kind]);k=row['k_'+kind]
        values[kind]={'node_f1':2*s.part_hits/(k+s.n_generated),'pose_f1':2*s.placed/(k+s.n_generated),
                      'exact':s.exact,'parse_ok':s.parse_error is None,'n_generated':s.n_generated,'parse_error':s.parse_error}
    v=values[row['expected']];stack=stacked_part(row)
    leg_root=row['n_prefix']-row.get('prime',0)
    ls=leg_stack(row['prefix']+row['target_short'],leg_root,stack)
    ll=leg_stack(row['prefix']+row['target_long'],leg_root,stack)
    actual=leg_stack(row['prefix']+text,leg_root,stack)
    _,_,collisions=score_text(row['prefix']+text)
    return {'text':text,**v,'strict_pose_f1':v['pose_f1'] if v['parse_ok'] else 0.,
       'pose_pref':values['long']['pose_f1']-values['short']['pose_f1'],
       'short_pose_f1':values['short']['pose_f1'],'long_pose_f1':values['long']['pose_f1'],
       'length_abs_error':abs(v['n_generated']-row['k_'+row['expected']]),
       'leg_short':ls,'leg_long':ll,'leg':actual,'leg_is_long':actual==ll,
       'eos_emitted':eos,'hit_token_cap':cap,'collisions':collisions,
       'valid_collision_free':v['parse_ok'] and collisions==[]}

def evaluate_suite(model,tok,arm,adapters,suite,n,captions):
    inputs=[json.loads(s) for s in (ROOT/f'{suite}_triads.jsonl').open()]
    dest=ROOT/f'{suite}.{arm}.jsonl'
    done={r['id'] for r in map(json.loads,dest.open())} if dest.exists() else set()
    rows=[r for r in inputs if r['id'] not in done]
    if not rows:return
    probabilities=logprobs(model,tok,[(r['id']+':'+k,prompt_for(r,arm,tok,captions),r['target_'+k]) for r in rows for k in ['short','long']])
    groups=defaultdict(list)
    for r in rows:groups[64*(r['k_long']+4)+32].append(r)
    with dest.open('a',buffering=1) as out:
        for cap,rs in sorted(groups.items()):
            rs.sort(key=lambda r:len(tok.encode(prompt_for(r,arm,tok,captions),add_special_tokens=False)))
            for start in range(0,len(rs),8):
                chunk=rs[start:start+8]
                prompts=[prompt_for(r,arm,tok,captions) for r in chunk for _ in range(n)]
                batch=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
                seed=1729+int(hashlib.sha256(('|'.join(r['id'] for r in chunk)).encode()).hexdigest()[:8],16)%10000000
                torch.manual_seed(seed)
                begin=time.time()
                with torch.inference_mode():
                    sequences=model.generate(**batch,max_new_tokens=cap,do_sample=True,temperature=1.,top_k=20,top_p=.95,
                                              pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                generated=sequences[:,batch.input_ids.shape[1]:].cpu().tolist()
                for j,r in enumerate(chunk):
                    samples=[]
                    for ids in generated[j*n:(j+1)*n]:
                        eos=tok.eos_token_id in ids
                        actual=ids[:ids.index(tok.eos_token_id)] if eos else ids
                        text=tok.decode(actual,skip_special_tokens=True)
                        samples.append(score_completion(r,text,eos,not eos and len(ids)>=cap))
                    lp_short=probabilities[r['id']+':short'];lp_long=probabilities[r['id']+':long']
                    record={k:r[k] for k in ['id','source_key','source_family','condition','expected','prime','r','n_prefix','k_short','k_long']}
                    record.update(arm=arm,suite=suite,adapters=adapters,log_odds=lp_long['logp']-lp_short['logp'],
                                  lp_short=lp_short,lp_long=lp_long,samples=samples,seed=seed,max_new_tokens=cap)
                    out.write(json.dumps(record)+'\n')
                print(arm,suite,'chunk',len(chunk),'samples',len(generated),'seconds',round(time.time()-begin,1),flush=True)
                del sequences,batch;gc.collect();torch.cuda.empty_cache()

def full_test(model,tok,arm,adapters):
    dest=ROOT/f'full_test.{arm}.jsonl'
    if dest.exists():return
    rows=[]
    for dataset in ['v2','v3']:
        for line in (Path('runs/reference_last_06')/dataset/'test.jsonl').open():
            r=json.loads(line);r['dataset']=dataset;rows.append(r)
    items=[]
    for r in rows:
        prefix=r['prompt']
        if arm=='frozen_pt_double':prefix=r['prompt']+r['completion']+tok.eos_token+prefix
        items.append((r['id'],prefix,r['completion']))
    probs=logprobs(model,tok,items)
    temp=dest.with_suffix('.tmp')
    with temp.open('w') as out:
        for r in rows:out.write(json.dumps({k:r[k] for k in ['id','dataset','source_key','source_family','condition']}|
                              {'arm':arm,**probs[r['id']]})+'\n')
    os.replace(temp,dest);print(arm,'full test',len(rows),'done',flush=True)

def main():
    os.umask(0o022)
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arms',nargs='+',default=ARMS);ap.add_argument('--samples',type=int,default=4)
    args=ap.parse_args();captions={r['id']:r['caption'] for r in map(json.loads,lzma.open('baseline/bricknet/data/dataset/captions_val.jsonl.xz','rt'))}
    for arm in args.arms:
        print('loading',arm,flush=True);model,tok,adapters=load_model(arm)
        (ROOT/f'provenance.{arm}.json').write_text(json.dumps({'arm':arm,'adapters':adapters,'samples':args.samples,
             'temperature':1.,'top_k':20,'top_p':.95,'selection':'dev best checkpoints only','test_used_for_selection':False},indent=2)+'\n')
        if arm!='legacy_sft_caption':
            full_test(model,tok,arm,adapters)
            evaluate_suite(model,tok,arm,adapters,'heldout',args.samples,captions)
        evaluate_suite(model,tok,arm,adapters,'legacy',args.samples,captions)
        del model;gc.collect();torch.cuda.empty_cache()
    print('ALL EVALUATIONS COMPLETED',flush=True)

if __name__=='__main__':main()
