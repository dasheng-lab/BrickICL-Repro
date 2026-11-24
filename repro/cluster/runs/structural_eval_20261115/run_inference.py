import gc,hashlib,importlib.util,json,lzma,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

ROOT=Path('runs/structural_eval_20261115')
TRAIN=Path('runs/structural_20261114')
OLD=Path('runs/history_first_brick06')
ARMS={'post_pt':('PT_M','M-only PT'),'post_sft':('SFT_M','M-only SFT'),
      'fullg_pt':('PT_G','Full-G PT'),'fullg_sft':('SFT_G','Full-G SFT')}
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)
def load_model(adapters):
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    for adapter in adapters:model=PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()
    model.eval();model.config.use_cache=True
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True,padding_side='left')
    tok.pad_token=tok.eos_token;model.generation_config.pad_token_id=tok.eos_token_id
    return model,tok

def main():
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    assert json.loads((TRAIN/'COMPLETED.json').read_text()).keys()=={'PT_M','SFT_M','PT_G','SFT_G'}
    models={}
    for arm,(key,label) in ARMS.items():
        run=TRAIN/'training'/key
        assert json.loads((run/'COMPLETED.json').read_text())['state']=='completed'
        best=json.loads((run/'best.json').read_text());cp=Path(best['path'])
        adapters=['kulits/BrickNet-0.6B-PT']+(['kulits/BrickNet-0.6B-SFT'] if 'sft' in arm else [])+[str(cp)]
        models[arm]={'label':label,'step':best['optimizer_step'],'adapters':adapters,
                     'checkpoint':str(cp),'weights_sha256':sha(cp/'adapter_model.safetensors'),'dev_best':best}
    for arm,mode in [('frozen_pt','PT'),('frozen_sft','SFT')]:
        models[arm]={'label':'官方 '+mode,'adapters':['kulits/BrickNet-0.6B-PT']+(['kulits/BrickNet-0.6B-SFT'] if mode=='SFT' else [])}
    for arm,step,label in [('plain',1750,'对照 PT'),('masked',2000,'增强 PT')]:
        cp=Path('/home/nzd0420/brickICL/runs/nzd-supertry/train_v2')/arm/f'checkpoint-{step}'
        models[arm]={'label':label,'step':step,'adapters':['kulits/BrickNet-0.6B-PT',str(cp)],'checkpoint':str(cp),
                     'weights_sha256':sha(cp/'adapter_model.safetensors')}
    inputs={}
    for prime in (0,1):
        rows=[r for r in read(OLD/f'p{prime}_triads.jsonl') if r['condition'] in ('original','stretched')]
        assert len(rows)==36
        inputs[prime]={r['id']:r for r in rows}
        (ROOT/f'p{prime}_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan=[]
    for old in read_plan():
        ids=[k for k in old['ids'] if k in inputs[0]]
        if ids:plan.append(dict(old,ids=ids))
    assert sum(len(p['ids']) for p in plan)==36
    captions=[]
    with lzma.open('baseline/bricknet/data/dataset/captions_val.jsonl.xz','rt') as f:
        for _,line in zip(range(64),f):captions.append(json.loads(line)['caption'])
    reference_caps=Path('/home/nzd0420/brickICL/runs/nzd-supertry/report_metrics/physical/captions.jsonl')
    assert [r['caption'] for r in read(reference_caps)]==captions
    (ROOT/'captions.jsonl').write_text(''.join(json.dumps({'caption':s})+'\n' for s in captions))
    protocol={'models':models,'conditions':['original','stretched'],'prime':[0,1],'sources':18,'samples':4,
              'temperature':1.,'top_k':20,'top_p':.95,'completion_caption':False,'rollback_in_main_metrics':False,
              'pose_f1':'first target-length parsed generated parts, color ignored, same_pose tolerance 3 LDU',
              'delta':'mean source-paired sampled long-feature log-odds with +0.5 smoothing, full output',
              'batch_plan':plan,'baseline_refresh':'all eight models rerun on filtered identical batches',
              'physical':{'n':64,'batch_size':32,'batch_seeds':[0,32],'max_new_tokens':4096,'max_parts_scored':100,
                          'PT_prompt':'a','SFT_prompts':'first 64 official validation captions, one sample each'}}
    atomic(ROOT/'protocol.json',protocol)
    order=['post_pt','post_sft','fullg_pt','fullg_sft','frozen_pt','frozen_sft','plain','masked']
    for arm in order:
        spec=models[arm]
        todo=not all((ROOT/f'p{p}.{arm}.jsonl').exists() and len(read(ROOT/f'p{p}.{arm}.jsonl'))==36 for p in (0,1))
        scratch=ROOT/f'{arm}.free.jsonl'
        if not todo and scratch.exists() and len(read(scratch))==64:continue
        print('LOAD',arm,spec['adapters'],flush=True)
        model,tok=load_model(spec['adapters'])
        for prime in (0,1):
            dest=ROOT/f'p{prime}.{arm}.jsonl'
            prior=read(dest) if dest.exists() else []
            done={r['plan_group'] for r in prior}
            for gi,group in enumerate(plan):
                existing=[r for r in prior if r['plan_group']==gi]
                if existing:
                    assert {r['id'] for r in existing}==set(group['ids']) and all(len(r['samples'])==4 for r in existing)
            with dest.open('a',buffering=1) as f:
                for gi,group in enumerate(plan):
                    if gi in done:continue
                    rows=[inputs[prime][key] for key in group['ids']]
                    batch=tok([r['prefix'] for r in rows for _ in range(4)],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                    torch.manual_seed(group['seed']);start=time.time()
                    with torch.inference_mode():
                        outputs=model.generate(**batch,max_new_tokens=group['max_new_tokens'],do_sample=True,temperature=1.,top_k=20,top_p=.95,
                            eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
                    generated=outputs[:,batch.input_ids.shape[1]:].cpu().tolist()
                    for j,row in enumerate(rows):
                        samples=[]
                        for ids in generated[j*4:(j+1)*4]:
                            eos=tok.eos_token_id in ids
                            ids=ids[:ids.index(tok.eos_token_id)] if eos else ids
                            samples.append({'text':tok.decode(ids,skip_special_tokens=True),'eos_emitted':eos,'hit_token_cap':not eos,
                                            'tokens':len(ids)})
                        rec={k:row[k] for k in ('id','source_key','condition','n_prefix','k_short','k_long','expected','prime')}
                        rec.update(arm=arm,adapters=spec['adapters'],samples=samples,plan_group=gi,seed=group['seed'],
                                   max_new_tokens=group['max_new_tokens'],prefix_sha256=hashlib.sha256(row['prefix'].encode()).hexdigest())
                        f.write(json.dumps(rec)+'\n')
                    print(arm,'p'+str(prime),gi+1,'/',len(plan),round(time.time()-start,1),'seconds',flush=True)
                    del outputs,batch;gc.collect();torch.cuda.empty_cache()
            assert len(read(dest))==36
        # Match the official generate.py free-generation protocol exactly, with natural EOS.
        if not scratch.exists() or len(read(scratch))!=64:
            sft='sft' in arm
            prompts=[tok.encode(c,add_special_tokens=False)+tok.encode('\n',add_special_tokens=False) for c in captions] if sft else [tok.encode('a',add_special_tokens=False)]*64
            temp=scratch.with_suffix('.tmp')
            with temp.open('w',buffering=1) as f:
                for start in (0,32):
                    p=prompts[start:start+32];width=max(map(len,p))
                    ids=torch.tensor([[tok.eos_token_id]*(width-len(x))+x for x in p],device='cuda')
                    mask=(ids!=tok.eos_token_id).long();torch.manual_seed(start)
                    with torch.inference_mode():
                        seq=model.generate(ids,attention_mask=mask,max_new_tokens=4096,do_sample=True,temperature=1.,top_k=20,top_p=.95)
                    for i,tokens in enumerate(seq[:,width:].cpu().tolist()):
                        eos=tok.eos_token_id in tokens
                        text=tok.decode(tokens,skip_special_tokens=True)
                        if not sft:text='a'+text
                        f.write(json.dumps({'id':start+i,'sample':0,'text':text,'eos_emitted':eos,'arm':arm,
                                            'caption':captions[start+i] if sft else None,'seed':start,'adapters':spec['adapters']})+'\n')
                    print(arm,'scratch',start+32,'/64',flush=True)
                    del ids,mask,seq;gc.collect();torch.cuda.empty_cache()
            temp.replace(scratch)
        atomic(ROOT/f'GENERATED.{arm}.json',{'state':'generated','completion_samples':288,'scratch_samples':64,'model':spec})
        del model;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/'INFERENCE_COMPLETED.json',{'state':'completed','completion_samples':2304,'scratch_samples':512,'unix_time':time.time()})

def read_plan():return json.loads((OLD/'generation_plan.json').read_text())
if __name__=='__main__':main()
