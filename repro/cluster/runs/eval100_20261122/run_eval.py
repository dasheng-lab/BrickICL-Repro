import copy,fcntl,gc,hashlib,json,os,time,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,LogitsProcessor
from peft import PeftModel
from reference_scoring import score_row,physical_row,summarize_completion
ROOT=Path(__file__).resolve().parent
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atom(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def jsonl(p,rs):
 tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rs));tmp.replace(p)
class BlockEOS(LogitsProcessor):
 def __init__(self,ids):self.ids=ids
 def __call__(self,input_ids,scores):scores[:,self.ids]=-float('inf');return scores
def main():
 lock=(ROOT/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert j(ROOT/'VISUAL_APPROVED.json')['approved']
 p=j(ROOT/'plan.json');assert sha(ROOT/'p1_inputs.jsonl')==p['input_sha256'] and sha(ROOT/'reference_scoring.py')==p['scoring_sha256']
 inputs={r['id']:r for r in map(json.loads,(ROOT/'p1_inputs.jsonl').open())}
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.cuda.set_per_process_memory_fraction(.35)
 start_all=time.time()
 for mi,spec in enumerate(p['models']):
  arm=spec['arm'];dest=ROOT/'results'/arm;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'VERIFIED.json').exists():continue
  for adapter,digest in spec['local_weight_sha256'].items():assert sha(Path(adapter)/'adapter_model.safetensors')==digest
  begin=time.time();print('LOAD',arm,flush=True)
  model=AutoModelForCausalLM.from_pretrained(spec['base'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
  for adapter in spec['adapters']:model=PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()
  model.eval();model.config.use_cache=True;tok=AutoTokenizer.from_pretrained(spec['base'],local_files_only=True,padding_side='left');tok.pad_token=tok.eos_token;model.generation_config.pad_token_id=tok.eos_token_id
  raw=[]
  for gi,group in enumerate(p['batch_plan']):
   shard=dest/f'completion_{gi:03d}.json'
   if shard.exists():rs=j(shard)
   else:
    atom(ROOT/'status.json',{'state':'running','model':arm,'model_index':mi+1,'models':len(p['models']),'stage':'completion','batch':gi+1,'batches':len(p['batch_plan']),'updated_unix':time.time()})
    rows=[inputs[k] for k in group['ids']];batch=tok([r['prefix'] for r in rows for _ in range(4)],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');torch.manual_seed(group['seed']);t=time.time()
    with torch.inference_mode():out=model.generate(**batch,max_new_tokens=group['max_new_tokens'],do_sample=True,temperature=1.,top_k=20,top_p=.95,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
    seq=out[:,batch.input_ids.shape[1]:].cpu().tolist();rs=[]
    for ri,row in enumerate(rows):
     samples=[]
     for ids in seq[4*ri:4*ri+4]:
      eos=tok.eos_token_id in ids;ids=ids[:ids.index(tok.eos_token_id)] if eos else ids
      samples.append({'text':tok.decode(ids,skip_special_tokens=True),'tokens':len(ids),'eos_emitted':eos,'hit_token_cap':not eos})
     rs.append({k:row[k] for k in ['id','source_key','condition','n_prefix','k_short','k_long','expected','prime']}|{'arm':arm,'adapters':spec['adapters'],'samples':samples,'plan_group':gi,'seed':group['seed'],'max_new_tokens':group['max_new_tokens'],'prefix_sha256':row['prefix_sha256']})
    atom(shard,rs);print(arm,'completion',gi+1,'/',len(p['batch_plan']),round(time.time()-t,1),'s',flush=True);del batch,out
   assert [r['id'] for r in rs]==group['ids'];raw.extend(rs)
  jsonl(dest/'p1.jsonl',raw);assert len(raw)==200
  eos=model.generation_config.eos_token_id;forbidden=sorted(set(([eos] if isinstance(eos,int) else list(eos or []))+[tok.eos_token_id]))
  cfg=copy.deepcopy(model.generation_config);cfg.eos_token_id=None;cfg.forced_eos_token_id=None;cfg.pad_token_id=tok.eos_token_id
  for mode in ['natural','no_eos']:
   records=[]
   for start in (0,32):
    shard=dest/f'{mode}_{start}.json'
    if shard.exists():rs=j(shard)
    else:
     atom(ROOT/'status.json',{'state':'running','model':arm,'model_index':mi+1,'models':len(p['models']),'stage':mode,'batch':start,'updated_unix':time.time()})
     ids=torch.tensor([tok.encode('a',add_special_tokens=False)]*32,device='cuda');torch.manual_seed(start);t=time.time();kwargs={'generation_config':cfg,'logits_processor':[BlockEOS(forbidden)]} if mode=='no_eos' else {}
     with torch.inference_mode():out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=4096,do_sample=True,temperature=1.,top_k=20,top_p=.95,**kwargs)
     rs=[]
     for i,tokens in enumerate(out[:,ids.shape[1]:].cpu().tolist()):
      if mode=='no_eos':assert len(tokens)==4096 and not set(tokens)&set(forbidden)
      rs.append({'id':start+i,'sample':0,'text':'a'+tok.decode(tokens,skip_special_tokens=True),'generated_token_ids':tokens,'eos_emitted':tok.eos_token_id in tokens,'forbidden_eos_ids':forbidden if mode=='no_eos' else [],'arm':arm,'adapters':spec['adapters'],'caption':None,'seed':start})
     atom(shard,rs);print(arm,mode,start,round(time.time()-t,1),'s',flush=True);del ids,out
    records.extend(rs)
   jsonl(dest/('free.jsonl' if mode=='natural' else 'free_no_eos.jsonl'),records)
  del model,tok;gc.collect();torch.cuda.empty_cache()
  atom(ROOT/'status.json',{'state':'running','model':arm,'stage':'scoring','updated_unix':time.time()})
  with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
   scored=list(pool.map(score_row,[(inputs[r['id']],r) for r in raw],chunksize=2));jsonl(dest/'p1.scored.jsonl',scored)
   metrics={'model':spec,'completion':{'1':summarize_completion(scored)},'source_count':100,'legacy_source_count':18,'synthetic_source_count':82}
   for mode,path in [('physical','free'),('no_eos','free_no_eos')]:
    rs=[json.loads(s) for s in (dest/f'{path}.jsonl').open()];phys=list(pool.map(physical_row,rs,chunksize=4));jsonl(dest/f'{path}.scored.jsonl',phys)
    known=[r['collision_free_length'] for r in phys if r['collision_free_length'] is not None]
    metrics[mode]={'samples':64,'valid_length':mean(r['valid_length'] for r in phys),'collision_free_length':mean(known) if known else None,'collision_coverage':len(known),'parse_rate':mean(r['invalid'] is None for r in phys)}
    if mode=='no_eos':metrics[mode]['eos_suppression_verified']=all(len(r['generated_token_ids'])==4096 and not set(r['generated_token_ids'])&set(r['forbidden_eos_ids']) for r in rs);assert metrics[mode]['eos_suppression_verified']
  assert metrics['completion']['1']['samples']==800;metrics['elapsed_seconds']=time.time()-begin;atom(dest/'metrics.json',metrics)
  atom(dest/'VERIFIED.json',{'sources':100,'completion_samples':800,'independent_samples':128,'weights_sha256':spec['local_weight_sha256'],'input_sha256':p['input_sha256'],'output_sha256':{name:sha(dest/name) for name in ['p1.jsonl','p1.scored.jsonl','free.jsonl','free.scored.jsonl','free_no_eos.jsonl','free_no_eos.scored.jsonl']}})
  print('COMPLETE',arm,'seconds',round(time.time()-begin,1),metrics['completion']['1']['f1'],flush=True)
 atom(ROOT/'COMPLETED.json',{'models':len(p['models']),'completion_samples':12000,'independent_samples':1920,'elapsed_seconds':time.time()-start_all});atom(ROOT/'status.json',{'state':'completed','updated_unix':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:atom(ROOT/'FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
