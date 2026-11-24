import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
root=Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122')
spec=next(s for s in json.load(open(root/'plan.json'))['models'] if s['arm']=='plain')
model=AutoModelForCausalLM.from_pretrained(spec['base'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
for a in spec['adapters']:
 path='/home/shensx3017/brickicl_eval/proposal_dlo_100/weights/official' if a=='kulits/BrickNet-0.6B-PT' else a
 model=PeftModel.from_pretrained(model,path,local_files_only=True).merge_and_unload()
model.eval()
tok=AutoTokenizer.from_pretrained(spec['base'],local_files_only=True,padding_side='left')
tok.pad_token=tok.eos_token
row=json.loads(open('/home/shensx3017/brickicl_eval/frontier_testset/p1_inputs.jsonl').readline())
batch=tok([row['prefix']]*4,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
with torch.inference_mode():
 out=model.generate(**batch,max_new_tokens=256,do_sample=True,temperature=1.,top_k=20,top_p=.95,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
print('SMOKE_GENERATED',out.shape,flush=True)
from bricknet import parse_sample
for ids in out[:,batch.input_ids.shape[1]:]:
 text=tok.decode(ids,skip_special_tokens=True)
 p=parse_sample(row['prefix']+text)
 print('SMOKE_PARSED',len(p.tree.parts),p.error,flush=True)
