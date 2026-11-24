import json,math,time
from pathlib import Path
import torch
from prepare import ROOT,BASELINE,atomic
from bricknet_runtime import load_bricknet_with_fresh_adapter,completion_loss
from train import Corpus
from data import CompletionCollator

torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
cfg=json.loads((ROOT/'training/Full_G_reg/config.json').read_text())
model,tok=load_bricknet_with_fresh_adapter('0.6B',True,'RAW_QWEN3')
dropouts=[module.p for name,module in model.named_modules() if 'lora_dropout' in name and isinstance(module,torch.nn.Dropout)]
assert len(dropouts)==196 and set(dropouts)=={.1}
assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if 'lora_B' in n)
corpus=Corpus(Path(cfg['data_dir']),'dev',tok.eos_token_id);refs=json.loads(Path(cfg['dev_selection']).read_text());collator=CompletionCollator(tok)
model.eval();scores={}
with torch.no_grad():
    for name in ('v2','v3'):
        selected=[r for r in refs if r['dataset']==name];selected.sort(key=lambda r:corpus.indices[name][r['row_index']]['length']);values=[]
        for i in range(0,len(selected),4):
            block=selected[i:i+4];batch={k:v.cuda() for k,v in collator([corpus.item(r) for r in block]).items()}
            value=completion_loss(model,batch).item();values.extend([value]*len(block))
        scores[name]=sum(values)/len(values)
scores['mean']=(scores['v2']+scores['v3'])/2
previous=next(json.loads(s)['dev'] for s in (BASELINE/'training/Full_G/metrics.jsonl').open() if json.loads(s)['event']=='baseline_dev')
assert abs(scores['mean']-previous['mean'])<1e-5,(scores,previous)
model.train();corpus=Corpus(Path(cfg['data_dir']),'train',tok.eos_token_id)
refs=json.loads(next(Path(cfg['schedule']).open()));batch={k:v.cuda() for k,v in collator([corpus.item(r) for r in refs]).items()}
params=[p for p in model.parameters() if p.requires_grad];optim=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
assert optim.param_groups[0]['weight_decay']==.05
start=time.time();loss=completion_loss(model,batch);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optim.step()
assert math.isfinite(loss.item()) and norm>0
atomic(ROOT/'SMOKE_PASSED.json',{'baseline_dev':scores,'matches_original_initial_dev':True,'lora_dropout_layers':len(dropouts),
     'dropout':.1,'weight_decay':.05,'trainable_parameters':sum(p.numel() for p in params),'train_loss':loss.item(),
     'grad_norm':float(norm),'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,'forward_backward_seconds':time.time()-start,
     'formal_training_reinitializes_model':True})
print('CHECK PASSED',scores,loss.item(),flush=True)
