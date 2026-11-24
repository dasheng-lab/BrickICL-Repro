import gc,json,math,time
import torch
from prepare import ROOT,atomic
from train import Corpus
from data import CompletionCollator
from bricknet_runtime import load_bricknet_with_fresh_adapter,completion_loss

def main():
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
    model,tok=load_bricknet_with_fresh_adapter('0.6B',True,'RAW_QWEN3')
    assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if 'lora_B' in n)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    optim=torch.optim.AdamW(trainable,lr=2e-4);collate=CompletionCollator(tok);results={}
    for arm in ('M_only','Full_G'):
        cfg=json.loads((ROOT/f'training/{arm}/config.json').read_text());data=Corpus(__import__('pathlib').Path(cfg['data_dir']),'train',tok.eos_token_id)
        batches=[json.loads(s) for s in open(cfg['schedule'])]
        rows=[data.item(r) for r in batches[0]]
        for row in rows:
            labels=row['labels'];prefix=next(i for i,x in enumerate(labels) if x!=-100)
            assert labels[-1]==tok.eos_token_id and prefix>=1
            if arm=='Full_G':assert prefix==1
            else:assert prefix>1
        model.train();optim.zero_grad(set_to_none=True);start=time.time();torch.cuda.reset_peak_memory_stats()
        batch={k:v.cuda() for k,v in collate(rows).items()};loss=completion_loss(model,batch);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optim.step()
        assert math.isfinite(loss.item()) and grad>0
        results[arm]={'loss':loss.item(),'grad_norm':float(grad),'max_input_length':batch['input_ids'].shape[1],
                      'seconds_forward_backward':time.time()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
        del batch,data;gc.collect();torch.cuda.empty_cache()
    assert any(torch.count_nonzero(p)>0 for n,p in model.named_parameters() if 'lora_B' in n)
    atomic(ROOT/'SMOKE_PASSED.json',{'base_model':'Qwen/Qwen3-0.6B','initial_adapters':[],
        'trainable_parameters':sum(p.numel() for p in trainable),'total_parameters':sum(p.numel() for p in model.parameters()),
        'tests':results,'training_restarts_from_fresh_initialization':True})
    print(json.dumps(results),flush=True)
if __name__=='__main__':main()
