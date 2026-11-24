import gc,json,lzma,math
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from brickicl.posttrain.bricknet_runtime import load_bricknet_with_fresh_adapter,completion_loss
from brickicl.posttrain.data import CompletionCollator
import prepare_corpus as pc

torch.set_num_threads(2)
pc.TOK=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
pc.CAPS=defaultdict(list)
for line in lzma.open(pc.DATA/'captions_sft.jsonl.xz','rt'):
    r=json.loads(line);pc.CAPS[str(r['id'])].append(r['caption'])
samples=[]
for v in ('v2','v3'):
    for line in Path(f'runs/reference_last_06/{v}/dev.jsonl').read_text().splitlines():
        result,err=pc.transform(line)
        if result:
            samples.append(result[1])
            if len(samples)%2==0:break
results={}
for mode in ('PT','SFT'):
    model,tok=load_bricknet_with_fresh_adapter('0.6B',True,mode)
    collate=CompletionCollator(tok)
    for scope in ('M','G'):
        arm=f'{mode}_{scope}'
        features=[]
        for r in samples:
            ids,n=r[arm]
            features.append(dict(input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]*n+ids[n:]))
        batch={k:v.cuda() for k,v in collate(features).items()}
        model.train();model.zero_grad(set_to_none=True)
        loss=completion_loss(model,batch)
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        results[arm]={'loss':float(loss.detach()),'nonzero_gradient':any(bool(g.ne(0).any()) for g in grads),
                      'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                      'max_batch_tokens':batch['input_ids'].shape[1]}
        assert results[arm]['nonzero_gradient']
        print(arm,results[arm],flush=True)
        model.zero_grad(set_to_none=True)
        del loss,batch,grads
    del model;gc.collect();torch.cuda.empty_cache()
(pc.ROOT/'TRAINING_SMOKE.json').write_text(json.dumps(results,indent=2))
