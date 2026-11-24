"""Automatic small from-zero check after both whole-G training jobs finish."""
import gc,json,os
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from bricknet import parse_sample,score_text
from brickicl.posttrain.evaluate_posttrain import load_model
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/bricknet_06_fullg'))
    parser.add_argument('--inputs',type=Path,default=Path('runs/from_scratch06/inputs.json'))
    args = parser.parse_args(argv)
    ROOT=args.run_dir;OUT=ROOT/'generation_eval';OUT.mkdir(exist_ok=True);os.umask(0o022)
    tasks=json.loads(args.inputs.read_text())
    for mode in ['PT','SFT']:
        out=OUT/f'{mode}.jsonl'
        if out.exists():
            assert len(out.read_text().splitlines())==12,'partial evaluation needs explicit recovery'
            continue
        model,tok,adapters=load_model('frozen_pt' if mode=='PT' else 'frozen_sft')
        checkpoint=json.loads((ROOT/mode/'best.json').read_text())['path']
        model=PeftModel.from_pretrained(model,checkpoint,local_files_only=True).merge_and_unload();model.eval()
        with out.open('w',buffering=1) as stream:
            for prompt_mode in ['unconditional','caption']:
                jobs=[(t,i) for t in tasks if t['mode']==prompt_mode for i in range(t['n'])]
                seed=20261109+(prompt_mode=='caption');torch.manual_seed(seed)
                nl=tok.encode('\n',add_special_tokens=False);assert len(nl)==1
                lists=[tok.encode(t['prompt'][:-1],add_special_tokens=False)+nl if prompt_mode=='caption'
                       else tok.encode(t['prompt'],add_special_tokens=False) for t,i in jobs]
                length=max(map(len,lists))
                ids=torch.tensor([[tok.eos_token_id]*(length-len(x))+x for x in lists],device='cuda')
                mask=torch.tensor([[0]*(length-len(x))+[1]*len(x) for x in lists],device='cuda')
                with torch.inference_mode():
                    output=model.generate(input_ids=ids,attention_mask=mask,do_sample=True,temperature=1.,top_k=20,top_p=.95,
                                          max_new_tokens=4096,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                for (task,i),generated in zip(jobs,output[:,length:].cpu().tolist()):
                    eos=tok.eos_token_id in generated
                    if eos:generated=generated[:generated.index(tok.eos_token_id)]
                    text=tok.decode(generated,skip_special_tokens=True)
                    if prompt_mode=='unconditional':text='a'+text
                    parsed=parse_sample(text);attempted,invalid,collisions=score_text(text)
                    stream.write(json.dumps({'model':'fullG_'+mode,'mode':prompt_mode,'task':task['id'],'sample':i,
                        'prompt':task['prompt'],'text':text,'seed':seed,'checkpoint':checkpoint,
                        'n_parts':len(parsed.tree.parts),'parse_ok':parsed.error is None and bool(parsed.tree.parts),
                        'parse_error':parsed.error,'collisions':collisions,
                        'valid_collision_free':parsed.error is None and bool(parsed.tree.parts) and collisions==[],
                        'eos_emitted':eos,'hit_token_cap':not eos and len(generated)>=4096})+'\n')
                print(mode,prompt_mode,'done',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    summary={}
    for mode in ['PT','SFT']:
        rows=[json.loads(s) for s in (OUT/f'{mode}.jsonl').open()]
        summary[mode]={m:{'samples':sum(r['mode']==m for r in rows),
            **{k:float(np.mean([r[k] for r in rows if r['mode']==m])) for k in ['n_parts','parse_ok','valid_collision_free','eos_emitted','hit_token_cap']}}
            for m in ['unconditional','caption']}
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
