"""Exploratory generation retention check: no given bricks; all outputs retained."""
import gc
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from bricknet import parse_sample,score_text
from brickicl.posttrain.evaluate_posttrain import load_model
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/from_scratch06'))
    args = parser.parse_args(argv)
    ROOT=args.out;ROOT.mkdir(parents=True,exist_ok=True);os.umask(0o022)
    CAPTIONS=[
     ('fox','狐狸','A LEGO model of a sitting orange fox with pointed ears, a white chest, a black nose and a large bushy tail.'),
     ('car','汽车','A small red LEGO car with four black wheels, a windshield and a compact body.'),
     ('chair','椅子','A LEGO chair with four legs, a flat seat and a tall backrest.'),
     ('robot','机器人','A small LEGO robot with a square torso, two arms, two legs and a head with two round eyes.'),
    ]
    tasks=[{'id':'unconditional','title':'无条件生成','prompt':'a','mode':'unconditional','n':4}]+[
     {'id':key,'title':title,'prompt':caption+'\n','mode':'caption','n':2} for key,title,caption in CAPTIONS]
    (ROOT/'inputs.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2)+'\n')
    arms=['frozen_pt','frozen_sft','post_pt','post_sft']
    protocol={'size':'0.6B','arms':arms,'provided_bricks':0,'unconditional_start':'a (node identifier only)',
              'caption_format':'encode(caption) + [newline_token_id], exactly as official generate.py; no chat template','temperature':1.,'top_k':20,'top_p':.95,
              'max_new_tokens':4096,'stop':'natural EOS or token cap; no forced minimum length',
              'samples_per_arm':12,'verifier_filtering':False,'purpose':'small exploratory generation-retention check, not a large benchmark'}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for arm in arms:
        path=ROOT/f'{arm}.jsonl'
        done=[json.loads(s) for s in path.open()] if path.exists() else []
        if len(done)==12:continue
        if done and (len(done)!=4 or any(r['mode']!='unconditional' for r in done)):
            raise RuntimeError('incomplete output outside supported unconditional-only resume')
        model,tok,adapters=load_model(arm)
        with path.open('a' if done else 'w',buffering=1) as stream:
            for mode in ['unconditional','caption']:
                if mode=='unconditional' and done:continue
                jobs=[(t,i) for t in tasks if t['mode']==mode for i in range(t['n'])]
                seed=20261109+(mode=='caption')
                torch.manual_seed(seed)
                nl_id=tok.encode('\n',add_special_tokens=False)[0]
                token_lists=[tok.encode(t['prompt'][:-1],add_special_tokens=False)+[nl_id] if mode=='caption'
                             else tok.encode(t['prompt'],add_special_tokens=False) for t,i in jobs]
                longest=max(map(len,token_lists))
                ids=torch.tensor([[tok.eos_token_id]*(longest-len(x))+x for x in token_lists],device='cuda')
                mask=torch.tensor([[0]*(longest-len(x))+[1]*len(x) for x in token_lists],device='cuda')
                batch={'input_ids':ids,'attention_mask':mask}
                start=time.time()
                with torch.inference_mode():
                    result=model.generate(**batch,do_sample=True,temperature=1.,top_k=20,top_p=.95,max_new_tokens=4096,
                                          pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                sequences=result[:,longest:].cpu().tolist()
                for (task,i),ids in zip(jobs,sequences):
                    eos=tok.eos_token_id in ids
                    generated=ids[:ids.index(tok.eos_token_id)] if eos else ids
                    completion=tok.decode(generated,skip_special_tokens=True)
                    text=('a'+completion) if mode=='unconditional' else completion
                    parsed=parse_sample(text)
                    try:
                        attempts,invalid,collisions=score_text(text)
                        error=None
                    except Exception as exc:
                        attempts,invalid,collisions=0,None,None;error=repr(exc)
                    record={'arm':arm,'task':task['id'],'title':task['title'],'mode':mode,'sample':i,
                        'prompt':task['prompt'],'text':text,'raw_completion':completion,'adapters':adapters,'seed':seed,
                        'input_tokenization':'official_separate_newline' if mode=='caption' else 'node_identifier_a',
                        'generated_tokens':len(generated),'eos_emitted':eos,'hit_token_cap':not eos and len(ids)>=4096,
                        'n_parts':len(parsed.tree.parts),'parse_error':parsed.error,'parse_ok':parsed.error is None,
                        'nonempty_parse_ok':parsed.error is None and bool(parsed.tree.parts),
                        'n_actions_attempted':attempts,'collisions':collisions,'collision_error':error,
                        'valid_collision_free':parsed.error is None and bool(parsed.tree.parts) and collisions==[]}
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(arm,mode,len(jobs),'done',round(time.time()-start,1),'seconds',flush=True)
                del result,batch;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    summary={}
    for arm in arms:
        rows=[json.loads(s) for s in (ROOT/f'{arm}.jsonl').open()]
        assert len(rows)==12
        summary[arm]={}
        for mode in ['unconditional','caption']:
            rs=[r for r in rows if r['mode']==mode]
            summary[arm][mode]={'samples':len(rs),
                **{k:float(np.mean([r[k] for r in rs])) for k in ['parse_ok','nonempty_parse_ok','valid_collision_free','eos_emitted','hit_token_cap','n_parts']},
                'zero_parsed_parts':sum(r['n_parts']==0 for r in rs),'known_collision_mesh_rate':float(np.mean([r['collisions'] is not None for r in rs]))}
    (ROOT/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
