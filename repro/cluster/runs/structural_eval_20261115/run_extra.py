"""Refresh displayed caption examples and the two post-trained PT rollback arms."""
import gc,importlib.util,json,time
from pathlib import Path
import torch
from run_inference import ROOT,read,load_model,atomic
from score_results import score_row,physical_row

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    while not (ROOT/'INFERENCE_COMPLETED.json').exists():time.sleep(20)
    while not (ROOT/'SCORING_COMPLETED.json').exists():time.sleep(10)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    old_viewer=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
    manifest=json.loads((old_viewer/'models/manifest.json').read_text())
    captions={}
    for tag in ('fox','car','chair','robot'):
        entry=next(r for r in manifest if r.get('file')==f'fullg_generation/{tag}_frozen_sft_s1.mpd')
        captions[tag]=entry['caption']
    atomic(ROOT/'gallery_prompts.json',captions)
    for arm in ('frozen_sft','post_sft','fullg_sft'):
        path=ROOT/f'gallery.{arm}.jsonl'
        if path.exists() and len(read(path))==8:continue
        model,tok=load_model(protocol['models'][arm]['adapters'])
        prompts=[(tag,caption,i) for tag,caption in captions.items() for i in range(2)]
        arrays=[tok.encode(c,add_special_tokens=False)+tok.encode('\n',add_special_tokens=False) for _,c,_ in prompts]
        width=max(map(len,arrays))
        ids=torch.tensor([[tok.eos_token_id]*(width-len(a))+a for a in arrays],device='cuda')
        torch.manual_seed(20261110)
        with torch.inference_mode():
            seq=model.generate(ids,attention_mask=(ids!=tok.eos_token_id).long(),max_new_tokens=4096,do_sample=True,
                               temperature=1.,top_k=20,top_p=.95)
        records=[]
        for (tag,caption,i),tokens in zip(prompts,seq[:,width:].cpu().tolist()):
            text=tok.decode(tokens,skip_special_tokens=True)
            rec={'id':tag,'sample':i,'text':text,'caption':caption,'arm':arm,'eos_emitted':tok.eos_token_id in tokens,
                 'seed':20261110,'adapters':protocol['models'][arm]['adapters']}
            records.append(physical_row(rec))
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        print('GALLERY',arm,'8 complete',flush=True)
        del model,ids,seq;gc.collect();torch.cuda.empty_cache()
    spec=importlib.util.spec_from_file_location('rollback_base',Path('runs/pt_rollback_20261111/run_conditions.py'))
    rb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rb)
    sources=(114,95,258,458)
    inputs={r['id']:r for r in read(ROOT/'p1_inputs.jsonl') if int(r['source_key'].split(':')[1]) in sources}
    assert len(inputs)==8
    for arm in ('frozen_pt','post_pt','fullg_pt'):
        path=ROOT/f'rollback.{arm}.jsonl';existing=read(path) if path.exists() else []
        done={(r['id'],r['sample_index']) for r in existing}
        if len(done)==32:continue
        if arm=='frozen_pt':
            legacy=[]
            for cond,folder in [('original','pt_rollback_normal_20261111'),('stretched','pt_rollback_20261111')]:
                legacy.extend(read(Path('runs')/folder/'frozen_pt.jsonl'))
            assert len(legacy)==32
            with path.open('w') as f:
                for r in legacy:
                    row=inputs[r['id']]
                    updated=score_row((row,{'samples':[r['sample']]}))['samples'][0]
                    f.write(json.dumps(r|{'sample':updated,'reused_official':True})+'\n')
            print('ROLLBACK frozen_pt official retained and rescored',flush=True);continue
        model,tok=load_model(protocol['models'][arm]['adapters'])
        baselines={r['id']:r for r in read(ROOT/f'p1.{arm}.scored.jsonl')}
        with path.open('a',buffering=1) as f:
            for source in sources:
                for condition in ('original','stretched'):
                    row=inputs[f'val:{source}:{condition}'];base=baselines[row['id']]
                    for i in range(4):
                        if (row['id'],i) in done:continue
                        seed=202611110+sources.index(source)*100+i
                        s=rb.completion(model,tok,row,seed,base['max_new_tokens'])
                        s=score_row((row,{'samples':[s]}))['samples'][0]
                        record={'id':row['id'],'sample_index':i,'arm':arm,'sample':s,'baseline':base['samples'][i],
                                'prefix_parts':row['n_prefix'],'target_parts':row['k_'+row['expected']],
                                'prefix_collisions':row['prefix_collisions'],'max_new_tokens':base['max_new_tokens'],
                                'adapters':protocol['models'][arm]['adapters']}
                        f.write(json.dumps(record)+'\n')
                        print('ROLLBACK',arm,row['id'],i+1,s['termination'],s['pose_f1'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/'EXTRA_COMPLETED.json',{'gallery_samples':24,'new_rollback_samples':64,'rescored_official_rollback_samples':32})
if __name__=='__main__':main()
