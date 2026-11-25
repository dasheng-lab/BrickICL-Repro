"""Build paired DFS and subtree-last caches from authorized assembly-order pools."""
import fcntl,json,random,shutil,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from prepare import ROOT,sha,atomic

REPO=ROOT.parents[3];ORDER_DATA=REPO/'runs/assembly_order'
BASE=ROOT.parent/'baseline'

def main():
    lock=(ROOT/'data.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'DATA_READY.json').exists():return
    started=time.time();prov=json.loads((BASE/'BASE_PROVENANCE.json').read_text())
    assert prov['verified'] and sha(Path(prov['model_directory'])/'model.safetensors')==prov['expected_model_sha256']
    shutil.copy2(BASE/'BASE_PROVENANCE.json',ROOT/'BASE_PROVENANCE.json')
    tok=AutoTokenizer.from_pretrained(prov['model_directory'],local_files_only=True)
    source_info={};records={};indices={};roundtrip=0
    for arm in ('plain','masked'):
        cfg=json.loads((ORDER_DATA/f'train_v2/{arm}/run_config.json').read_text());source_info[arm]={}
        for split,pool,expected_count in [('train','sft_train_v2',70722),('dev','sft_dev_v2',2133)]:
            source=ORDER_DATA/pool/(arm+'.jsonl');digest=sha(source)
            assert digest==cfg['train_sha256' if split=='train' else 'dev_sha256'],(arm,split,'source changed since original BrickNet training')
            rs=[json.loads(line) for line in source.open()];assert len(rs)==expected_count
            assert len({r['round'] for r in rs})==1
            records[arm,split]=rs;source_info[arm][split]={'path':str(source),'sha256':digest,'rows':len(rs),
                'sources':len({r['source'] for r in rs}),'matches_bricknet_run':True,'round':rs[0]['round']}
            dest=ROOT/f'cache/{arm}/v2';dest.mkdir(parents=True,exist_ok=True);offset=0;idx=[]
            with (dest/f'{split}.u32').open('wb') as bf,(dest/f'{split}.index.jsonl').open('w') as jf:
                for start in range(0,len(rs),128):
                    chunk=rs[start:start+128];texts=[r['path'].strip(' ') for r in chunk]
                    enc=tok(texts,add_special_tokens=False)['input_ids']
                    for i,(row,text,ids) in enumerate(zip(chunk,texts,enc)):
                        assert row['nodes']<=100 and ids[0]==tok.encode('a',add_special_tokens=False)[0]
                        if (start+i)%4000==0 or (split=='dev' and i==0):assert tok.decode(ids)==text;roundtrip+=1
                        ids=ids+[tok.eos_token_id];assert len(ids)<=6400
                        meta={'id':f'{split}-{start+i}','source_key':f'sft:{row["source"]}',
                            'source_row':start+i,'nodes':row['nodes'],'offset':offset,'length':len(ids),
                            'prefix_length':1,'target_length':len(ids)-1}
                        np.asarray(ids,dtype='<u4').tofile(bf);jf.write(json.dumps(meta)+'\n');idx.append(meta);offset+=len(ids)
            indices[arm,split]=idx
            print('TOKENIZED',arm,split,len(idx),offset,flush=True)
    for split in ('train','dev'):
        for i,(p,m) in enumerate(zip(records['plain',split],records['masked',split])):
            for key in ('source','sample_index','nodes','k'):assert p.get(key)==m.get(key),(split,i,key)
            assert sorted(p['order'])==sorted(m['order'])
    train_sources={r['source'] for r in records['plain','train']};dev_sources={r['source'] for r in records['plain','dev']}
    assert not train_sources&dev_sources
    # Check paired geometry on a reproducible sample, while preserving every source row.
    from bricknet import parse_sample,tree_to_graph,decode_graph
    for i in random.Random(20).sample(range(70722),32):
        p,m=records['plain','train'][i],records['masked','train'][i]
        pt,mt=parse_sample(p['path']),parse_sample(m['path']);assert pt.error is None and mt.error is None
        pm,mm=np.asarray(decode_graph(tree_to_graph(pt.tree))),np.asarray(decode_graph(tree_to_graph(mt.tree)))
        mi={node:j for j,node in enumerate(m['order'])}
        for j,node in enumerate(p['order']):
            assert pt.tree.parts[j].part_id==mt.tree.parts[mi[node]].part_id
            assert np.allclose(pm[j],mm[mi[node]],atol=1e-4,rtol=0)
    lengths=[max(indices[a,'train'][i]['length'] for a in ('plain','masked')) for i in range(70722)]
    order=list(LengthGroupedSampler(batch_size=4,lengths=lengths,generator=torch.Generator().manual_seed(0)))
    assert len(order)==len(set(order))==70722
    batches=[[{'dataset':'v2','row_index':j} for j in order[i:i+4]] for i in range(0,70722,4)]
    schedule=ROOT/'schedule.jsonl';schedule.write_text(''.join(json.dumps(b)+'\n' for b in batches))
    devfile=ROOT/'dev_selection.json';atomic(devfile,[{'dataset':'v2','row_index':i} for i in range(2133)])
    work={};configs={}
    for arm in ('plain','masked'):
        cache=ROOT/'cache'/arm
        artifacts={str(p.relative_to(cache)):sha(p) for p in (cache/'v2').iterdir()}
        atomic(cache/'READY.json',{'status':'ready','cross_split_overlap':0,'artifact_sha256':artifacts,
             'source_files':source_info[arm],'no_filter_no_replay_no_subsampling':True})
        cfg=json.loads((BASE/'training/Full_G/config.json').read_text())
        cfg.update(arm=arm,output_dir=str(ROOT/'training'/arm),data_dir=str(cache),draws=70722,
            schedule=str(schedule),schedule_sha256=sha(schedule),dev_selection=str(devfile),
            ready_sha256=sha(cache/'READY.json'),train_input_sha256={'v2':sha(cache/'v2/train.u32')},cache_sha256=artifacts,
            code_sha256={n:sha(ROOT/n) for n in ('train.py','bricknet_runtime.py','data.py')},
            sampling='Exact full paired-order corpus, one pass, shared length-grouped plain/masked schedule; no replay or extra filtering',
            official_replay_mix=False,experiment='Raw Qwen3 plain/masked on identical v2 source files used by original BrickNet post-training',
            validation='All original 2133 paired dev rows; sample-mean CE; no v2/v3 balancing')
        atomic(ROOT/f'training/{arm}/config.json',cfg);configs[arm]=cfg
        work[arm]={'input_tokens':sum(r['length'] for r in indices[arm,'train']),
            'padded_input_tokens':sum(max(indices[arm,'train'][r['row_index']]['length'] for r in b)*len(b) for b in batches),
            'dev_tokens':sum(r['length'] for r in indices[arm,'dev'])}
    report={'state':'ready','source_files':source_info,'train_rows':70722,'dev_rows':2133,
        'train_sources':len(train_sources),'dev_sources':len(dev_sources),'train_dev_source_overlap':0,
        'steps':2211,'last_step_examples':2,'one_epoch_no_replacement':True,'same_paired_schedule':True,
        'roundtrip_checks':roundtrip,'paired_geometry_checks':32,'workload':work,'elapsed_seconds':time.time()-started,
        'changes_from_prior_qwen':'Only switch to exact full paired-order train/dev; rank64/alpha128/dropout.05/lr2e-4/AdamW wd.01 unchanged.',
        'remaining_differences_from_original_bricknet':'Initialization, rank/alpha/dropout/lr/optimizer/warmup/seed/batching and loss aggregation still differ; this unifies data, not all training hyperparameters.'}
    atomic(ROOT/'DATA_READY.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'DATA_FAILED.json',{'traceback':traceback.format_exc()});raise
