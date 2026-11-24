"""Materialize and fully validate the authorized reference-last training corpus."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from bricknet import parse_sample,tree_to_graph,decode_graph,score_text
from transformers import AutoTokenizer
from brickicl.posttrain.reference_last import sample_reference_last,connected_subsets

TOKENIZER=None
def initialize():
    global TOKENIZER
    TOKENIZER=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)

def transform(line):
    row=json.loads(line)
    try:
        original=parse_sample(row['prompt']+row['completion'])
        assert original.error is None
        graph=tree_to_graph(original.tree)
        references=connected_subsets(row['evidence_positions'],graph.edges)
        targets=connected_subsets(row['target_positions'],graph.edges)
        reference=references[0]
        assert len(reference)==row['motif_size']
        smp=sample_reference_last(graph,targets,reference,seed=row['sampling_seed'])
        full=smp.prefix+smp.target;parsed=parse_sample(full)
        assert parsed.error is None and len(parsed.tree.parts)==len(original.tree.parts)
        old_pose=np.asarray(decode_graph(graph));new_pose=np.asarray(decode_graph(tree_to_graph(parsed.tree)))
        align=old_pose[smp.walk.order[0]]@np.linalg.inv(new_pose[0])
        aligned=align[None]@new_pose;expected=old_pose[list(smp.walk.order)]
        error=float(np.max(np.linalg.norm(aligned[:,:3,3]-expected[:,:3,3],axis=1)))
        assert error<.05 and np.allclose(aligned[:,:3,:3],expected[:,:3,:3],atol=1e-4)
        assert all(parsed.tree.parts[i]==original.tree.parts[v] for i,v in enumerate(smp.walk.order))
        if score_text(full)[2]:return None,'collision'
        p=TOKENIZER.encode(smp.prefix,add_special_tokens=False);t=TOKENIZER.encode(smp.target,add_special_tokens=False)
        assert TOKENIZER.encode(full,add_special_tokens=False)==p+t
        ids=p+t+[TOKENIZER.eos_token_id]
        if len(ids)>6400:return None,'overlength'
        n=smp.walk.n_prefix;assert n==row['n_prefix'] and len(smp.walk.order)-n==row['k']
        evidence=set(row['evidence_positions'])
        row.update(original_layout=row.get('layout','contiguous_late'),layout='reference_last',
                   prompt=smp.prefix,completion=smp.target,tokens_with_eos=len(ids),target_tokens_with_eos=len(t)+1,
                   loss_token_fraction=(len(t)+1)/len(ids),reference_last_positions=list(range(n-len(reference),n)),
                   evidence_positions=[i for i,v in enumerate(smp.walk.order) if v in evidence],
                   target_positions=list(range(n,len(smp.walk.order))),evidence_gap_parts=0,
                   order_from_original=list(smp.walk.order),reference_last_geometry_error_ldu=error,
                   target_anchor_position=parsed.tree.edges[n-1].parent)
        return (row,ids,len(p)),None
    except (ValueError,AssertionError,KeyError,IndexError) as exc:return None,type(exc).__name__

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'READY.json').exists():raise RuntimeError('already ready; refusing overwrite')
    roots={'v2':Path('runs/posttrain_icl_v2'),'v3':Path('runs/posttrain_icl_v3_local')}
    counts,skips,totals=Counter(),Counter(),Counter();seen=set();sources=defaultdict(set);families=defaultdict(set)
    provenance={};maximum_error=0.
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for dataset,root in roots.items():
            for split in ('train','dev','test'):
                source=root/f'{split}.jsonl';key=f'{dataset}/{split}'
                provenance[key]=hashlib.sha256(source.read_bytes()).hexdigest()
                dest=args.out/dataset;dest.mkdir(exist_ok=True)
                offset=0
                with (source.open() as inp,(dest/f'{split}.jsonl').open('w') as out,
                      (dest/f'{split}.u32').open('wb') as binary,(dest/f'{split}.index.jsonl').open('w') as index):
                    for result,error in pool.map(transform,inp,chunksize=16):
                        if error:skips[f'{key}:{error}']+=1;continue
                        row,ids,prefix=result;h=hashlib.sha256((row['prompt']+row['completion']).encode()).hexdigest()
                        if h in seen:skips[f'{key}:duplicate']+=1;continue
                        seen.add(h);assert row['split']==split
                        out.write(json.dumps(row)+'\n');binary.write(np.asarray(ids,dtype='<u4').tobytes())
                        record={k:row.get(k) for k in ['id','source_key','source_family','condition','layout','k']}
                        record.update(offset=offset,length=len(ids),prefix_length=prefix,target_length=len(ids)-prefix)
                        index.write(json.dumps(record)+'\n');offset+=len(ids)
                        counts[key]+=1;totals[f'{key}:tokens']+=len(ids);totals[f'{key}:targets']+=len(ids)-prefix
                        sources[split].add(row['source_key']);families[split].add(row['source_family'])
                        maximum_error=max(maximum_error,row['reference_last_geometry_error_ldu'])
                        if counts[key]%10000==0:print(key,counts[key],flush=True)
                print(key,counts[key],'ready',flush=True)
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not sources[a]&sources[b] and not families[a]&families[b]
    checksums={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in args.out.glob('*/*') if p.is_file()}
    report={'status':'ready','counts':dict(counts),'skips':dict(skips),'totals':dict(totals),'source_sha256':provenance,
            'artifact_sha256':checksums,'source_counts':{s:len(v) for s,v in sources.items()},'cross_split_overlap':0,
            'max_geometry_error_ldu':maximum_error,'all_rows_checked':['reference_last','parse','parts','relative_pose','collision','token_boundary','EOS'],
            'formal_training_started':False}
    (args.out/'READY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
