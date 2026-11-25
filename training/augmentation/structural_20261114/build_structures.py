import argparse,json
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from brickicl.sampling.build import read_graphs
from brickicl.posttrain.build_dataset import fingerprint
import factory

ROOT=Path('runs/structural_20261114')
DATA=Path('baseline/bricknet/data/dataset')
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pool',choices=['sft','pt'],required=True)
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    output=ROOT/args.pool;output.mkdir(exist_ok=True)
    if (output/'READY.json').exists():return
    if (output/'v3_train.jsonl').exists():raise RuntimeError('Partial pool exists; inspect before retrying')
    forbidden={fingerprint(g) for g in read_graphs(DATA/'val.npz.xz')}
    graphs=(enumerate(read_graphs(DATA/'sft.npz.xz')) if args.pool=='sft' else
            factory.common.compact_pt(DATA/'pt.npz.xz',Path('runs/posttrain_icl_v2/pt.cache.npz')))
    jobs=[]
    for source,g in graphs:
        if len(g.components)!=1 or not 16<=len(g.part_ids)<=384:continue
        family=fingerprint(g)
        if family in forbidden:continue
        bucket=int(family[:12],16)%20
        split='test' if bucket==0 else 'dev' if bucket==1 else 'train'
        jobs.append((args.pool,int(source),g,family,split))
    print('Source jobs',args.pool,len(jobs),flush=True)
    files={(v,s):(output/f'{v}_{s}.jsonl').open('w') for v in ('v2','v3') for s in ('train','dev','test')}
    seen={};counts=Counter();edits=Counter();skips=Counter();families=defaultdict(set)
    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=factory.init) as pool:
            for i,(rows,rejects) in enumerate(pool.map(factory.process,jobs,chunksize=1),1):
                skips.update(rejects)
                for row in rows:
                    key=(row['version'],row['geometry_id'])
                    if key in seen:
                        skips['duplicate_geometry_across_sources']+=1;continue
                    seen[key]=row['split']
                    assert row['deletion_count']==0 and row['condition'] in ('original','modified')
                    v,s=row['version'],row['split']
                    files[v,s].write(json.dumps(row)+'\n')
                    counts[f'{v}/{s}']+=1;edits[f'{v}/{s}/{row["edit"]["kind"]}']+=1
                    if row['has_semantic_caption']:counts[f'{v}/{s}/caption']+=1
                    families[s].add(row['source_family'])
                if i%500==0:
                    for stream in files.values():stream.flush()
                    print(i,'/',len(jobs),dict(counts),flush=True)
    finally:
        for stream in files.values():stream.close()
    for a,b in (('train','dev'),('train','test'),('dev','test')):assert not families[a]&families[b]
    result={'pool':args.pool,'counts':dict(counts),'edit_counts':dict(edits),'skips':dict(skips),
        'family_counts':{s:len(f) for s,f in families.items()},'no_deletion':True,'one_serialization_per_geometry':True}
    (output/'READY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
