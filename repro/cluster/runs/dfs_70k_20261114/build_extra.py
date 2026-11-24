"""Create additional official-SFT-derived v3 paths using new DFS seeds, with the same split and motif checks."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import build_v3_extra as builder
import prepare_corpus as corpus
from brickicl.sampling.build import read_graphs
from brickicl.posttrain.build_dataset import fingerprint
from caption_rules import ROOT,DATA,load_bases,caption_for

BASES={}
def initialize():
    global BASES
    builder.base.init_worker()
    BASES={int(k):v for k,v in json.loads((ROOT/'screened_bases.json').read_text()).items()}
    corpus.TOK=builder.base.TOKENIZER
    corpus.CAPS={str(k):[v] for k,v in BASES.items()}
def generate(job):
    candidates,skips=builder.process_source(job)
    result=[]
    counts=Counter(skips)
    for row in candidates:
        item,error=corpus.transform(json.dumps(row))
        if error:counts['final/'+error]+=1;continue
        r,_=item
        text,quality=caption_for(r,BASES)
        if not text:counts['no_screened_caption_after_selection']+=1;continue
        r.update(caption=text,caption_method=quality,origin='new_official_sft_DFS_walk')
        result.append(r)
    return result,dict(counts)
def main():
    if (ROOT/'EXTRA_READY.json').exists():return
    bases=load_bases()
    forbidden={fingerprint(g) for g in read_graphs(DATA/'val.npz.xz')}
    jobs=[]
    for i,g in enumerate(read_graphs(DATA/'sft.npz.xz')):
        if i not in bases or len(g.components)!=1 or not 50<=len(g.part_ids)<=100:continue
        family=fingerprint(g)
        if family in forbidden or int(family[:12],16)%20<2:continue
        jobs.append(('sft',i,g,family,'train',48))
    print('official source candidates',len(jobs),'new walk indices 6..47',flush=True)
    counts=Counter();skips=Counter();sources=set();families=set();seen=set()
    # Exclude exact serializations already present in the full existing v3 pool.
    for line in Path('runs/dfs_posttrain_20261114/corpus/v3/train.jsonl').open():
        r=json.loads(line);seen.add(builder.hashlib.sha256((r['prompt']+r['completion']).encode()).digest())
    with (ROOT/'extra_v3.jsonl').open('w') as out,ProcessPoolExecutor(max_workers=8,initializer=initialize) as pool:
        for i,(rows,rejects) in enumerate(pool.map(generate,jobs,chunksize=1),1):
            skips.update(rejects)
            for r in rows:
                h=builder.hashlib.sha256((r['prompt']+r['completion']).encode()).digest()
                if h in seen:skips['duplicate_path_removed']+=1;continue
                seen.add(h)
                out.write(json.dumps(r)+'\n');counts[r['condition']]+=1
                sources.add(r['source_key']);families.add(r['source_family'])
            if i%250==0:
                out.flush();print(i,'/',len(jobs),dict(counts),flush=True)
    report={'rows':sum(counts.values()),'condition_counts':dict(counts),'official_sources':len(sources),
            'families':len(families),'skips':dict(skips),'walk_indices':[6,47],'max_parts':100,
            'new_distinct_source_claim':False}
    (ROOT/'EXTRA_READY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
