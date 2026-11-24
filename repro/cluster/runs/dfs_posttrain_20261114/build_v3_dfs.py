"""Six-to-ten-part local completion inside a complex single-object context."""
import argparse
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bricknet import parse_sample, score_text
import build_v2_dfs as base
from brickicl.posttrain.build_dataset import fingerprint
from brickicl.sampling.build import read_graphs
from brickicl.sampling.edits import stack_joint, stretch
from dfs_sampling import masked_walk
from brickicl.sampling.missing import pendant_copies


def process_source(job):
    pool_name,source,g,family,split,walks=job
    stats=Counter()
    if any(not (Path(os.environ['BRICKNET_DATA'])/'cvx'/f'{base.CATALOG.id_to_stem[p]}.cvx.obj').is_file()
           for p in set(g.part_ids)):
        return [], {'missing_mesh_source':1}
    groups=[]
    for c in pendant_copies(g,max_size=10,min_copies=2):
        motif_nodes={v for nodes in c.copies for v in nodes}
        body=set(range(len(g.part_ids)))-motif_nodes
        if (4<=c.size<=10 and len({g.part_ids[v] for v in c.copies[0]})>=2
            and len(body)>=30 and len(body)>=len(g.part_ids)*.5
            and len({g.part_ids[v] for v in body})>=5
            and max(Counter(g.part_ids[v] for v in body).values())/len(body)<=.45):
            groups.append((c,body))
    groups.sort(key=lambda p:(abs(p[0].size-8),-len(p[1]),p[0].copies))
    selected=next(((c,b) for c,b in groups if base.exactly_repeated(g,c)),None)
    if selected is None:return [],{'no_complex_context_motif':1}
    grp,body=selected
    joints=[stack_joint(g,c,a) for c,a in zip(grp.copies,grp.anchors)]
    variants=[]
    if 6<=grp.size<=10:variants.append(('original',g,[list(c) for c in grp.copies],0))
    if all(joints):
        for r in (2,4):
            if 6<=grp.size+r<=10 and len(g.part_ids)+r*len(joints)<=384:
                edited,added=stretch(g,joints,r)
                variants.append(('stretched',edited,[list(c)+a for c,a in zip(grp.copies,added)],r))
    rows,cache,dedup=[],{},set()
    for target in range(min(3,len(grp.copies))):
        for cond,edited,copies,r in variants:
            evidence=[i for i in range(len(copies)) if i!=target]
            options=[(cond,[])]
            if len(evidence)>=2:options.append(('deleted',[evidence[-1]]))
            for condition,deleted in options:
                remaining=[i for i in evidence if i not in deleted]
                heldout=copies[target]
                excluded=[v for i in deleted for v in copies[i]]
                total=len(edited.part_ids)-len(excluded)
                if not 50<=total<=384 or len(heldout)/total>.18:continue
                for walk in range(walks):
                    layout=('scattered','contiguous','distant')[walk%3]
                    seed=(source+(1000000 if pool_name=='pt' else 0))*10007+target*101+walk
                    blocks=() if layout=='scattered' else copies
                    # Half the walks retain natural color; recolors affect all copies consistently.
                    palettes=(0,1) if split=='train' else (0,)
                    for palette in palettes:
                        graph=base.recolor_copies(edited,copies,0 if not palette else seed+1)
                        try:
                            smp=masked_walk(graph,heldout,exclude=excluded,seed=seed,blocks=blocks,
                                            early=[copies[remaining[0]]] if layout=='distant' else ())
                            full=smp.prefix+smp.target
                            parsed=parse_sample(full)
                            if parsed.error or len(parsed.tree.parts)!=total:
                                stats['parse_invalid']+=1;continue
                            digest=hashlib.sha256(full.encode()).hexdigest()
                            if digest in dedup:continue
                            geom='\n'.join(x.split(' | ')[0] for x in full.splitlines())
                            if geom not in cache:cache[geom]=score_text(full)[2] == []
                            if not cache[geom]:stats['collision']+=1;continue
                            p=base.TOKENIZER.encode(smp.prefix,add_special_tokens=False)
                            t=base.TOKENIZER.encode(smp.target,add_special_tokens=False)
                            f=base.TOKENIZER.encode(full,add_special_tokens=False)
                            if f!=p+t or len(f)+1>6400 or len(t)+1<120:
                                stats['token_gate']+=1;continue
                            order=smp.walk.order
                            evidence_nodes={v for i in remaining for v in copies[i]}
                            positions=[i for i,v in enumerate(order) if v in evidence_nodes]
                            # The distant mode must actually leave unrelated context after all evidence.
                            gap=smp.walk.n_prefix-1-max(positions)
                            if layout=='distant' and gap<10:
                                stats['distant_gap_below_10']+=1;continue
                            dedup.add(digest)
                            rows.append({
                                'id':f'v3-{pool_name}-{source}-t{target}-w{walk}-{condition}-r{r}-p{palette}',
                                'source_split':pool_name,'source':source,'source_key':f'{pool_name}:{source}',
                                'source_family':family,'split':split,'condition':condition,'r':r,
                                'layout':'dfs_context_then_M','requested_reference_layout':layout,'palette_id':palette,'walk_seed':walk,'sampling_seed':seed,
                                'n_prefix':smp.walk.n_prefix,'k':len(heldout),'motif_size':len(heldout),
                                'motif_size_original':grp.size,'target_copy_count':1,'target_copy_indices':[target],
                                'evidence_copy_count':len(remaining),'deleted_copy_count':len(deleted),'n_copies':len(copies),
                                'body_parts':len(body),'body_part_types':len({g.part_ids[v] for v in body}),
                                'body_max_part_type_fraction':max(Counter(g.part_ids[v] for v in body).values())/len(body),
                                'target_fraction_parts':len(heldout)/total,'evidence_gap_parts':gap,
                                'sampling_order':'dfs_context_then_M','tokens_with_eos':len(f)+1,'target_tokens_with_eos':len(t)+1,
                                'loss_token_fraction':(len(t)+1)/(len(f)+1),
                                'evidence_positions':positions,'target_positions':list(range(smp.walk.n_prefix,total)),
                                'motif_isomorphism':'VF2_directed_multigraph_part_color_connectors_joint_params_attachment',
                                'prompt':smp.prefix,'completion':smp.target,
                            })
                        except (ValueError,KeyError,AssertionError,IndexError) as exc:stats[type(exc).__name__]+=1
    return rows,dict(stats)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--pt-cache',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--walks',type=int,default=6)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.glob('*.jsonl')):raise RuntimeError('use a new output folder')
    root=Path(os.environ['BRICKNET_DATA'])/'dataset'
    forbidden={fingerprint(g) for g in read_graphs(root/'val.npz.xz')}
    pools=[('sft',list(enumerate(read_graphs(root/'sft.npz.xz')))),
           ('pt',base.compact_pt(root/'pt.npz.xz',args.pt_cache))]
    jobs=[]
    for name,candidates in pools:
        for source,g in candidates:
            if len(g.components)!=1 or not 50<=len(g.part_ids)<=384:continue
            family=fingerprint(g)
            if family in forbidden:continue
            bucket=int(family[:12],16)%20
            split='test' if bucket==0 else 'dev' if bucket==1 else 'train'
            jobs.append((name,source,g,family,split,args.walks if split=='train' else 3))
    counts,skipped,totals=Counter(),Counter(),Counter()
    sources,families=defaultdict(set),defaultdict(set)
    seen=set()
    streams={s:(args.out/f'{s}.jsonl').open('w',encoding='utf-8') for s in ('train','dev','test')}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker) as pool:
        try:
            for done,(rows,skip) in enumerate(pool.map(process_source,jobs,chunksize=4),1):
                skipped.update(skip)
                for row in rows:
                    h=hashlib.sha256((row['prompt']+row['completion']).encode()).hexdigest()
                    if h in seen:skipped['duplicate_text']+=1;continue
                    seen.add(h);s=row['split']
                    streams[s].write(json.dumps(row,ensure_ascii=False)+'\n')
                    counts[s]+=1;counts[f'{s}:{row["condition"]}']+=1
                    sources[s].add(row['source_key']);families[s].add(row['source_family'])
                    totals[f'{s}:supervised_tokens']+=row['target_tokens_with_eos']
                if done%2000==0:
                    for f in streams.values():f.flush()
                    print(f'{done}/{len(jobs)}; counts={dict(counts)}',flush=True)
        finally:
            for f in streams.values():f.close()
    for a,b in itertools.combinations(streams,2):assert not sources[a]&sources[b] and not families[a]&families[b]
    meta={'version':3,'task':'complex_context_local_repeat','model_sizes':['0.6B','1.7B','4B'],
          'counts':dict(counts),'source_counts':{k:len(v) for k,v in sources.items()},
          'family_counts':{k:len(v) for k,v in families.items()},'token_totals':dict(totals),
          'max_tokens_with_eos':6400,'upstream_splits':{s:['pt','sft'] for s in streams},
          'quality_gates':{'target_parts':[6,10],'full_parts':[50,384],'max_target_fraction':.18,
                           'min_body_parts':30,'min_body_part_types':5,'min_body_fraction_before_edit':.5,
                           'max_body_part_type_fraction':.45,
                           'min_evidence_copies':1,'distant_min_gap_parts':10},
          'max_source_parts':384,'sampling':'DFS within each visible/missing phase','walks_train':args.walks,'walks_dev_test':3,'skips':dict(skipped),
          'split_rule':'same source-family hash buckets as v1/v2; no val sources used for training',
          'limitations':'Post-training heldout sources may have been seen during official PT. Source fingerprints are not exhaustive near-duplicate detection. Collision validation is not mechanical stability.'}
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
