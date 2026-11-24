"""Long-target single-object ICL data; fixed source splits, explicit motif evidence."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import lzma
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
from bricknet import Graph, parse_sample, score_text
from bricknet.data import load_catalog
from bricknet.graph import _graph_from_arrays, _NPZ_KEYS
from transformers import AutoTokenizer

from brickicl.posttrain.build_dataset import fingerprint
from brickicl.sampling.build import read_graphs
from brickicl.sampling.edits import stack_joint, stretch
from dfs_sampling import masked_walk
from brickicl.sampling.missing import pendant_copies

TOKENIZER = None
CATALOG = None


def init_worker():
    global TOKENIZER, CATALOG
    TOKENIZER = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    CATALOG = load_catalog()


def motif_graph(g, nodes):
    nodes = set(nodes)
    motif = nx.MultiDiGraph()
    for v in nodes:
        motif.add_node(v, label=(int(g.part_ids[v]), int(g.colors[v])), hook=[])
    for e in g.edges:
        a, b = int(e['a']), int(e['b'])
        label = (str(e['family']), int(e['a_conn']), int(e['b_conn']), bool(e['flip']),
                 round(float(e['yaw']), 6), tuple(round(float(x), 6) for x in e['rot']))
        if a in nodes and b in nodes:
            motif.add_edge(a, b, label=label)
        elif a in nodes:
            motif.nodes[a]['hook'].append((True, label))
        elif b in nodes:
            motif.nodes[b]['hook'].append((False, label))
    for v in nodes:
        motif.nodes[v]['hook'] = tuple(sorted(motif.nodes[v]['hook']))
    return motif


def exactly_repeated(g, group):
    ref = motif_graph(g, group.copies[0])
    node_match = lambda a, b: a['label'] == b['label'] and a['hook'] == b['hook']
    edge_match = lambda a, b: sorted(v['label'] for v in a.values()) == sorted(v['label'] for v in b.values())
    return all(nx.is_isomorphic(ref, motif_graph(g, c), node_match=node_match, edge_match=edge_match)
               for c in group.copies[1:])


def recolor_copies(g, copies, palette_seed):
    if palette_seed == 0:
        return g
    palette = np.array([0, 1, 2, 4, 14, 15, 25])
    np.random.default_rng(palette_seed).shuffle(palette)
    nodes = {v for c in copies for v in c}
    original = sorted({int(g.colors[v]) for v in nodes})
    mapping = {c: int(palette[i % len(palette)]) for i, c in enumerate(original)}
    colors = tuple(mapping[int(c)] if i in nodes else int(c) for i, c in enumerate(g.colors))
    return Graph(g.part_ids, colors, g.transforms, g.edges, g.components)


def masks_for(group):
    n = len(group.copies)
    # Hold up to half the copies out, or more if needed to reach eight target parts.
    m = min(3, n - 1, max((8 + group.size - 1) // group.size, n // 2))
    if group.size * m < 8:
        return []
    choices = [tuple((start + j) % n for j in range(m)) for start in range(min(n, 4))]
    return list(dict.fromkeys(tuple(sorted(c)) for c in choices))


def process_source(job):
    pool_name, source, g, family, split, walks = job
    stats = Counter()
    if any(not (Path(os.environ['BRICKNET_DATA'])/'cvx'/f'{CATALOG.id_to_stem[p]}.cvx.obj').is_file()
           for p in set(g.part_ids)):
        return [], {'missing_mesh_source': 1}
    groups = [c for c in pendant_copies(g, max_size=32, min_copies=2)
              if c.size >= 4 and len({g.part_ids[v] for v in c.copies[0]}) >= 2
              and c.size * len(c.copies) >= .35 * len(g.part_ids) and masks_for(c)]
    groups.sort(key=lambda c: (-c.size * len(masks_for(c)[0]), -c.size, c.copies))
    grp = next((c for c in groups if exactly_repeated(g, c)), None)
    if grp is None:
        return [], {'no_large_verified_motif': 1}
    rows, geometry_cache, dedup = [], {}, set()
    joints = [stack_joint(g, c, a) for c, a in zip(grp.copies, grp.anchors)]
    variants = [('original', g, [list(c) for c in grp.copies], 0)]
    if all(joints):
        for r in (2, 4):
            if len(g.part_ids) + r * len(joints) <= 384:
                edited, added = stretch(g, joints, r)
                variants.append(('stretched', edited, [list(c)+a for c,a in zip(grp.copies, added)], r))
    for mask_number, indices in enumerate(masks_for(grp)):
        for cond, edited, copies, r in variants:
            evidence = [i for i in range(len(copies)) if i not in indices]
            options = [(cond, [])]
            if cond == 'original' and len(evidence) >= 2:
                options.append(('deleted', [evidence[-1]]))
            for condition, deleted in options:
                remaining = [i for i in evidence if i not in deleted]
                heldout = sorted(v for i in indices for v in copies[i])
                excluded = sorted(v for i in deleted for v in copies[i])
                total = len(edited.part_ids) - len(excluded)
                if not 8 <= len(heldout) <= 64 or len(heldout)/total < .2:
                    stats['small_answer'] += 1
                    continue
                if set(grp.anchors[i] for i in indices) & set(heldout + excluded):
                    stats['masked_anchor'] += 1
                    continue
                for walk in range(walks):
                    # All augmentation parameters depend only on this source, never on a split's outcomes.
                    seed = (source + (1000000 if pool_name=='pt' else 0)) * 10007 + mask_number * 101 + walk
                    for palette_id in (0, 1):
                        graph = recolor_copies(edited, copies, 0 if palette_id == 0 else source * 31 + walk + 1)
                        try:
                            sample = masked_walk(graph, heldout, exclude=excluded, seed=seed,
                                                 blocks=copies, late=[copies[remaining[0]]])
                            full = sample.prefix + sample.target
                            digest = hashlib.sha256(full.encode()).hexdigest()
                            if digest in dedup:
                                stats['duplicate_text'] += 1
                                continue
                            parsed = parse_sample(full)
                            if parsed.error is not None or len(parsed.tree.parts) != total:
                                stats['parse_invalid'] += 1
                                continue
                            # Colors cannot change collision geometry; use the same parsed part/edge text.
                            geometry = '\n'.join(line.split(' | ')[0] for line in full.splitlines())
                            if geometry not in geometry_cache:
                                geometry_cache[geometry] = score_text(full)[2] == []
                            if not geometry_cache[geometry]:
                                stats['collision'] += 1
                                continue
                            p = TOKENIZER.encode(sample.prefix, add_special_tokens=False)
                            t = TOKENIZER.encode(sample.target, add_special_tokens=False)
                            f = TOKENIZER.encode(full, add_special_tokens=False)
                            if f != p+t or len(t)+1 < 192 or len(t)/(len(f)+1) < .2 or len(f)+1 > 6400:
                                stats['token_quality_gate'] += 1
                                continue
                            dedup.add(digest)
                            order = sample.walk.order
                            held_positions = list(range(sample.walk.n_prefix, len(order)))
                            evidence_nodes = {v for i in remaining for v in copies[i]}
                            example_id = f'{pool_name}-{source}-m{mask_number}-w{walk}-{condition}-r{r}-p{palette_id}'
                            rows.append({
                                'id': example_id, 'split': split, 'source_split': pool_name, 'source': source,
                                'source_key': f'{pool_name}:{source}',
                                'source_family': family, 'condition': condition, 'r': r,
                                'palette_id': palette_id, 'walk_seed': walk, 'sampling_seed': seed,
                                'target_copy_indices': list(indices), 'target_copy_count': len(indices),
                                'evidence_copy_count': len(remaining), 'deleted_copy_count': len(deleted),
                                'motif_size_original': grp.size, 'motif_size': len(copies[0]),
                                'n_copies': len(copies), 'n_prefix': sample.walk.n_prefix, 'k': len(heldout),
                                'target_fraction_parts': len(heldout)/total,
                                'target_tokens_with_eos': len(t)+1, 'sampling_order':'dfs_context_then_M','tokens_with_eos': len(f)+1,
                                'loss_token_fraction': (len(t)+1)/(len(f)+1),
                                'motif_isomorphism': 'VF2_directed_multigraph_part_color_connectors_joint_params_attachment',
                                'evidence_positions': [i for i,v in enumerate(order) if v in evidence_nodes],
                                'target_positions': held_positions,
                                'prompt': sample.prefix, 'completion': sample.target,
                            })
                        except (KeyError, ValueError, AssertionError, IndexError) as exc:
                            stats[type(exc).__name__] += 1
    return rows, dict(stats)


def compact_pt(path, cache):
    """Decode only 16..384-part graph records; retain the official graph row IDs."""
    if not cache.exists():
        print('decompressing PT CSR archive', flush=True)
        temporary = cache.with_suffix('.tmp')
        with lzma.open(path, 'rb') as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
        temporary.replace(cache)
    with np.load(cache, allow_pickle=False) as d:
        nptr, eptr = d['node_ptr'], d['edge_ptr']
        selected = np.flatnonzero((np.diff(nptr)>=16)&(np.diff(nptr)<=384))
        print(f'loading PT arrays for {len(selected)} compact source graphs',flush=True)
        pids, colors, kind, idx, yaw, flip, rot = (d[k] for k in _NPZ_KEYS)
        result=[]
        for source in selected:
            lo,hi=nptr[source:source+2]; a,b=eptr[source:source+2]
            g=_graph_from_arrays(pids[lo:hi],colors[lo:hi],kind[a:b],idx[a:b],yaw[a:b],flip[a:b],rot[a:b],None)
            if len(g.components)==1: result.append((int(source),g))
        print(f'loaded {len(result)} connected PT candidates',flush=True)
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--walks', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit-sources', type=int, default=0)
    ap.add_argument('--include-pt', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob('*.jsonl')):
        raise RuntimeError('output already contains rows; use a new versioned folder')
    root = Path(os.environ['BRICKNET_DATA'])/'dataset'
    graphs = read_graphs(root/'sft.npz.xz')
    forbidden = {fingerprint(g) for g in read_graphs(root/'val.npz.xz')}
    pools=[('sft',list(enumerate(graphs)))]
    if args.include_pt:
        pools.append(('pt', compact_pt(root/'pt.npz.xz', Path('runs/posttrain_icl_v2/pt.cache.npz'))))
    jobs = []
    for pool_name, candidates in pools:
        for source,g in candidates:
            if len(g.components) != 1 or not 16 <= len(g.part_ids) <= 384:
                continue
            family = fingerprint(g)
            if family in forbidden:
                continue
            bucket = int(family[:12],16)%20
            split = 'test' if bucket == 0 else 'dev' if bucket == 1 else 'train'
            jobs.append((pool_name,source,g,family,split,args.walks if split=='train' else 2))
    if args.limit_sources:
        jobs = jobs[:args.limit_sources]
    stats, counts = Counter(), Counter()
    sources, families = defaultdict(set), defaultdict(set)
    totals = Counter()
    seen = set()
    streams = {s:(args.out/f'{s}.jsonl').open('w',encoding='utf-8') for s in ('train','dev','test')}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        try:
            for done,(rows,skips) in enumerate(pool.map(process_source,jobs,chunksize=4),1):
                stats.update(skips)
                for row in rows:
                    h=hashlib.sha256((row['prompt']+row['completion']).encode()).hexdigest()
                    if h in seen:
                        stats['global_duplicate_text']+=1
                        continue
                    seen.add(h)
                    s=row['split']
                    streams[s].write(json.dumps(row,ensure_ascii=False)+'\n')
                    counts[s]+=1; counts[f'{s}:{row["condition"]}']+=1
                    sources[s].add(row['source_key']); families[s].add(row['source_family'])
                    totals[f'{s}:supervised_tokens']+=row['target_tokens_with_eos']
                    totals[f'{s}:all_tokens']+=row['tokens_with_eos']
                if done%1000==0:
                    for f in streams.values(): f.flush()
                    print(f'{done}/{len(jobs)} sources; rows={dict(counts)}; tokens={dict(totals)}',flush=True)
        finally:
            for f in streams.values(): f.close()
    meta={'version':2,'model_sizes':['0.6B','1.7B','4B'],'max_tokens_with_eos':6400,
          'counts':dict(counts),'source_counts':{k:len(v) for k,v in sources.items()},
          'family_counts':{k:len(v) for k,v in families.items()},'token_totals':dict(totals),
          'max_source_parts':384,'sampling':'DFS within each visible/missing phase','quality_gates':{'min_motif_parts':4,'min_part_types':2,'min_motif_coverage':.35,
                           'min_target_parts':8,'min_target_part_fraction':.2,'min_target_tokens':192,
                           'min_target_token_fraction':.2,'min_complete_evidence_copies':1},
          'walks_train':args.walks,'walks_dev_test':2,'skips':dict(stats),
          'upstream_splits':{'train':['sft','pt'] if args.include_pt else ['sft'],
                             'dev':['sft','pt'] if args.include_pt else ['sft'],
                             'test':['sft','pt'] if args.include_pt else ['sft']},
          'split_rule':'same color-independent source-family fingerprint and bucket as v1; 0 test,1 dev,2..19 train',
          'collision_validation':'every unique geometry serialization checked by score_text with all colliders; recolors reuse identical geometry validation',
          'limitations':'held out from our post-training, not from upstream PT which included SFT graphs; no exhaustive near-duplicate clustering or mechanical stability test'}
    for a,b in itertools.combinations(streams,2):
        assert not sources[a]&sources[b] and not families[a]&families[b]
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__': main()
