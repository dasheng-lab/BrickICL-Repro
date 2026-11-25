"""Materialize paired DFS/DFS-with-M-last paths and grounded captions for four training objectives."""
import hashlib
import json
import lzma
import time
import fcntl
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from bricknet import parse_sample,serialize_tree,tree_to_graph,decode_graph,score_text
from bricknet.graph import _build_tree
from transformers import AutoTokenizer
from brickicl.posttrain.reference_last import connected_subsets
from brickicl.posttrain.build_dataset import fingerprint
from brickicl.sampling.build import read_graphs
from dfs_sampling import masked_walk

ROOT=Path('runs/dfs_posttrain_20261114')
DATA=Path('data/bricknet/dataset')
TOK=None
CAPS={}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def init():
    global TOK,CAPS
    TOK=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    CAPS=json.loads((ROOT/'official_captions.json').read_text())

def dfs(order,parent):
    kids=defaultdict(list)
    for v in order[1:]:kids[parent[v][0]].append(v)
    result=[];stack=[order[0]]
    while stack:
        v=stack.pop();result.append(v);stack.extend(reversed(kids[v]))
    return result

def check_dfs(tree):
    active=[0]
    for edge in tree.edges:
        if edge.parent not in active:return False
        active=active[:active.index(edge.parent)+1]+[edge.child]
    return True

def inventory(lines):
    return Counter(line.split(' ',1)[1] for line in lines if ' | ' in line)
def caption(row,full,completion,cropped):
    lines=full.splitlines()
    whole=inventory(lines)
    colors=Counter()
    for part,n in whole.items():colors[part.rsplit(' | ',1)[1]]+=n
    remaining=row['evidence_copy_count']+row['target_copy_count']
    palette=', '.join(f'{color} ({n} {"part" if n==1 else "parts"})' for color,n in sorted(colors.items()))
    target_inventory=inventory(completion.splitlines())
    copies=row['target_copy_count']
    assert all(n%copies==0 for n in target_inventory.values())
    motif=', '.join(f'{n//copies} {part.replace(" | "," in ")}' for part,n in sorted(target_inventory.items()))
    # This specifies the requested whole object, independently of which copy is hidden during M training.
    structure=(f'Construct a connected LEGO assembly of {sum(whole.values())} parts. '
               f'It contains {remaining} matching subassemblies, each made of {row["motif_size"]} parts, '
               f'attached to a shared body. Each matching subassembly contains {motif}. '
               f'The complete assembly uses these colors and counts: {palette}.')
    choices=CAPS.get(str(row['source']),[]) if row['source_split']=='sft' else []
    if not choices or cropped:
        origin='graph_grounded_structural_description'
        return structure,origin
    original=choices[row['sampling_seed']%len(choices)]
    edits=[]
    if row.get('r',0):edits.append(f'Add {row["r"]} copies of the repeated stacking element to each matching subassembly.')
    if row.get('deleted_copy_count',0):edits.append(f'Remove {row["deleted_copy_count"]} matching subassemblies, leaving {remaining} in total.')
    if row.get('palette_id',0):edits.append('Use the final subassembly colors specified below in place of their original colors.')
    if not edits:
        return original+'\n'+structure,'official_caption_with_verified_structure'
    # Explicit editing task keeps the original and the edited-state assertions distinct.
    return ('Starting design: '+original+'\nBuild the following edited version: '+' '.join(edits)+
            '\nFinal structural specification: '+structure),'official_caption_with_verified_edit_instructions'

def transform(line):
    row=json.loads(line)
    parsed=parse_sample(row['prompt']+row['completion'])
    assert parsed.error is None
    tree=parsed.tree;graph=tree_to_graph(tree)
    total=len(tree.parts)
    missing=set(row['target_positions'])
    evidence=set(row['evidence_positions'])
    assert len(missing)==row['k']
    # Keep all repeated copies, their anchors, and their ancestors while selecting a <=100-part connected tree.
    keep=set(range(total))
    parents={e.child:e.parent for e in tree.edges}
    protected=missing|evidence|{0}
    for v in list(protected):
        while v:
            v=parents[v];protected.add(v)
    if len(protected)>100:return None,'required_motif_and_ancestors_over_100'
    children=Counter(parents.values())
    while len(keep)>100:
        leaves=[v for v in keep-protected if children[v]==0]
        if not leaves:return None,'cannot_select_connected_100_parts'
        v=max(leaves);keep.remove(v);children[parents[v]]-=1
    cropped=len(keep)<total
    if row['id'].startswith('v3-'):
        body=keep-missing-evidence
        types=Counter(tree.parts[v].part_id for v in body)
        if len(body)<30 or len(body)<len(keep)*.5 or len(types)<5 or max(types.values())/len(body)>.45:
            return None,'cropped_v3_body_quality'
    refs=connected_subsets(evidence,graph.edges)
    reference=refs[0]
    assert len(reference)==row['motif_size']
    sample=masked_walk(graph,missing,exclude=set(range(total))-keep,seed=row['sampling_seed'],late=[reference])
    full_masked=sample.prefix+sample.target
    masked=parse_sample(full_masked).tree
    plain_order=dfs(sample.walk.order,sample.walk.parent_of)
    plain_tree=_build_tree(graph,plain_order,sample.walk.parent_of)
    assert check_dfs(plain_tree)
    full=serialize_tree(plain_tree)
    expected=np.asarray(decode_graph(graph))
    for order,new_tree in ((sample.walk.order,masked),(plain_order,plain_tree)):
        actual=np.asarray(decode_graph(tree_to_graph(new_tree)))
        alignment=expected[order[0]]@np.linalg.inv(actual[0])
        assert np.allclose(alignment[None]@actual,expected[list(order)],atol=1e-4)
        assert all(new_tree.parts[i]==tree.parts[j] for i,j in enumerate(order))
    _,invalid,collisions=score_text(full)
    if invalid is not None or collisions!=[]:return None,'collision_or_unchecked'
    n=sample.walk.n_prefix
    updated=dict(row,prompt=sample.prefix,completion=sample.target,n_prefix=n,
                 target_positions=list(range(n,len(keep))),evidence_positions=[i for i,v in enumerate(sample.walk.order) if v in evidence],
                 full_dfs_path=full,selected_parts=len(keep),preselection_parts=total,
                 cropped_to_path_budget=cropped,full_dfs=True,layout='dfs_context_reference_then_M',
                 target_anchor_position=masked.edges[n-1].parent)
    updated['target_fraction_parts']=len(missing)/len(keep)
    if row['id'].startswith('v3-'):
        updated.update(body_parts=len(body),body_part_types=len(types),
                       body_max_part_type_fraction=max(types.values())/len(body))
    cap,kind=caption(updated,full,sample.target,cropped)
    updated.update(caption=cap,caption_method=kind,caption_checked=['part_counts','palette','motif_inventory','copy_counts','edits'])
    newline=TOK.encode('\n',add_special_tokens=False)
    eos=[TOK.eos_token_id]
    prefix=TOK.encode(sample.prefix,add_special_tokens=False)
    target=TOK.encode(sample.target,add_special_tokens=False)
    all_g=TOK.encode(full,add_special_tokens=False)
    cap_tokens=TOK.encode(cap,add_special_tokens=False)+newline
    assert TOK.encode(full_masked,add_special_tokens=False)==prefix+target
    start=TOK.encode('a',add_special_tokens=False)
    assert all_g[:len(start)]==start
    items={
        'PT_M':(prefix+target+eos,len(prefix)),
        'PT_G':(all_g+eos,len(start)),
        'SFT_M':(cap_tokens+prefix+target+eos,len(cap_tokens)+len(prefix)),
        'SFT_G':(cap_tokens+all_g+eos,len(cap_tokens)),
    }
    if any(len(ids)>6400 for ids,p in items.values()):return None,'over_6400_tokens_with_caption'
    updated.update(tokens_with_eos=len(items['PT_M'][0]),target_tokens_with_eos=len(target)+1,
                   loss_token_fraction=(len(target)+1)/len(items['PT_M'][0]),
                   objective_token_lengths={k:len(v[0]) for k,v in items.items()})
    assert all(ids[-1]==TOK.eos_token_id and 0<p<len(ids) for ids,p in items.values())
    return (updated,items),None

def main():
    if (ROOT/'corpus/READY.json').exists():print('corpus already ready');return
    captions=defaultdict(list)
    for line in lzma.open(DATA/'captions_sft.jsonl.xz','rt'):
        r=json.loads(line);captions[str(r['id'])].append(r['caption'].strip())
    (ROOT/'official_captions.json').write_text(json.dumps(captions))
    counts=Counter();skips=Counter();methods=Counter();lengths=defaultdict(Counter);cropped_counts=Counter()
    families=defaultdict(set);seen={};caption_examples=[]
    with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
        for version in ('v2','v3'):
            while not (ROOT/f'raw_{version}/metadata.json').exists():
                print('Waiting for completed raw data',version,flush=True)
                time.sleep(30)
            for split in ('dev','test','train'):
                dest=ROOT/'corpus'/version;dest.mkdir(parents=True,exist_ok=True)
                handles={};offsets=Counter()
                for objective in ('PT_M','PT_G','SFT_M','SFT_G'):
                    target_dir=ROOT/'cache'/objective/version;target_dir.mkdir(parents=True,exist_ok=True)
                    handles[objective]=((target_dir/f'{split}.u32').open('wb'),(target_dir/f'{split}.index.jsonl').open('w'))
                with (ROOT/f'raw_{version}/{split}.jsonl').open() as inp,(dest/f'{split}.jsonl').open('w') as human:
                    try:
                        for result,error in pool.map(transform,inp,chunksize=12):
                            key=f'{version}/{split}'
                            if error:skips[key+':'+error]+=1;continue
                            row,items=result
                            digest=hashlib.sha256((row['full_dfs_path']+row['prompt']+row['completion']).encode()).hexdigest()
                            if digest in seen:
                                reason='duplicate' if seen[digest]==split else 'cross_split_exact_duplicate_removed'
                                skips[key+':'+reason]+=1;continue
                            seen[digest]=split
                            human.write(json.dumps(row)+'\n')
                            meta={k:row[k] for k in ('id','source_key','source_family','condition','k','selected_parts','caption_method')}
                            for obj,(ids,prefix) in items.items():
                                binary,index=handles[obj]
                                binary.write(np.asarray(ids,dtype='<u4').tobytes())
                                index.write(json.dumps(meta|dict(offset=offsets[obj],length=len(ids),prefix_length=prefix,target_length=len(ids)-prefix))+'\n')
                                offsets[obj]+=len(ids)
                            counts[key]+=1;methods[row['caption_method']]+=1;lengths[key][row['selected_parts']]+=1
                            cropped_counts[key]+=row['cropped_to_path_budget']
                            families[split].add(row['source_family'])
                            if len(caption_examples)<12 or (row['caption_method']=='graph_grounded_structural_description' and len(caption_examples)<24):
                                caption_examples.append({k:row[k] for k in ('id','source_key','condition','r','palette_id','selected_parts','caption_method','caption')})
                            if counts[key]%5000==0:print(key,counts[key],flush=True)
                    finally:
                        for binary,index in handles.values():binary.close();index.close()
                print(version,split,'DONE',counts[f'{version}/{split}'],flush=True)
    for a,b in (('train','dev'),('train','test'),('dev','test')):assert not families[a]&families[b]
    report={'status':'ready','counts':dict(counts),'skips':dict(skips),'caption_methods':dict(methods),
            'cropped_rows':dict(cropped_counts),
            'length_histograms':{k:dict(v) for k,v in lengths.items()},'cross_split_overlap':0,
            'full_order':'strict DFS','missing_order':'DFS context, complete reference, then M',
            'path_parts_limit':100,'source_graph_parts_limit':384,'token_limit':6400,
            'sft_caption_input':'both M-only and Full-G','paired_geometry_across_four_arms':True,
            'raw_data_sha256':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('raw_*/*.jsonl')},
            'data_code_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},
            'original_source_archive_sha256':sha(ROOT/'sources.tgz')}
    for obj in handles:
        cache=ROOT/'cache'/obj
        hashes={str(p.relative_to(cache)):sha(p) for p in cache.glob('*/*') if p.is_file()}
        (cache/'READY.json').write_text(json.dumps(report|{'artifact_sha256':hashes},indent=2))
    (ROOT/'caption_examples.json').write_text(json.dumps(caption_examples,ensure_ascii=False,indent=2))
    (ROOT/'corpus/READY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    with (ROOT/'prepare_corpus.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
