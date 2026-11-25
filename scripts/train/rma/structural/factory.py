"""Geometry-edit augmentation. Each candidate structure gets one fixed DFS serialization."""
import hashlib,itertools,json,os,sys
from collections import Counter,deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'preparation'))
import networkx as nx
import numpy as np
from bricknet import parse_sample,serialize_tree,tree_to_graph,decode_graph,score_text,load_catalog
from bricknet.graph import _build_tree,_part_info
from brickicl.sampling.edits import stretch,stack_joint,decorate,free_connectors
from brickicl.sampling.missing import pendant_copies
import build_v2_dfs as common
from dfs_sampling import masked_walk
from prepare_corpus import dfs,check_dfs,inventory

ROOT=Path('runs/structural_20261114')
CAT=None;CAPS={};PARTS=[]
def hash_text(s):return hashlib.sha256(s.encode()).hexdigest()
def init():
    global CAT,CAPS,PARTS
    common.init_worker();CAT=load_catalog()
    CAPS={int(k):v for k,v in json.loads((ROOT/'screened_bases.json').read_text()).items()}
    for stem in ('3024','3023','3022','3623','3710','3021','3005','3004','3003','3062b','4073'):
        pid=CAT.stem_to_id.get(stem)
        if pid is None:continue
        if not (Path(os.environ['BRICKNET_DATA'])/'cvx'/f'{CAT.id_to_stem[pid]}.cvx.obj').exists():continue
        conns=_part_info(pid).conns
        holes=[i for i,c in enumerate(conns) if c[0]=='hole' and c[1] in ('hole','tube')]
        studs=[i for i,c in enumerate(conns) if c[0]=='stud' and c[1] in ('stud','open')]
        if holes and studs:PARTS.append((pid,holes[0],studs[0]))
    assert len(PARTS)>=6,PARTS

def shape_id(parts,mats,nodes=None):
    """Frame/order invariant geometry fingerprint; color changes do not create another geometry."""
    nodes=list(range(len(parts))) if nodes is None else list(nodes)
    pids=[parts[i].part_id for i in nodes]
    counts=Counter(pids)
    anchor_pid=min(counts,key=lambda p:(counts[p],p))
    best=None
    for a in nodes:
        if parts[a].part_id!=anchor_pid:continue
        rel=np.linalg.inv(mats[a])[None]@np.asarray(mats)[nodes]
        values=[]
        for pid,T in zip(pids,rel):
            values.append((int(pid),*np.rint(T[:3,:3].ravel()*100000).astype(int).tolist(),
                           *np.rint(T[:3,3]*1000).astype(int).tolist()))
        key=tuple(sorted(values))
        if best is None or key<best:best=key
    return hash_text(repr(best))

def corresponding_maps(graph,group):
    reference=common.motif_graph(graph,group.copies[0])
    maps=[{v:v for v in group.copies[0]}]
    def node_match(a,b):return a['label']==b['label'] and a['hook']==b['hook']
    def edge_match(a,b):return sorted(v['label'] for v in a.values())==sorted(v['label'] for v in b.values())
    for copy in group.copies[1:]:
        match=nx.algorithms.isomorphism.MultiDiGraphMatcher(reference,common.motif_graph(graph,copy),
                    node_match=node_match,edge_match=edge_match)
        mapping=next(match.isomorphisms_iter(),None)
        if mapping is None:return None
        maps.append(mapping)
    return maps

def module_options(min_add,max_add,limit,seed):
    choices=[]
    for size in range(max(1,min_add),min(4,max_add)+1):
        options=list(itertools.product(range(len(PARTS)),repeat=size))
        options.sort(key=lambda ids:hash_text(f'{seed}:{ids}'))
        choices.extend(options[:max(1,limit//max(1,min(4,max_add)-max(1,min_add)+1))])
    return choices

def build_row(graph,copies,targets,source_info,edit,base_parts,base_mats,body,seed):
    pool,source,family,split,version,group_id=source_info
    missing={v for i in targets for v in copies[i]}
    visible=[i for i in range(len(copies)) if i not in targets]
    if not visible:return None,'no_reference'
    smp=masked_walk(graph,missing,root=0,seed=seed,late=[copies[visible[0]]])
    masked=parse_sample(smp.prefix+smp.target)
    assert masked.error is None
    order=dfs(smp.walk.order,smp.walk.parent_of)
    full_tree=_build_tree(graph,order,smp.walk.parent_of)
    assert check_dfs(full_tree)
    text=serialize_tree(full_tree)
    n,invalid,coll=score_text(text)
    if invalid is not None or coll!=[]:return None,'collision_or_unknown'
    if n>100:return None,'over_100'
    if version=='v2' and (not 8<=len(missing)<=64 or len(missing)/n<.2):return None,'v2_target_size'
    if version=='v3' and (not 6<=len(missing)<=10 or len(missing)/n>.18):return None,'v3_target_size'
    mats=np.asarray(decode_graph(tree_to_graph(full_tree)))
    by_node=np.empty_like(mats)
    for j,node in enumerate(order):by_node[node]=mats[j]
    if not np.allclose(by_node[list(body)],np.asarray(base_mats)[list(body)],atol=1e-4):
        return None,'body_pose_changed'
    parts=[None]*len(order)
    for j,node in enumerate(order):parts[node]=full_tree.parts[j]
    keys=[shape_id(parts,by_node,c) for c in copies]
    if len(set(keys))!=1:return None,'copies_not_same_geometry'
    gid=shape_id(parts,by_node)
    if edit['kind']!='none' and gid==shape_id(base_parts,base_mats):return None,'no_geometry_change'
    p=common.TOKENIZER.encode(smp.prefix,add_special_tokens=False)
    t=common.TOKENIZER.encode(smp.target,add_special_tokens=False)
    g=common.TOKENIZER.encode(text,add_special_tokens=False)
    if max(len(p)+len(t)+1,len(g)+1)>6400:return None,'over_tokens'
    assert common.TOKENIZER.encode(smp.prefix+smp.target,add_special_tokens=False)==p+t
    colors=Counter(CAT.code_to_color[x.color] for x in parts)
    motif=inventory(smp.target.splitlines())
    divisor=len(targets)
    assert all(v%divisor==0 for v in motif.values())
    spec=(f'Construct a connected LEGO assembly of {n} parts with {len(copies)} matching subassemblies '
          f'of {len(copies[0])} parts each. Each matching subassembly contains '+
          ', '.join(f'{count//divisor} {name.replace(" | "," in ")}' for name,count in sorted(motif.items()))+
          '. Its complete color inventory is '+
          ', '.join(f'{color} ({count} parts)' for color,count in sorted(colors.items()))+'.')
    semantic=CAPS.get(source) if pool=='sft' and source_info[-1]['caption_complete'] else None
    if semantic:
        caption=(semantic+'\n'+spec if edit['kind']=='none' else
                 'Starting design: '+semantic+'\nCreate a modified design: '+edit['instruction']+'\n'+spec)
    else:caption=spec
    cap_tokens=common.TOKENIZER.encode(caption,add_special_tokens=False)
    if len(cap_tokens)+1+max(len(p)+len(t),len(g))+1>6400:return None,'caption_over_tokens'
    evidence={v for i in visible for v in copies[i]}
    row={'id':f'{version}-{pool}-{source}-{gid[:20]}','version':version,'source_split':pool,'source':source,
         'source_key':f'{pool}:{source}','source_family':family,'split':split,'geometry_id':gid,
         'condition':'original' if edit['kind']=='none' else 'modified','edit':edit,
         'group':group_id,'selected_parts':n,'k':len(missing),'n_prefix':smp.walk.n_prefix,
         'target_copy_count':len(targets),'evidence_copy_count':len(visible),'motif_size':len(copies[0]),
         'target_positions':list(range(smp.walk.n_prefix,n)),
         'evidence_positions':[j for j,v in enumerate(smp.walk.order) if v in evidence],
         'prompt':smp.prefix,'completion':smp.target,'full_dfs_path':text,'caption':caption,
         'has_semantic_caption':bool(semantic),'caption_method':'official_screened_with_actual_edits' if semantic else 'structural_specification',
         'seed':seed,'serialization_count':1,'body_parts':len(body),'deletion_count':0}
    return row,None

def process(job):
    pool,source,g,family,split=job
    rejected=Counter();rows=[];seen=set()
    if any(c not in CAT.code_to_color for c in g.colors):return [],{'unknown_color':1}
    if any(not (Path(os.environ['BRICKNET_DATA'])/'cvx'/f'{CAT.id_to_stem[p]}.cvx.obj').exists() for p in set(g.part_ids)):
        return [],{'missing_collider':1}
    groups=pendant_copies(g,max_size=32,min_copies=2)
    for version in ('v2','v3'):
        eligible=[]
        for group in groups:
            motif={v for c in group.copies for v in c};body=set(range(len(g.part_ids)))-motif
            if group.size<4 or len({g.part_ids[v] for v in group.copies[0]})<2 or len(body)<2:continue
            if version=='v2' and len(motif)<.35*len(g.part_ids):continue
            if version=='v3':
                types=Counter(g.part_ids[v] for v in body)
                if group.size>10 or len(body)<30 or len(body)<len(g.part_ids)*.5 or len(types)<5 or max(types.values())/len(body)>.45:continue
            maps=corresponding_maps(g,group)
            if maps:eligible.append((group,body,maps))
        eligible.sort(key=lambda x:(abs(x[0].size-(8 if version=='v3' else 16)),x[0].copies))
        for group_number,(group,body,maps) in enumerate(eligible[:4 if version=='v3' else 3]):
            masks=common.masks_for(group)
            targets=(0,) if version=='v3' or not masks else masks[0]
            reference=next(i for i in range(len(group.copies)) if i not in targets)
            missing={v for i in targets for v in group.copies[i]}
            seed=int(hash_text(f'structure:{pool}:{source}:{version}:{group_number}')[:8],16)
            sampled=masked_walk(g,missing,root=min(body),seed=seed,late=[group.copies[reference]])
            tree=parse_sample(sampled.prefix+sampled.target).tree
            n=len(tree.parts);old_to_pos={v:i for i,v in enumerate(sampled.walk.order)}
            protected={old_to_pos[v] for c in group.copies for v in c}|{old_to_pos[a] for a in group.anchors}|{0}
            parent={e.child:e.parent for e in tree.edges}
            for v in list(protected):
                while v:v=parent[v];protected.add(v)
            max_add=min(6,10-group.size) if version=='v3' else min(4,64//len(targets)-group.size)
            if version=='v3':max_add=min(max_add,50//len(group.copies)-group.size)
            if pool=='sft' and source in CAPS:max_add=min(max_add,(100-n)//len(group.copies))
            max_add=max(max_add,0)
            while max_add and len(protected)>100-max_add*len(group.copies):max_add-=1
            budget=100-max_add*len(group.copies)
            if len(protected)>budget:rejected['protected_over_budget']+=1;continue
            keep=set(range(n));children=Counter(parent.values())
            while len(keep)>budget:
                leaves=[v for v in keep-protected if children[v]==0]
                if not leaves:break
                v=max(leaves);keep.remove(v);children[parent[v]]-=1
            if len(keep)>budget:rejected['cannot_crop_body']+=1;continue
            order=[v for v in sampled.walk.order if old_to_pos[v] in keep]
            parents={v:sampled.walk.parent_of[v] for v in order[1:]}
            base_tree=_build_tree(g,order,parents);base_graph=tree_to_graph(base_tree)
            remap={v:i for i,v in enumerate(order)}
            copies=[list(map(remap.__getitem__,c)) for c in group.copies]
            anchors=[remap[v] for v in group.anchors]
            base_body=set(range(len(order)))-{v for c in copies for v in c}
            if version=='v3':
                types=Counter(base_graph.part_ids[v] for v in base_body)
                final_n=len(order)+max_add*len(copies)
                if len(base_body)<30 or len(base_body)<final_n*.5 or len(types)<5 or max(types.values())/len(base_body)>.45:
                    rejected['body_quality']+=1;continue
            base_mats=np.asarray(decode_graph(base_graph));base_parts=base_tree.parts
            info=(pool,source,family,split,version,{'index':group_number,'caption_complete':len(order)==len(g.part_ids)})
            variants=[(base_graph,copies,{'kind':'none','instruction':'Keep the original subassemblies.'})]
            joints=[stack_joint(base_graph,c,a) for c,a in zip(copies,anchors)]
            if all(joints):
                for r in range(1,max_add+1):
                    edited,added=stretch(base_graph,joints,r)
                    variants.append((edited,[c+a for c,a in zip(copies,added)],{'kind':'stack_insert','added_per_copy':r,
                        'instruction':f'Insert {r} additional copies of the existing stacking element into each matching subassembly.'}))
            ref=group.copies[0];dist={v:0 for v in ref if any((e['a']==v and e['b']==group.anchors[0]) or (e['b']==v and e['a']==group.anchors[0]) for e in g.edges)}
            adj={v:set() for v in ref}
            for e in g.edges:
                a,b=int(e['a']),int(e['b'])
                if a in adj and b in adj:adj[a].add(b);adj[b].add(a)
            queue=deque(dist)
            while queue:
                v=queue.popleft()
                for u in sorted(adj[v]):
                    if u not in dist:dist[u]=dist[v]+1;queue.append(u)
            sites=[]
            for v in sorted(ref,key=lambda v:(-dist.get(v,0),g.part_ids[v],v)):
                mapped=[remap[m[v]] for m in maps]
                for side in ('stud','hole'):
                    connectors=[free_connectors(base_graph,u,side,('stud','open') if side=='stud' else ('hole','tube')) for u in mapped]
                    common_conn=sorted(set.intersection(*(set(x) for x in connectors)))
                    if common_conn:sites.append([(u,common_conn[0],side) for u in mapped])
                if len(sites)>=2:break
            limit=128 if pool=='sft' and source in CAPS and split=='train' else 24
            if split!='train':limit=12
            min_add=max(1,6-group.size) if version=='v3' else max(1,(8+len(targets)-1)//len(targets)-group.size)
            for site_idx,site in enumerate(sites[:2]):
                color=base_graph.colors[site[0][0]]
                for module in module_options(min_add,max_add,limit,seed):
                    desc=[(PARTS[i][0],color,PARTS[i][1],PARTS[i][2]) for i in module]
                    edited,added=decorate(base_graph,site,desc)
                    instruction=('Extend each matching subassembly with the same small module, using '+
                                 ', '.join(CAT.id_to_name[p]+' in '+CAT.code_to_color[c] for p,c,_,_ in desc)+'.')
                    variants.append((edited,[c+a for c,a in zip(copies,added)],{'kind':'tip_extension',
                        'added_per_copy':len(module),'site':site_idx,'part_ids':[p for p,_,_,_ in desc],'instruction':instruction}))
            for edited,newcopies,edit in variants:
                try:row,error=build_row(edited,newcopies,targets,info,edit,base_parts,base_mats,base_body,seed)
                except (ValueError,AssertionError,KeyError,IndexError,np.linalg.LinAlgError) as exc:
                    rejected['invalid_'+type(exc).__name__]+=1;continue
                if error:rejected[error]+=1;continue
                key=(version,row['geometry_id'])
                if key in seen:rejected['same_geometry']+=1;continue
                seen.add(key);rows.append(row)
    return rows,dict(rejected)
