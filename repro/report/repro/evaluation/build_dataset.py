"""Official-validation-derived repeated-subassembly benchmark with training-identical DFS."""
import hashlib,importlib.util,json,lzma,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from bricknet import Graph,parse_sample,serialize_tree,tree_to_graph,decode_graph,score_text,sample_collision_free_tree,load_catalog
from bricknet.graph import _build_tree,_part_info,_components_from_edges
from bricknet.collision import colliding_pairs
from brickicl.sampling.build import read_graphs
from brickicl.sampling.missing import pendant_copies
from brickicl.sampling.edits import stack_joint,stretch,decorate,free_connectors
from brickicl.sampling.masked_walk import split_text
from brickicl.eval.length_probe import shared_head
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];TRAIN=REPO/'runs/dfs_posttrain_20261114';sys.path.insert(0,str(TRAIN))
from dfs_sampling import masked_walk as training_walk
from prepare_corpus import dfs as training_dfs,check_dfs
CAT=load_catalog()
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def atom(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def connected_graph(pids,colors,edges):
 comps=_components_from_edges(len(pids),edges['a'],edges['b']);assert len(comps)==1
 return Graph(tuple(pids),tuple(colors),None,edges,comps)
def clear(g):
 try:return colliding_pairs(g.part_ids,np.asarray(decode_graph(g)))==[]
 except (FileNotFoundError,KeyError):return False
def signature(g,nodes):return tuple(sorted(Counter(CAT.id_to_name[g.part_ids[i]] for i in nodes).items()))
def boundary(g,nodes):
 s=set(nodes);cross=[(i,int(e['b']) if int(e['a']) in s else int(e['a']),int(e['a']) if int(e['a']) in s else int(e['b'])) for i,e in enumerate(g.edges) if (int(e['a']) in s)!=(int(e['b']) in s)]
 assert len(cross)==1;return cross[0]
def crop(g,copies,anchors,budget):
 protected={v for c in copies for v in c}|set(anchors);keep=set(range(len(g.part_ids)));adj=defaultdict(set)
 for e in g.edges:a,b=int(e['a']),int(e['b']);adj[a].add(b);adj[b].add(a)
 while len(keep)>budget:
  leaves=[v for v in keep-protected if len(adj[v]&keep)<=1]
  if not leaves:return None
  v=max(leaves);keep.remove(v)
 if len(keep)<.8*len(g.part_ids):return None
 order=sorted(keep);mapping={v:i for i,v in enumerate(order)};edges=g.edges[[int(e['a']) in keep and int(e['b']) in keep for e in g.edges]].copy()
 edges['a']=[mapping[int(x)] for x in edges['a']];edges['b']=[mapping[int(x)] for x in edges['b']]
 return connected_graph([g.part_ids[v] for v in order],[g.colors[v] for v in order],edges),[[mapping[v] for v in c] for c in copies],[mapping[v] for v in anchors],order
def clone(g,module,old_anchor,new_anchor,conn):
 n=len(g.part_ids);mapping={v:n+i for i,v in enumerate(module)};s=set(module);edges=[]
 for e in g.edges:
  a,b=int(e['a']),int(e['b'])
  if a in s and b in s:
   r=e.copy();r['a']=mapping[a];r['b']=mapping[b];edges.append(r)
  elif (a in s)!=(b in s):
   assert old_anchor in (a,b);r=e.copy()
   if a in s:r['a']=mapping[a];r['b']=new_anchor;r['b_conn']=conn
   else:r['b']=mapping[b];r['a']=new_anchor;r['a_conn']=conn
   edges.append(r)
 out=connected_graph(list(g.part_ids)+[g.part_ids[v] for v in module],list(g.colors)+[g.colors[v] for v in module],np.concatenate([g.edges,np.array(edges,dtype=g.edges.dtype)]))
 return out,list(mapping.values()),mapping
def dfs_sample(g,missing,root,seed,priority=None):
 sample=training_walk(g,missing,root=root,seed=seed)
 if priority is None:priority={v:i for i,v in enumerate(sample.walk.order)}
 raw=[root]+sorted((v for v in range(len(g.part_ids)) if v!=root),key=lambda v:priority.get(v,len(priority)+v))
 order=training_dfs(raw,sample.walk.parent_of);assert set(order[-len(missing):])==set(missing)
 tree=_build_tree(g,order,sample.walk.parent_of);assert check_dfs(tree)
 prefix,target=split_text(serialize_tree(tree),len(order)-len(missing));return prefix,target,order,tree,priority
def edit_options(g,copies,anchors):
 joints=[stack_joint(g,c,a) for c,a in zip(copies,anchors)]
 if all(joints):yield {'kind':'stack_insert','joints':joints,'part_id':g.part_ids[joints[0][0]]}
 # Append at a free connector on the existing module, preferably using a part already in it.
 for local in reversed(range(len(copies[0]))):
  nodes=[c[local] for c in copies]
  if len({g.part_ids[v] for v in nodes})!=1:continue
  for side in ('stud','hole'):
   ports=[free_connectors(g,v,side,('stud','open') if side=='stud' else ('hole','tube')) for v in nodes]
   common=sorted(set.intersection(*map(set,ports)))
   if not common:continue
   candidates=list(dict.fromkeys([g.part_ids[v] for v in reversed(copies[0])]))
   for name in ['plate 1x1 round','plate 1x1','brick 1x1 round with hollow stud']:
    candidates.append(CAT.name_to_id[name])
   for pid in dict.fromkeys(candidates):
    cs=_part_info(pid).conns;holes=[i for i,c in enumerate(cs) if c[0]=='hole' and c[1] in ('hole','tube')];studs=[i for i,c in enumerate(cs) if c[0]=='stud' and c[1] in ('stud','open')]
    if not holes or not studs:continue
    for port in common[:2]:yield {'kind':'tip_extension','sites':[(v,port,side) for v in nodes],'part_id':pid,'color':g.colors[nodes[0]],'hole':holes[0],'stud':studs[0]}
def edited(g,opt,indices):
 if opt['kind']=='stack_insert':return stretch(g,[opt['joints'][i] for i in indices],4)
 desc=[(opt['part_id'],opt['color'],opt['hole'],opt['stud'])]*4
 return decorate(g,[opt['sites'][i] for i in indices],desc)
def make_pair(g,copies,anchors,opt,sid,origin,retained,original_count):
 t=len(copies)-1;others=list(range(t));root=anchors[t];seed=20261122+sid*1009
 gt,at=edited(g,opt,[t]);go,ao=edited(g,opt,others);ga,aa=edited(g,opt,others+[t])
 if max(len(gt.part_ids),len(go.part_ids),len(ga.part_ids))>100 or not all(clear(x) for x in (g,gt,go,ga)):return None
 base=dfs_sample(g,copies[t],root,seed);priority=base[-1];rows=[];orders=[];bodies=[];missing_ref=[]
 for condition,gs,gl,lm in [('original',g,gt,copies[t]+at[0]),('stretched',go,ga,copies[t]+aa[-1])]:
  sh=dfs_sample(gs,copies[t],root,seed,priority);lo=dfs_sample(gl,lm,root,seed,priority);assert sh[0]==lo[0]
  r={'id':f'val:{sid}:{condition}','source_key':f'val:{sid}','source':sid,'condition':condition,'prefix':sh[0],'target_short':sh[1],'target_long':lo[1],'n_prefix':len(gs.part_ids)-len(copies[t]),'k_short':len(copies[t]),'k_long':len(lm),'r':4,'prime':0,'expected':'short' if condition=='original' else 'long','seed':seed,'copies':len(copies),'layout':'training_DFS_preorder','origin':origin}
  head=shared_head(r,1);r.update(prefix=r['prefix']+head,target_short=r['target_short'][len(head):],target_long=r['target_long'][len(head):],n_prefix=r['n_prefix']+1,k_short=r['k_short']-1,k_long=r['k_long']-1,prime=1)
  for text in [r['prefix'],r['prefix']+r['target_short'],r['prefix']+r['target_long']]:
   tree=parse_sample(text);assert tree.error is None and check_dfs(tree.tree);n,fail,coll=score_text(text);assert fail is None and coll==[] and n<=100
  r['prefix_sha256']=sha(r['prefix']);r['prefix_collisions']=[];rows.append(r)
  body=set(range(len(g.part_ids)))-set(v for c in copies for v in c);order=lo[2];mats=np.asarray(decode_graph(tree_to_graph(lo[3])));pos={v:i for i,v in enumerate(order)}
  bodies.append({v:mats[pos[v]] for v in body});orders.append([v for v in order if v in body]);missing_ref.append(mats[pos[copies[t][0]]])
 assert orders[0]==orders[1] and all(np.allclose(bodies[0][v],bodies[1][v],atol=1e-4) for v in bodies[0])
 record={'source':sid,'origin':origin,'source_original_parts':original_count,'source_retained_parts':len(retained),'retained_sampled_tree_node_indices':retained,'source_sampling_seed':20261122+sid,'module_size':len(copies[0]),'module_part_names':[CAT.id_to_name[g.part_ids[v]] for v in copies[0]],'copies':len(copies),'module_signature':signature(g,copies[0]),'extension':{k:v for k,v in opt.items() if k not in ('joints','sites')},'strict_full_DFS':True,'body_pose_and_relative_order_unchanged':True,'rows':rows}
 return record
def legacy(sid,old):
 rows=[]
 for r0 in [r for r in old if r['source_key']==f'val:{sid}']:
  short=parse_sample(r0['prefix']+r0['target_short']).tree;long=parse_sample(r0['prefix']+r0['target_long']).tree;gs=tree_to_graph(short);gl=tree_to_graph(long);start=r0['n_prefix']-1;missing=list(range(start,len(short.parts)));ml=list(range(start,len(long.parts)));anchor=short.edges[start-1].parent;seed=20261122+sid*1009
  sh=dfs_sample(gs,missing,anchor,seed);lo=dfs_sample(gl,ml,anchor,seed,sh[-1]);assert sh[0]==lo[0]
  r=dict(r0,prefix=sh[0],target_short=sh[1],target_long=lo[1],n_prefix=start,k_short=len(missing),k_long=len(ml),prime=0,layout='training_DFS_preorder',origin='legacy_official_val_reordered_DFS',seed=seed)
  head=shared_head(r,1);r.update(prefix=r['prefix']+head,target_short=r['target_short'][len(head):],target_long=r['target_long'][len(head):],n_prefix=start+1,k_short=len(missing)-1,k_long=len(ml)-1,prime=1);r['prefix_sha256']=sha(r['prefix']);r['prefix_collisions']=[]
  assert check_dfs(parse_sample(r['prefix']+r['target_'+r['expected']]).tree);assert score_text(r['prefix']+r['target_'+r['expected']])[2]==[];rows.append(r)
 return {'source':sid,'origin':'legacy_official_val_reordered_DFS','rows':rows,'strict_full_DFS':True}
def classify(text):
 s=text.lower()
 groups=[('animal',['fox','cat','dog','bird','rabbit','dragon','dinosaur','elephant','horse','penguin','animal','bear','turtle','owl','duck','fish']),('vehicle',['car','truck','train','bus','plane','spaceship','spacecraft','helicopter','vehicle','boat','ship','motorcycle','rocket']),('plant',['flower','tree','plant','garden','cactus','bamboo']),('furniture',['chair','table','bench','bed','sofa','cabinet','desk','furniture','bookshelf']),('building',['house','castle','building','bridge','tower','street','wall','gate','temple','door','church']),('machine',['robot','machine','cannon','gun','instrument','engine','computer'])]
 return next((name for name,words in groups if any(w in s for w in words)),'other')
def process_source(sid,g,counts,seen_signatures):
 original_count=len(g.part_ids)
 if len(g.components)!=1 or original_count<12 or original_count>125:return None
 if any(c not in CAT.code_to_color for c in g.colors):return None
 if any(not (REPO/'baseline/bricknet/data/cvx'/f'{CAT.id_to_stem[p]}.cvx.obj').exists() for p in set(g.part_ids)):return None
 try:
  tree=sample_collision_free_tree(g,seed=20261122+sid,max_parts=100);text=serialize_tree(tree);parsed=parse_sample(text)
  if parsed.error or len(tree.parts)<.8*original_count or score_text(text)[2]!=[]:return None
  base=tree_to_graph(tree);n=len(base.part_ids)
  groups=pendant_copies(base,max_size=10,min_copies=2)
  eligible=[grp for grp in groups if 3<=grp.size<=10 and len(grp.copies)<=4 and len(signature(base,grp.copies[0]))>=2]
  for grp in eligible:
   sig=str(signature(base,grp.copies[0]))
   if seen_signatures[sig]>=4:continue
   cropped=crop(base,[list(c) for c in grp.copies],list(grp.anchors),100-4*len(grp.copies))
   if cropped is None:continue
   bg,cs,anchors,kept=cropped
   for oi,opt in enumerate(edit_options(bg,cs,anchors)):
    if oi>=28:break
    result=make_pair(bg,cs,anchors,opt,sid,'official_val_existing_repeat',kept,original_count)
    if result:return result
  children=defaultdict(list)
  for e in tree.edges:children[e.parent].append(e.child)
  modules=[]
  for root in range(1,n):
   nodes=[];todo=[root]
   while todo:v=todo.pop();nodes.append(v);todo.extend(children[v])
   if not 3<=len(nodes)<=10 or len(signature(base,nodes))<2:continue
   modules.append(sorted(nodes))
  modules.sort(key=lambda m:(abs(len(m)-5),sha(str((sid,m)))))
  for module in modules[:12]:
   sig=str(signature(base,module))
   if seen_signatures[sig]>=4:continue
   ei,anchor,attach=boundary(base,module);e=base.edges[ei]
   if e['family']!='stud':continue
   anchor_conn=int(e['a_conn'] if int(e['a'])==anchor else e['b_conn']);c0=_part_info(base.part_ids[anchor]).conns[anchor_conn]
   possible=[]
   for a,pid in enumerate(base.part_ids):
    if a in module:continue
    for ci,c in enumerate(_part_info(pid).conns):
     if c[:3]!=c0[:3]:continue
     used={int(x['a_conn']) for x in base.edges if int(x['a'])==a}|{int(x['b_conn']) for x in base.edges if int(x['b'])==a}
     if ci in used:continue
     possible.append((a,ci))
   possible.sort(key=lambda ac:(ac[0]!=anchor,base.part_ids[ac[0]]!=base.part_ids[anchor],abs(ac[0]-anchor),ac))
   for a,ci in possible[:24]:
    cropped=crop(base,[module],[anchor,a],100-len(module)-8)
    if cropped is None:continue
    bg,cs,aa,kept=cropped;module2=cs[0];old_anchor,new_anchor=aa
    cg,copy,mapping=clone(bg,module2,old_anchor,new_anchor,ci)
    if not clear(cg):continue
    copies=[module2,copy]
    for oi,opt in enumerate(edit_options(cg,copies,[old_anchor,new_anchor])):
     if oi>=20:break
     result=make_pair(cg,copies,[old_anchor,new_anchor],opt,sid,'official_val_duplicated_own_subassembly',kept,original_count)
     if result:return result
 except (AssertionError,ValueError,IndexError,KeyError,FileNotFoundError,np.linalg.LinAlgError) as e:
  counts['error_'+type(e).__name__]+=1
 return None
def main():
 start=time.time();graphs=read_graphs(REPO/'baseline/bricknet/data/dataset/val.npz.xz');caps={int(r['id']):r['caption'] for r in map(json.loads,lzma.open(REPO/'baseline/bricknet/data/dataset/captions_val.jsonl.xz','rt'))};old=[json.loads(s) for s in (REPO/'runs/viewer_parse_fox_fix_20261121/p1_inputs.jsonl').open()];oldids=list(dict.fromkeys(int(r['source_key'].split(':')[1]) for r in old));records=[];counts=Counter();signatures=Counter();buckets=Counter()
 for sid in oldids:
  r=legacy(sid,old);r.update(caption=caps[sid],category=classify(caps[sid]),label=f'#{sid} · '+('狐狸' if sid==114 else caps[sid].split('.')[0][:70]));records.append(r);buckets[r['category']]+=1
 candidates=[i for i in range(len(graphs)) if i not in oldids];candidates.sort(key=lambda i:sha(f'official-dfs:{i}'))
 # Round-robin source categories before seeing any model output.
 bycat=defaultdict(list)
 for i in candidates:bycat[classify(caps.get(i,''))].append(i)
 ordered=[]
 while any(bycat.values()):
  for key in sorted(bycat):
   if bycat[key]:ordered.append(bycat[key].pop())
 for sid in ordered:
  if len(records)>=100:break
  category=classify(caps.get(sid,''))
  if buckets[category]>=24 or (category=='furniture' and buckets[category]>=10):continue
  cache=ROOT/'candidates'/f'{sid}.json';cache.parent.mkdir(exist_ok=True)
  if cache.exists():r=json.loads(cache.read_text())
  else:
   r=process_source(sid,graphs[sid],counts,signatures);atom(cache,r)
  if r:
   sig=str(tuple(tuple(x) for x in r['module_signature']))
   if signatures[sig]>=4:continue
   signatures[sig]+=1;r.update(caption=caps[sid],category=category,label=f'#{sid} · '+caps[sid].split('.')[0][:70]);records.append(r);buckets[category]+=1;print('ACCEPT',len(records),sid,category,r['origin'],r['module_part_names'],flush=True)
  counts['tried']+=1
  atom(ROOT/'build_status.json',{'accepted':len(records),'tried':counts['tried'],'categories':dict(buckets),'elapsed_seconds':time.time()-start})
 atom(ROOT/'sources.json',records);rows=[row for r in records for row in r['rows']];(ROOT/'p1_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 report={'sources':len(records),'official_val_sources':len(records),'synthetic_template_sources':0,'conditions':['original','stretched'],'prime':1,'strict_DFS':True,'categories':dict(buckets),'construction_counts':dict(Counter(r['origin'] for r in records)),'errors':dict(counts),'seconds':time.time()-start,'training_sampler_sha256':hashlib.sha256((TRAIN/'dfs_sampling.py').read_bytes()).hexdigest(),'training_dfs_sha256':hashlib.sha256((TRAIN/'prepare_corpus.py').read_bytes()).hexdigest(),'state':'ready_for_visual_review' if len(records)==100 else 'insufficient_candidates'}
 atom(ROOT/'DATASET_STATUS.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
