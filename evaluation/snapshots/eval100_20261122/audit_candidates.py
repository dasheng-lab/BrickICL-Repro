"""Count distinct source/subassembly tasks without changing stretch or model limits."""
import hashlib,json,time
from collections import Counter
from pathlib import Path
from bricknet import score_text,parse_sample
from brickicl.sampling.build import read_graphs
from brickicl.sampling.missing import pendant_copies
from brickicl.sampling.edits import stack_joint,stretch
from brickicl.sampling.masked_walk import masked_walk
from brickicl.eval.length_probe import shared_head
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def main():
 graphs=read_graphs(REPO/'baseline/bricknet/data/dataset/val.npz.xz');units=[];reasons=Counter();seen=set();start=time.time()
 for gid,g in enumerate(graphs):
  for gi,grp in enumerate(pendant_copies(g,max_size=12,min_copies=2)):
   joints=[stack_joint(g,c,a) for c,a in zip(grp.copies,grp.anchors)]
   if not all(joints) or len(g.part_ids)+4*len(grp.copies)>100:continue
   for t,copy in enumerate(grp.copies):
    try:
     others=[j for j in range(len(grp.copies)) if j!=t];seed=gid*1000+t*10
     g_t,(new_t,)=stretch(g,[joints[t]],4);g_o,_=stretch(g,[joints[j] for j in others],4);g_all,added=stretch(g,[joints[j] for j in others]+[joints[t]],4)
     plain=[grp.copies[j] for j in others];grown=[list(grp.copies[j])+added[i] for i,j in enumerate(others)]
     variants=[('original',g,g_t,list(copy)+new_t,plain),('stretched',g_o,g_all,list(copy)+added[-1],grown)];rows=[];invalid=False;collisions=[]
     for cond,gs,gl,long_m,blocks in variants:
      sh=masked_walk(gs,copy,seed=seed,blocks=blocks);lo=masked_walk(gl,long_m,seed=seed,blocks=blocks);assert sh.prefix==lo.prefix
      row={'source':gid,'group_index':gi,'target_index':t,'pair':f'{gid}-g{gi}-t{t}','condition':cond,'layout':'contiguous','r':4,'copies':len(grp.copies),'n_prefix':sh.walk.n_prefix,'k_short':len(copy),'k_long':len(long_m),'prefix':sh.prefix,'target_short':sh.target,'target_long':lo.target,'seed':seed}
      head=shared_head(row,1);row.update(prefix=row['prefix']+head,n_prefix=row['n_prefix']+1,k_short=row['k_short']-1,k_long=row['k_long']-1,target_short=row['target_short'][len(head):],target_long=row['target_long'][len(head):],prime=1)
      for target in ('short','long'):
       n,fail,coll=score_text(row['prefix']+row['target_'+target]);assert parse_sample(row['prefix']+row['target_'+target]).error is None
       if fail is not None or coll is None or n>100:invalid=True
       collisions.append(coll)
      rows.append(row)
     if invalid:reasons['parse_unknown_or_length']+=1;continue
     # Match the historical rule: stretching may not add collisions to the original geometry.
     if len(collisions[3])>len(collisions[0]):reasons['added_collision']+=1;continue
     key=(gid,tuple(copy))
     if key in seen:continue
     seen.add(key);clean=all(c==[] for c in collisions)
     units.append({'source':gid,'group_index':gi,'target_nodes':list(copy),'target_index':t,'all_variants_collision_free':clean,'collision_counts':[len(c) for c in collisions],'rows':rows})
    except Exception as e:reasons[type(e).__name__+': '+str(e)[:100]]+=1
 summary={'distinct_sources':len({u['source'] for u in units}),'distinct_source_target_tasks':len(units),'clean_distinct_sources':len({u['source'] for u in units if u['all_variants_collision_free']}),'clean_distinct_tasks':sum(u['all_variants_collision_free'] for u in units),'rejections':dict(reasons),'seconds':time.time()-start,'rule':'r4, original/stretched, prime1, max100 parts, repeat group<=12; alternative target copies/subassemblies allowed; no repeated walk seeds'}
 (ROOT/'CANDIDATE_AUDIT.json').write_text(json.dumps(summary,indent=2));(ROOT/'candidate_tasks.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in units));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
