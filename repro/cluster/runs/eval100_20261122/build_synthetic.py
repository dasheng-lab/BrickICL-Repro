"""Construct new evaluation-only LEGO objects from empty LDraw scenes, not corpus augmentations."""
import hashlib,json,re
from collections import Counter
from pathlib import Path
import numpy as np
from bricknet import load_catalog,parse_ldr,parse_sample,score_text,decode_graph,tree_to_graph
from brickicl.sampling.masked_walk import masked_walk
from brickicl.sampling.edits import stack_joint,stretch
from brickicl.eval.length_probe import shared_head
from brickicl.posttrain.build_dataset import fingerprint
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];CAT=load_catalog()
FAMILIES=['靠背长凳','扶手座椅','高低床架','双侧书架','侧柜书桌','阶梯展台','控制台','四足登陆器','盆栽展示台','围栏庭院','门廊平台','仪器工作台']
PROFILES=[('brick 2x2','plate 2x2',24,8,0,0),('brick 1x2','plate 1x2',24,8,0,-10),('brick 1x1','plate 1x1',24,8,-10,-10),('brick 1x1 round with hollow stud','plate 1x1 round',24,8,-10,-10),('brick 2x2 round reinforced','plate 2x2 round with axlehole',24,8,0,0),('plate 1x2','brick 1x2',8,24,0,-10),('plate 2x2','brick 2x2',8,24,0,0),('plate 1x1 round','brick 1x1 round with hollow stud',8,24,-10,-10)]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def stem(name):return CAT.id_to_stem[CAT.name_to_id[name]]
def make(family,shape):
 sizes=[(4,6),(4,8),(6,8),(6,10),(6,12),(4,10),(8,8),(4,12)];w,l=sizes[shape];nx,nz=w//2,l//2
 lines=[];parts=[]
 def put(name,color,x,y,z):
  s=stem(name);parts.append((s,x,y,z));lines.append(f'1 {color} {x} {y} {z} 1 0 0 0 1 0 0 0 1 {s}.dat');return len(parts)-1
 put(f'plate {w}x{l}',71,0,0,0)
 # Standard rectangular LDraw plates have their long side on X; turn the base so X is width.
 lines[0]=f'1 71 0 0 0 0 0 1 0 1 0 -1 0 0 {stem(f"plate {w}x{l}")}.dat'
 color=[19,4,1,2,70,14,72,15,25,9,28,7][family];heights=[];upward=family in (8,9,10)
 for ix in range(nx):
  for iz in range(nz):
   if family==0:h=3 if iz==nz-1 else 1
   elif family==1:h=4 if iz==nz-1 else (2 if ix in (0,nx-1) else 1)
   elif family==2:h=3 if iz==nz-1 else (2 if iz==0 else 1)
   elif family==3:h=3 if ix in (0,nx-1) else (2 if iz==nz-1 else 0)
   elif family==4:h=2 if ix==0 else (1 if iz==nz-1 else 0)
   elif family==5:h=1+min(2,iz*3//nz)
   elif family==6:h=3 if iz==nz-1 and ix==nx//2 else (1 if iz>=nz//2 else 0)
   elif family==7:h=2 if ix==nx//2 and 0<iz<nz-1 else (1 if iz==nz//2 else 0)
   elif family==8:h=2 if (ix+iz)%2==0 else 0
   elif family==9:h=2 if ix in (0,nx-1) or iz==nz-1 else 0
   elif family==10:h=3 if ix in (0,nx-1) and iz>=nz-2 else (1 if iz==0 else 0)
   else:h=1+(2 if ix==nx//2 and iz==nz//2 else 0)
   if upward and ix in (0,nx-1) and iz in (0,nz-1):h=0
   x=-10*(w-1)+40*ix+10;z=-10*(l-1)+40*iz+10
   for level in range(h):put('brick 2x2',color,x,-24*(level+1),z)
   if h and (ix+iz)%3==family%3:put('plate 2x2',15,x,-24*h-8,z)
   heights.append([ix,iz,h])
 copies=[];anchors=[];brick_n=2+(shape%2);plate_n=1+(family%2)
 core,cap,core_h,cap_h,dx,dz=PROFILES[(family+3*shape)%len(PROFILES)]
 for ix,iz in [(0,0),(nx-1,0),(0,nz-1),(nx-1,nz-1)]:
  x=-10*(w-1)+40*ix+10+dx;z=-10*(l-1)+40*iz+10+dz;y=0 if upward else 8;copy=[]
  for name,h,col in [(core,core_h,72)]*brick_n+[(cap,cap_h,0)]*plate_n:
   if upward:y-=h
   copy.append(put(name,col,x,y,z))
   if not upward:y+=h
  copies.append(copy);anchors.append(0)
 text='\n'.join(lines)+'\n';g=parse_ldr(text)
 assert len(g.components)==1,('disconnected',family,shape,lines[:3],g.components)
 assert len(g.part_ids)+16<=100,('over100',len(g.part_ids)+16)
 recipe={'family':FAMILIES[family],'family_index':family,'footprint':[w,l],'height_map':heights,'leg_bricks':brick_n,'leg_plates':plate_n,'core_part':core,'cap_part':cap,'orientation':'upward' if upward else 'downward','construction':'new empty LDraw scene; no source LEGO was transformed, recolored or resampled to create another case'}
 return g,copies,anchors,text,recipe
def inputs(g,copies,anchors,sid):
 joints=[stack_joint(g,c,a) for c,a in zip(copies,anchors)];assert all(joints)
 t=0;others=[1,2,3];seed=100000+sid*1000
 g_t,(new_t,)=stretch(g,[joints[t]],4);g_o,_=stretch(g,[joints[j] for j in others],4);g_all,added=stretch(g,[joints[j] for j in others]+[joints[t]],4)
 rows=[];full=[]
 for cond,gs,gl,long_m,blocks in [('original',g,g_t,copies[t]+new_t,[copies[j] for j in others]),('stretched',g_o,g_all,copies[t]+added[-1],[copies[j]+added[i] for i,j in enumerate(others)])]:
  sh=masked_walk(gs,copies[t],root=0,seed=seed,blocks=blocks);lo=masked_walk(gl,long_m,root=0,seed=seed,blocks=blocks);assert sh.prefix==lo.prefix
  r={'id':f'synthetic:{sid}:{cond}','source_key':f'synthetic:{sid}','source':sid,'condition':cond,'n_prefix':sh.walk.n_prefix,'prefix':sh.prefix,'target_short':sh.target,'target_long':lo.target,'k_short':len(copies[t]),'k_long':len(long_m),'r':4,'prime':0,'expected':'short' if cond=='original' else 'long','seed':seed,'copies':4,'layout':'contiguous','origin':'newly_constructed_evaluation_only'}
  head=shared_head(r,1);r.update(prefix=r['prefix']+head,target_short=r['target_short'][len(head):],target_long=r['target_long'][len(head):],n_prefix=r['n_prefix']+1,k_short=r['k_short']-1,k_long=r['k_long']-1,prime=1)
  for txt in [r['prefix'],r['prefix']+r['target_short'],r['prefix']+r['target_long']]:
   n,invalid,coll=score_text(txt);assert invalid is None and coll==[] and n<=100,(sid,cond,n,invalid,coll)
   assert parse_sample(txt).error is None
  r['prefix_collisions']=[];r['prefix_sha256']=sha(r['prefix']);rows.append(r);full.append(r['prefix']+r['target_'+r['expected']])
 return rows,full
def main():
 out=ROOT/'synthetic';out.mkdir(exist_ok=True);old=[json.loads(s) for s in (REPO/'runs/viewer_parse_fox_fix_20261121/p1_inputs.jsonl').open()];assert len(old)==36
 sources=[];rows=[];reject=[];seen=set();sid=1000
 # Interleave design families so selecting 82 does not privilege one family. Construction precedes all model inference.
 for shape in range(8):
  for family in range(12):
   if len(sources)==82:break
   try:
    g,copies,anchors,ldr,recipe=make(family,shape);new,full=inputs(g,copies,anchors,sid)
    key=fingerprint(g);assert key not in seen,'duplicate color-ignored graph';seen.add(key)
    p=out/str(sid);p.mkdir(exist_ok=True);(p/'design.ldr').write_text(ldr);(p/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2))
    for condition,txt in zip(('original','stretched'),full):(p/f'{condition}.txt').write_text(txt)
    label=f'#{sid} · {recipe["family"]} {recipe["footprint"][0]}×{recipe["footprint"][1]}'
    sources.append({'source':sid,'source_key':f'synthetic:{sid}','label':label,'caption':f'新构造的{recipe["family"]}，底板{recipe["footprint"][0]}×{recipe["footprint"][1]}，四个对应重复部件（{recipe["core_part"]}）。正常组每部件{len(copies[0])}块；加长组每部件插入4块。','origin':'synthetic_eval_only','geometry_fingerprint':key,'design_sha256':sha(ldr),'recipe':recipe})
    rows.extend(new);sid+=1
   except Exception as e:reject.append({'family':family,'shape':shape,'error':str(e)})
 (ROOT/'CONSTRUCTION_REJECTIONS.json').write_text(json.dumps(reject,ensure_ascii=False,indent=2))
 assert len(sources)==82,(len(sources),Counter(r['error'].split(',')[0] for r in reject))
 for r in old:
  for txt in [r['prefix'],r['prefix']+r['target_'+r['expected']]]:
   n,invalid,c=score_text(txt);assert invalid is None and c==[],(r['id'],invalid,c)
 merged=old+rows;assert len(merged)==200 and len({r['source_key'] for r in merged})==100
 (ROOT/'p1_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged));(ROOT/'synthetic_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
 audit={'sources':100,'legacy_sources':18,'new_constructed_sources':82,'conditions':2,'prime':1,'samples_per_condition':4,'all_prefixes_and_expected_references_collision_free':True,'all_new_counterfactual_targets_collision_free':True,'max_parts':100,'new_designs_color_ignored_fingerprint_unique':len(seen),'rejections':reject,'synthetic_distribution_notice':'82 new procedural furniture/architectural/landing-platform designs; not 82 new natural official-val works; no corpus augmentation or repeated walk counted as a new work.'}
 (ROOT/'DATASET_VALIDATED.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
