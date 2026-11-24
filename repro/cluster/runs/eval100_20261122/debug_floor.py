import json
from bricknet import parse_ldr,load_catalog
from bricknet.graph import _part_info
import numpy as np
c=load_catalog()
for stem in ['3032','3035','3036','3033']:
 for mat in ['1 0 0 0 1 0 0 0 1','0 0 1 0 1 0 -1 0 0']:
  text=f'1 71 0 0 0 {mat} {stem}.dat\n';g=parse_ldr(text);info=_part_info(g.part_ids[0]);pts=[]
  for i,cc in enumerate(info.conns):
   if cc[0]=='stud':pts.append((g.transforms[0]@info.frames[i])[:3,3])
  p=np.array(pts);print(stem,mat,c.id_to_stem[g.part_ids[0]],p.min(0),p.max(0),g.transforms[0].tolist())
import build_synthetic as bs
for mat in ['1 0 0 0 1 0 0 0 1','0 0 1 0 1 0 -1 0 0']:
 for x,z in [(-20,-40),(0,0),(-20,-20)]:
  for y in [-24,8]:
   g=parse_ldr(f'1 71 0 0 0 {mat} 3032.dat\n1 72 {x} {y} {z} 1 0 0 0 1 0 0 0 1 3003.dat\n')
   print('PAIR',mat,x,y,z,len(g.edges))
try:bs.make(0,0)
except Exception as e:print('ERROR',e)
