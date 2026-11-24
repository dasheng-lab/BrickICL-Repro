from bricknet import load_catalog
from bricknet.graph import _part_info
import numpy as np
c=load_catalog()
print('PROFILES',[(n,n in c.name_to_id) for n in ['brick 1x1','brick 1x2','plate 1x1','plate 1x2','plate 1x1 round','brick 1x1 round with open stud','brick 2x2 round with axlehole','plate 2x2 round with axlehole']])
for name in ['plate 4x6','plate 4x8','plate 6x8','plate 6x10','plate 6x12','plate 4x10','plate 8x8','plate 8x16','brick 2x2','plate 2x2']:
 if name not in c.name_to_id:print(name,'MISSING');continue
 p=c.name_to_id[name];info=_part_info(p)
 print(name,c.id_to_stem[p],[(kind,np.round(np.asarray(info.frames)[[i for i,x in enumerate(info.conns) if x[0]==kind],:3,3],2).tolist()) for kind in ['stud']])
