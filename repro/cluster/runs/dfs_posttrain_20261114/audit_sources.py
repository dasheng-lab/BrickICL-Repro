import io,json,lzma
from collections import Counter
from pathlib import Path
import numpy as np
root=Path('runs/dfs_posttrain_20261114')
summary={}
for name in ('pt','sft'):
    path=Path('runs/posttrain_icl_v2/pt.cache.npz') if name=='pt' else io.BytesIO(lzma.open('baseline/bricknet/data/dataset/sft.npz.xz','rb').read())
    with np.load(path,allow_pickle=False) as data:
        lengths=np.diff(data['node_ptr'])
    summary[name]={'source_models':len(lengths),'above_100':int((lengths>100).sum()),
                   'max_parts':int(lengths.max()),'mean_parts':float(lengths.mean()),
                   'above_384':int((lengths>384).sum()),
                   'part_histogram':dict(sorted(Counter(map(int,lengths)).items()))}
(root/'source_model_audit.json').write_text(json.dumps(summary,indent=2))
print({k:{a:b for a,b in v.items() if a!='part_histogram'} for k,v in summary.items()},flush=True)
