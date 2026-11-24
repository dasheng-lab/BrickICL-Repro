"""Stream all shipped paths; audit counts and DFS order without loading model weights."""
import json
import lzma
import time
from collections import Counter
from pathlib import Path

out=Path('runs/dfs_posttrain_20261114')
data=Path('baseline/bricknet/data/dataset')
def is_dfs(path):
    lines=path.splitlines()
    if not lines:return False
    active=[lines[0].split(' ',1)[0]]
    for i in range(1,len(lines),2):
        if i+1>=len(lines):return False
        parent=lines[i+1].split(' ',1)[0]
        if parent not in active:return False
        active=active[:active.index(parent)+1]
        active.append(lines[i].split(' ',1)[0])
    return True
result={}
for split in ('sft','pt'):
    lengths=Counter()
    checked=Counter()
    longest=0
    begin=time.time()
    with lzma.open(data/f'paths_{split}.jsonl.xz','rt') as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            n=row['nodes']
            lengths[n]+=1
            longest=max(longest,n)
            # Deterministically sampled across the entire file.
            if i%997==0:
                checked['rows']+=1
                checked['dfs']+=is_dfs(row['path'])
                checked['over_100']+=n>100
            if (i+1)%100000==0:
                print(split,i+1,'max',longest,'over100',sum(v for k,v in lengths.items() if k>100),flush=True)
    report={'rows':sum(lengths.values()),'max_parts':longest,
            'above_100':sum(v for k,v in lengths.items() if k>100),
            'at_100':lengths[100],'length_histogram':dict(sorted(lengths.items())),
            'dfs_checks':dict(checked),'seconds':time.time()-begin}
    result[split]=report
    (out/f'official_{split}_audit.json').write_text(json.dumps(report,indent=2))
    print(split,'DONE',report['rows'],longest,flush=True)
(out/'official_path_audit.json').write_text(json.dumps(result,indent=2))
