import itertools,json,lzma
from pathlib import Path
root=Path('runs/dfs_posttrain_20261114')
result={}
for split in ('pt','sft'):
    checked=0;max_parts=0
    with lzma.open(f'baseline/bricknet/data/dataset/paths_{split}.jsonl.xz','rt') as f:
        for line in itertools.islice(f,50000):
            row=json.loads(line)
            lines=row['path'].splitlines()
            assert len(lines)%2==1
            n=(len(lines)+1)//2
            assert n==row['nodes'],(split,row['source'],n,row['nodes'])
            max_parts=max(max_parts,n);checked+=1
    result[split]={'checked':checked,'metadata_matches_text':True,'max_parts':max_parts}
(root/'PATH_LENGTH_FIELDS_CHECK.json').write_text(json.dumps(result,indent=2))
print(result,flush=True)
