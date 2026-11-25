import json
from collections import defaultdict
from pathlib import Path
root=Path('runs/dfs_posttrain_20261114')
out={}
for v in ('v2','v3'):
    rows=[json.loads(s) for s in (root/f'cache/SFT_G/{v}/train.index.jsonl').open()]
    families=defaultdict(set)
    for r in rows:families[r['source_family']].add(r['selected_parts'])
    counts={}
    for mode in ('PT','SFT'):
        used={r['row_index'] for l in (root/f'{mode}_schedule.jsonl').open() for r in json.loads(l) if r['dataset']==v}
        seen={rows[i]['source_family'] for i in used}
        missing=set(families)-seen
        counts[mode]={'unseen_families':len(missing),'unseen_only_100_parts':sum(families[f]=={100} for f in missing),
                      'total_families_only_100_parts':sum(x=={100} for x in families.values())}
    out[v]=counts
(root/'caption_audit/coverage_explanation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
