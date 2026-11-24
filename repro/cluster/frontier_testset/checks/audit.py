import json
import statistics
from pathlib import Path
from bricknet import parse_sample
from runs.dfs_posttrain_20261114.prepare_corpus import check_dfs
D=Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122/p1_inputs.jsonl')
F=Path('/home/shensx3017/brickicl_eval/frontier_testset/p1_inputs.jsonl')
a={r['id']:r for r in map(json.loads,D.open())}
b={r['id']:r for r in map(json.loads,F.open())}
assert a.keys()==b.keys()
print('same_prefixes',sum(a[k]['prefix']==b[k]['prefix'] for k in a))
print('strict_dfs_frontier',sum(check_dfs(parse_sample(r['prefix']+r['target_'+r['expected']]).tree) for r in b.values()))
print('median_prefix_chars_dfs',statistics.median(len(r['prefix']) for r in a.values()))
print('median_prefix_chars_frontier',statistics.median(len(r['prefix']) for r in b.values()))
