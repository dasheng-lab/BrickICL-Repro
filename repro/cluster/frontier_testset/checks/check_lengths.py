import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer
base=Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122')
front=Path('/home/shensx3017/brickicl_eval/frontier_testset')
spec=next(s for s in json.loads((front/'plan.json').read_text())['models'] if s['arm']=='plain')
tok=AutoTokenizer.from_pretrained(spec['base'],local_files_only=True)
for name,p in [('dfs',base/'p1_inputs.jsonl'),('frontier',front/'p1_inputs.jsonl')]:
 rows=[json.loads(l) for l in p.open()]
 lengths=[len(tok.encode(r['prefix'],add_special_tokens=False)) for r in rows]
 targets=[len(tok.encode(r['target_long'],add_special_tokens=False)) for r in rows]
 print(name,'prefix median',statistics.median(lengths),'target median',statistics.median(targets),'prefix max',max(lengths))
