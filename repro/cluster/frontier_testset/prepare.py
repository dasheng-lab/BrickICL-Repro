import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
root=Path('/home/shensx3017/brickicl_eval/frontier_testset')
base=Path('/home/sunshq114514/brickICL/runs/eval100_official_dfs_20261122')
old=json.loads((base/'plan.json').read_text())
models=[s for s in old['models'] if s['arm'] in ('plain','masked')]
assert len(models)==2
tok=AutoTokenizer.from_pretrained(models[0]['base'],local_files_only=True)
rows=[json.loads(l) for l in (root/'p1_inputs.jsonl').open()]
rows.sort(key=lambda r:(len(tok.encode(r['prefix'],add_special_tokens=False)),r['id']))
groups=[]
for start in range(0,200,8):
    batch=rows[start:start+8]
    cap=max(512,64*max(r['k_long'] for r in batch)+256)
    assert max(len(tok.encode(r['prefix'],add_special_tokens=False)) for r in batch)+cap<=6400
    groups.append({'ids':[r['id'] for r in batch],'seed':20261122+start*1009,'max_new_tokens':cap})
plan={'models':models,'source_testset':str(base/'p1_inputs.jsonl'),
      'source_input_sha256':old['input_sha256'],
      'input_sha256':hashlib.sha256((root/'p1_inputs.jsonl').read_bytes()).hexdigest(),
      'scoring_sha256':hashlib.sha256((root/'reference_scoring.py').read_bytes()).hexdigest(),
      'strict_DFS':False,'layout':'masked_frontier_walk','sources':100,
      'conditions':['original','stretched'],'prime':1,'samples_per_condition':4,
      'temperature':1.,'top_k':20,'top_p':.95,'batch_plan':groups,
      'checkpoint_selection':'original development-best unchanged',
      'independent_generation':'reuse original 64-sample results; same checkpoints and no reference input'}
(root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
print('PLAN',len(models),len(groups),'max_new_tokens',max(g['max_new_tokens'] for g in groups))
