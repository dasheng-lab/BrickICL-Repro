import json
from transformers import AutoTokenizer
spec=next(s for s in json.load(open('plan.json'))['models'] if s['arm']=='plain')
tok=AutoTokenizer.from_pretrained(spec['base'],local_files_only=True)
rows=[json.loads(l) for l in open('p1_inputs.jsonl')]
assert all(tok.decode(tok.encode(r['prefix'],add_special_tokens=False)+tok.encode(r['target_long'],add_special_tokens=False),clean_up_tokenization_spaces=False)==r['prefix']+r['target_long'] for r in rows)
print('DLO_TOKEN_BOUNDARIES_VERIFIED',len(rows))
