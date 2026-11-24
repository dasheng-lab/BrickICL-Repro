import json
from pathlib import Path
from caption_rules import ROOT,load_bases,caption_for
import hashlib
bases=load_bases();result={}
for v in ('v2','v3'):
    sets={a:set() for a in ('PT','SFT')};rows=0
    for line in Path(f'runs/dfs_posttrain_20261114/corpus/{v}/train.jsonl').open():
        r=json.loads(line);cap,_=caption_for(r,bases);rows+=1
        sets['PT'].add(hashlib.sha256(r['full_dfs_path'].encode()).digest())
        if cap:sets['SFT'].add(hashlib.sha256((cap+'\n'+r['full_dfs_path']).encode()).digest())
    result[v]={'rows':rows,'unique_PT_G':len(sets['PT']),'unique_screened_SFT_G':len(sets['SFT'])}
(ROOT/'BASE_CAPACITY.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
