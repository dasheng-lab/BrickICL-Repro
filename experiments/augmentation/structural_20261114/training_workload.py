import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ARMS=('PT_M','SFT_M','PT_G','SFT_G')
HISTORY={'PT_M':'bricknet_06_reference_last/PT','SFT_M':'bricknet_06_reference_last/SFT',
         'PT_G':'bricknet_06_fullg/PT','SFT_G':'bricknet_06_fullg/SFT'}
def workload(cfg):
    root=Path(cfg['data_dir'])
    rows={v:[json.loads(s) for s in (root/v/'train.index.jsonl').open()] for v in ('v2','v3')}
    input_tokens=0;attention=0;projected=0
    for line in Path(cfg['schedule']).open():
        batch=[rows[r['dataset']][r['row_index']] for r in json.loads(line)]
        longest=max(r['length']-1 for r in batch);input_tokens+=len(batch)*longest;attention+=len(batch)*longest**2
        intervals=sorted((r['prefix_length']-1,r['length']-1) for r in batch)
        union=0;start,end=intervals[0]
        for lo,hi in intervals[1:]:
            if lo>end:union+=end-start;start,end=lo,hi
            else:end=max(end,hi)
        union+=end-start
        projected+=len(batch)*union
    return {'input_tokens':input_tokens,'attention_pairs':attention,'projected_tokens':projected}
result={}
for arm in ARMS:
    old=REPO/'runs'/HISTORY[arm]
    result[arm]={'current':workload(json.loads((ROOT/f'training/{arm}/config.json').read_text())),
                 'historical':workload(json.loads((old/'config.json').read_text())),
                 'historical_seconds':json.loads((old/'COMPLETED.json').read_text())['elapsed_seconds']}
    result[arm]['ratios']={k:result[arm]['current'][k]/result[arm]['historical'][k] for k in result[arm]['current']}
(ROOT/'training_workload.json').write_text(json.dumps(result,indent=2))
print(json.dumps({a:{'ratios':r['ratios'],'historical_hours':r['historical_seconds']/3600} for a,r in result.items()},indent=2))
