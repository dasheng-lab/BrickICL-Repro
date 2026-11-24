"""Read-only status of the DFS data and four post-training runs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
status={}
if (root/'SUPERSEDED.json').exists():
    status['superseded']=json.loads((root/'SUPERSEDED.json').read_text())
    status['replacement_directory']=str(root.parent/'dfs_curated_20261114')
for name in ('raw_v2','raw_v3'):
    path=root/name/'metadata.json'
    status[name]=json.loads(path.read_text())['counts'] if path.exists() else 'building or queued'
for name in ('official_sft_audit.json','official_pt_audit.json','TRAINING_READY.json','COMPLETED.json'):
    if (root/name).exists():
        data=json.loads((root/name).read_text())
        if name.startswith('official_'):
            data={k:v for k,v in data.items() if k!='length_histogram'}
        status[name]=data
corpus=root/'corpus/READY.json'
if corpus.exists():
    data=json.loads(corpus.read_text())
    status['corpus']={k:data[k] for k in ('status','counts','skips','caption_methods','cross_split_overlap')}
for arm in ('PT_M','SFT_M','PT_G','SFT_G'):
    folder=root/'training'/arm
    path=folder/'status.json'
    status[arm]=json.loads(path.read_text()) if path.exists() else 'queued'
    if 'superseded' in status:
        if isinstance(status[arm],dict):status[arm]['state']='stopped_old_plan'
        else:status[arm]='cancelled_old_queue'
print(json.dumps(status,ensure_ascii=False,indent=2))
