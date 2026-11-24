import json
from pathlib import Path
root=Path(__file__).resolve().parent
result={'directory':str(root)}
for file in ('SUPERSEDED.json','EXTRA_READY.json','DATA_READY.json','TRAINING_READY.json','COMPLETED.json'):
    path=root/file
    if path.exists():result[file]=json.loads(path.read_text())
for arm in ('PT_M','SFT_M','PT_G','SFT_G'):
    path=root/'training'/arm/'status.json'
    result[arm]=json.loads(path.read_text()) if path.exists() else 'queued'
    if 'SUPERSEDED.json' in result:
        if isinstance(result[arm],dict):result[arm]['state']='stopped_previous_plan'
        else:result[arm]='cancelled_previous_queue'
print(json.dumps(result,ensure_ascii=False,indent=2))
