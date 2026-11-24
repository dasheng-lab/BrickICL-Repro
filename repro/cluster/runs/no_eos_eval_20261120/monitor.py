import json,time
from common import ROOT,atomic
while True:
    p=json.loads((ROOT/'plan.json').read_text());results={};times=[]
    for spec in p['models']:
        dest=ROOT/'results'/spec['key']
        if (dest/'GENERATED.json').exists():times.append(json.loads((dest/'GENERATED.json').read_text())['elapsed_seconds'])
        if (dest/'metrics.json').exists():
            r=json.loads((dest/'metrics.json').read_text());results[spec['key']]={k:r[k] for k in ('no_eos_valid_length','no_eos_collision_free_length','collision_coverage')}
    r={'scored':len(results),'total':12,'results':results,'generated':len(times),
       'estimated_inference_remaining_hours':(12-len(times))*sum(times[-3:])/len(times[-3:])/3600 if times else None,'unix_time':time.time()}
    if (ROOT/'status.json').exists():r['inference']=json.loads((ROOT/'status.json').read_text())
    atomic(ROOT/'PROGRESS.json',r);print(json.dumps(r),flush=True)
    if (ROOT/'COMPLETED.json').exists() or (ROOT/'FINISH_FAILED.json').exists():break
    time.sleep(45)
