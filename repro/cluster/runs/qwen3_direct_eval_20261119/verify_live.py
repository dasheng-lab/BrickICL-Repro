import json,urllib.request,time,zipfile,shutil
from common import *

APP='http://127.0.0.1:17860';VIEW='http://127.0.0.1:18765'
def get(url):
    with urllib.request.urlopen(url,timeout=20) as f:return json.load(f)
def objects(value):
    if isinstance(value,dict):
        if 'adapters' in value and 'training_objective' in value:yield value
        for child in value.values():yield from objects(child)
    elif isinstance(value,list):
        for child in value:yield from objects(child)
    elif isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try:parsed=json.loads(value)
        except json.JSONDecodeError:return
        yield from objects(parsed)
def main():
    assert (ROOT/'PUBLISHED.json').exists();calls=[]
    config=get(APP+'/config');dropdown=next(c['props'] for c in config['components'] if c['props'].get('label')=='01 / 选择模型')
    assert len(dropdown['choices'])==11
    assert all('training_qwen3_direct/' not in c['props'].get('value','') for c in config['components'] if c['type']=='markdown')
    assert get(VIEW+'/qwen3_direct_compare.json')==json.loads((ROOT/'viewer_stage/qwen3_direct_compare.json').read_text())
    manifest=get(VIEW+'/models/manifest.json');assert sum(r.get('evaluation_run')==ROOT.name for r in manifest)==876
    prefix=next(r['prefix'] for r in read(ROOT/'p1_inputs.jsonl') if r['id']=='val:114:original')
    for spec in plan()['checkpoints']:
        key=f'direct-{spec["training_arm"]}-{Path(spec["checkpoint"]).name}'
        data=[key,'','PT','补全搭建序列','',prefix,3,64,1.,.95,20,20261119,1]
        request=urllib.request.Request(APP+'/gradio_api/call/generate',data=json.dumps({'data':data,'session_hash':'qwen3-direct-live-validation'}).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=20) as f:event=json.load(f)['event_id']
        result=None;kind=None
        with urllib.request.urlopen(APP+'/gradio_api/call/generate/'+event,timeout=180) as stream:
            for raw in stream:
                line=raw.decode().strip()
                if line.startswith('event:'):kind=line.split(':',1)[1].strip()
                elif line.startswith('data:') and kind in ('complete','error'):
                    result=json.loads(line[5:].strip())
                    if kind=='error':raise RuntimeError(result)
                    break
        assert result is not None
        matches=[r for r in objects(result) if r['adapters']==spec['adapters'] and r['base']==spec['base']]
        assert matches,(key,result)
        record=matches[-1];assert len(record['adapters'])==1
        (ROOT/'live_checks').mkdir(exist_ok=True);atomic(ROOT/'live_checks'/f'{spec["arm"]}.json',result)
        calls.append({'arm':spec['arm'],'key':key,'adapters':record['adapters'],'base':record['base'],'seconds':record['seconds'],'stop_reason':record['stop_reason']})
        print('LIVE VERIFIED',key,flush=True)
    atomic(ROOT/'LIVE_VERIFIED.json',{'model_calls':calls,'default_models':11,'new_viewer_entries':876,'ui_samples_excluded_from_metrics':True})
    atomic(ROOT/'COMPLETED.json',{'state':'completed','models':3,'completion_samples':864,'independent_samples':192,'published':True,'unix_time':time.time()})
    for name in ('COMPLETED.json','VERIFIED.json','PUBLISHED.json','LIVE_VERIFIED.json','VIEWER_VALIDATED.json','plan.json'):
        shutil.copy2(ROOT/name,ROOT/'delivery'/name)
    shutil.copy2(ROOT/'LOSS_WEB_WITHDRAWN.json',ROOT/'delivery/LOSS_WEB_WITHDRAWN.json')
    shutil.copy2(ROOT/'README.zh-CN.md',ROOT/'delivery/README.zh-CN.md')
    with zipfile.ZipFile(ROOT/'delivery/results_loss_viewer.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT/'delivery').iterdir()):
            if f.suffix!='.zip':z.write(f,f.name)
    print('Complete: metrics, loss, viewer and Gradio',flush=True)
if __name__=='__main__':main()
