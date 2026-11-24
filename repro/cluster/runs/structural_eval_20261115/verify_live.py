import json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
APP='http://127.0.0.1:17860'
VIEW='http://127.0.0.1:18765'
def get(url):
    with urllib.request.urlopen(url,timeout=20) as f:return json.load(f)
protocol=json.loads((ROOT/'protocol.json').read_text())
config=get(APP+'/config')
summary=(ROOT/'results_summary.md').read_text().strip()
assert summary in [r['props'].get('value','').strip() for r in config['components'] if r['type']=='markdown']
for name in ('unified_compare','nzd_order_compare'):
    live=get(VIEW+'/'+name+'.json');stage=json.loads((ROOT/'viewer_stage'/f'{name}.json').read_text())
    assert live==stage
assert get(VIEW+'/rollback_conditions.json')==json.loads((ROOT/'viewer_stage/rollback_conditions.json').read_text())
manifest=get(VIEW+'/models/manifest.json')
new=[r for r in manifest if r.get('evaluation_run')==ROOT.name]
assert len(new)==2596
assert not any(r.get('condition') in ('ablated','deleted') for r in new)
for e in manifest:
    if e.get('arm') in ('post_pt','post_sft','fullg_pt','fullg_sft'):
        assert e.get('evaluation_run')==ROOT.name
rows=[json.loads(s) for s in (ROOT/'p1_inputs.jsonl').open()]
prefix=next(r['prefix'] for r in rows if r['id']=='val:114:original')
keys={'post_pt':'post-PT-checkpoint-000750','post_sft':'post-SFT-checkpoint-000250',
      'fullg_pt':'fullg-PT-checkpoint-000500','fullg_sft':'fullg-SFT-checkpoint-000250'}
calls=[]
def metadata_objects(value):
    if isinstance(value,dict):
        if 'adapters' in value and 'training_objective' in value:
            yield value
        for child in value.values():yield from metadata_objects(child)
    elif isinstance(value,list):
        for child in value:yield from metadata_objects(child)
    elif isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try:parsed=json.loads(value)
        except json.JSONDecodeError:return
        yield from metadata_objects(parsed)
for arm,key in keys.items():
    data=[key,'','SFT' if 'sft' in arm else 'PT','补全搭建序列','',prefix,3,64,1.,.95,20,20261115,1]
    req=urllib.request.Request(APP+'/gradio_api/call/generate',data=json.dumps({'data':data,'session_hash':'structural-live-verification'}).encode(),
                               headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=20) as f:event=json.load(f)['event_id']
    kind=None;result=None
    with urllib.request.urlopen(APP+'/gradio_api/call/generate/'+event,timeout=180) as stream:
        for raw in stream:
            line=raw.decode().strip()
            if line.startswith('event:'):kind=line.split(':',1)[1].strip()
            elif line.startswith('data:') and kind in ('complete','error'):
                value=json.loads(line[5:].strip())
                if kind=='error':raise RuntimeError((arm,value))
                result=value;break
    assert result is not None
    checks_dir=ROOT/'live_checks';checks_dir.mkdir(exist_ok=True)
    (checks_dir/f'{arm}.json').write_text(json.dumps(result,ensure_ascii=False))
    matches=[m for m in metadata_objects(result) if m['adapters']==protocol['models'][arm]['adapters']]
    assert matches,(arm,[type(x).__name__ for x in result])
    metadata=matches[-1]
    assert metadata['adapters']==protocol['models'][arm]['adapters'],metadata['adapters']
    assert metadata['training_objective']==('M-only' if arm.startswith('post_') else 'Full-G')
    calls.append({'arm':arm,'checkpoint':metadata['adapters'][-1],'new_tokens':metadata['new_tokens'],
                  'seconds':metadata['seconds'],'stop_reason':metadata['stop_reason']})
    print('LIVE GENERATION VERIFIED',arm,metadata['adapters'][-1],flush=True)
report={'published_models_verified':len(new),'default_models':8,'samples':72,'model_calls':calls,
        'boards_match_staged':True,'main_metrics_include_ui_checks':False,'verified_unix':time.time()}
(ROOT/'LIVE_VERIFIED.json').write_text(json.dumps(report,indent=2))
(ROOT/'COMPLETED.json').write_text(json.dumps({'state':'completed','published':True,'evaluation_models':8,
    'completion_samples':2304,'physical_samples':512,'new_viewer_models':2596,'live_generation_checks':4},indent=2))
print(json.dumps(report),flush=True)
