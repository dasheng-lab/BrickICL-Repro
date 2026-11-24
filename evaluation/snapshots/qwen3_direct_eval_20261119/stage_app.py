import json,shutil,subprocess,os
from pathlib import Path
from common import *

SERVICE=Path('/home/sunshq114514/brickicl-gradio-service')
def main():
    stage=ROOT/'gradio_stage';stage.mkdir(exist_ok=True)
    shutil.copytree(SERVICE/'brickicl',stage/'brickicl',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('app.py','backend.py','evaluated_models.py'):
        src=ROOT/'brickicl/playground'/name;compile(src.read_text(),str(src),'exec')
        shutil.copy2(src,stage/'brickicl/playground'/name)
    shutil.copy2(ROOT/'results_summary.md',stage/'brickicl/playground/results_summary.md')
    baseline=stage/'baseline'
    if not baseline.exists():baseline.symlink_to(SERVICE/'baseline')
    code='''import json
from brickicl.playground.backend import registry,resolve_spec
from brickicl.playground.evaluated_models import evaluated_registry
from brickicl.playground.app import build_ui
from pathlib import Path
root=Path('/home/sunshq114514/brickICL/runs/qwen3_direct_eval_20261119')
plan=json.loads((root/'plan.json').read_text())
selected=evaluated_registry();assert len(selected)==11
for r in plan['checkpoints']:
    cp=Path(r['checkpoint']);key=f'direct-{r["training_arm"]}-{cp.name}'
    spec=selected[key];assert spec.adapters==tuple(r['adapters']) and spec.base==r['base'] and spec.mode=='PT'
    custom=resolve_spec(key,str(cp),'PT');assert custom.adapters==spec.adapters and custom.base==spec.base
assert sum(k.startswith('direct-') for k in registry())==27
app=build_ui();app=app[0] if isinstance(app,tuple) else app;cfg=app.config
dropdown=next(c['props'] for c in cfg['components'] if c['props'].get('label')=='01 / 选择模型')
assert dropdown['value']=='direct-M_only-checkpoint-000750'
summary=(root/'results_summary.md').read_text().strip()
assert summary in [c['props'].get('value','').strip() for c in cfg['components'] if c['type']=='markdown']
assert all('training_qwen3_direct/' not in c['props'].get('value','') for c in cfg['components'] if c['type']=='markdown')
(root/'REGISTRY_VALIDATED.json').write_text(json.dumps({'models':list(selected),'all_models':len(registry()),'default':dropdown['value']}))
print('STAGED APP VERIFIED',len(selected),len(registry()))
'''
    script=stage/'check.py';script.write_text(code)
    env=dict(os.environ,PYTHONPATH=str(stage),BRICKICL_POSTTRAIN_ROOT=str(REPO))
    subprocess.run(['/home/sunshq114514/.venvs/brickicl-gradio/bin/python',str(script)],cwd=stage,env=env,check=True)
if __name__=='__main__':main()
