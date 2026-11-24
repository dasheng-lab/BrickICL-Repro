import importlib,json,sys,types
from pathlib import Path
root=Path('runs/structural_eval_20261115').resolve()
package=types.ModuleType('structural_stage')
package.__path__=[str(root/'gradio_stage/brickicl/playground')]
sys.modules[package.__name__]=package
mod=importlib.import_module('structural_stage.evaluated_models')
backend=importlib.import_module('structural_stage.backend')
models=mod.evaluated_registry()
assert len(models)==8
assert len(backend.sample_library())==72
expected={'post-PT-checkpoint-000750':'PT_M','post-SFT-checkpoint-000250':'SFT_M',
          'fullg-PT-checkpoint-000500':'PT_G','fullg-SFT-checkpoint-000250':'SFT_G'}
for key,arm in expected.items():
    cp=Path(models[key].adapters[-1])
    assert cp.parent.name==arm and 'structural_20261114' in str(cp)
assert not any('bricknet_06_reference_last' in a or 'bricknet_06_fullg' in a for s in backend.registry().values() for a in s.adapters)
report={'models':{k:{'label':v.label,'adapters':v.adapters} for k,v in models.items()},'samples':72}
(root/'REGISTRY_VALIDATED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'model_keys':list(models),'samples':72}))
