"""Expose checkpoints whose saved inference records match the current evaluation."""
import json
from dataclasses import replace
from pathlib import Path
from .backend import POSTTRAIN,registry

EVAL=POSTTRAIN/'runs/structural_eval_20261115'
def evaluated_registry(show_all=False):
    all_models=registry()
    selected={}
    direct = POSTTRAIN/'runs/qwen3_direct_eval_20261119'
    if (direct/'VERIFIED.json').exists() and json.loads((direct/'VERIFIED.json').read_text())['all_completed']:
        plan=json.loads((direct/'plan.json').read_text())
        for record in plan['checkpoints']:
            cp=Path(record['checkpoint']);key=f'direct-{record["training_arm"]}-{cp.name}'
            spec=all_models[key]
            assert spec.adapters==(str(cp),) and spec.base==record['base']
            assert Path(json.loads((cp.parent/'best.json').read_text())['path'])==cp
            assert (direct/'results'/record['key']/'metrics.json').exists()
            for prime in (0,1):
                rows=[json.loads(s) for s in (direct/'results'/record['key']/f'p{prime}.jsonl').open()]
                assert len(rows)==36 and all(len(r['samples'])==4 and tuple(r['adapters'])==spec.adapters for r in rows)
            selected[key]=replace(spec,label='★ '+record['label']+f' · step {record["step"]} · dev-best · 已完成评测')
    if (EVAL/'SCORING_COMPLETED.json').exists():
        protocol=json.loads((EVAL/'protocol.json').read_text())
        keys={'post_pt':'post-PT-','post_sft':'post-SFT-','fullg_pt':'fullg-PT-','fullg_sft':'fullg-SFT-'}
        for arm in ('post_pt','post_sft','fullg_pt','fullg_sft','plain','masked','frozen_pt','frozen_sft'):
            record=protocol['models'][arm]
            if arm in keys:
                cp=Path(record['checkpoint'])
                best=json.loads((cp.parent/'best.json').read_text())
                assert Path(best['path']).resolve()==cp.resolve()
                assert json.loads((cp.parent/'COMPLETED.json').read_text())['state']=='completed'
                key=keys[arm]+cp.name
            elif arm in ('plain','masked'):key=f'nzd-{arm}-PT-{record["step"]}'
            else:key='0.6B-'+('SFT' if arm=='frozen_sft' else 'PT')
            spec=all_models[key]
            assert tuple(record['adapters'])==spec.adapters,(arm,spec.adapters)
            for prime in (0,1):
                inputs={json.loads(s)['id'] for s in (EVAL/f'p{prime}_inputs.jsonl').open()}
                outputs=[json.loads(s) for s in (EVAL/f'p{prime}.{arm}.jsonl').open()]
                assert len(inputs)==len(outputs)==36 and {r['id'] for r in outputs}==inputs
                assert all(len(r['samples'])==4 and tuple(r['adapters'])==spec.adapters for r in outputs)
            label=('新版 ' if arm in keys else '')+record['label']+(f' · step {record["step"]} · dev-best' if 'step' in record else '')
            selected[key]=replace(spec,label='★ '+label+' · 已完成新版评测')
    if show_all:selected.update({k:v for k,v in all_models.items() if k not in selected})
    return selected

def evaluation_summary():
    try:return Path(__file__).with_name('results_summary.md').read_text(encoding='utf-8')
    except OSError:return '评测总表暂不可用。'
