"""Select dev-best checkpoints with matching completed evaluation provenance."""
import json
from dataclasses import replace
from pathlib import Path
from .backend import POSTTRAIN, registry

def evaluated_registry(show_all=False):
    all_models=registry()
    best_models={}
    for objective,folder,eval_folder,prefix in [
        ('M-only','bricknet_06_reference_last','history_first_brick06','post'),
        ('Full-G','bricknet_06_fullg','history_first_brick06','fullg'),
    ]:
        for mode in ('PT','SFT'):
            try:
                run=POSTTRAIN/'runs'/folder/mode
                best=Path(json.loads((run/'best.json').read_text())['path']).resolve()
                completed=json.loads((run/'COMPLETED.json').read_text())
                if completed.get('state')!='completed':continue
                arm=f'{prefix}_{mode.lower()}'
                ev=POSTTRAIN/'runs'/eval_folder
                provenance=json.loads((ev/f'provenance.{arm}.json').read_text())
                if Path(provenance['adapters'][-1]).resolve()!=best:continue
                # Do not label a checkpoint as tested based only on a started run.
                for suite in ('p0','p1'):
                    inputs=[json.loads(s)['id'] for s in (ev/f'{suite}_triads.jsonl').open()]
                    outputs=[json.loads(s) for s in (ev/f'{suite}.{arm}.jsonl').open()]
                    if len(outputs)!=len(inputs) or {r['id'] for r in outputs}!=set(inputs):raise ValueError('incomplete evaluation')
                    if not all(len(r.get('samples',[]))==4 for r in outputs):raise ValueError('incomplete generations')
                key=f'{prefix}-{mode}-{best.name}'
                spec=all_models[key]
                if Path(spec.adapters[-1]).resolve()!=best:continue
                best_models[key]=replace(spec,label=f'★ {objective} · 0.6B · {mode} · step {int(best.name.split("-")[-1])} · dev-best／已完成统一评测')
            except (OSError,ValueError,KeyError,TypeError):
                continue
    official={k:all_models[k] for k in ('0.6B-PT','0.6B-SFT') if k in all_models}
    selected={**best_models,**official}
    if show_all:selected.update({k:v for k,v in all_models.items() if k not in selected})
    return selected

def evaluation_summary():
    try:return (POSTTRAIN / 'runs/history_first_brick06/reports/GRADIO_SUMMARY.zh-CN.md').read_text(encoding='utf-8')
    except OSError:return '评测总表暂不可用。'
