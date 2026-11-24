"""Expose checkpoints whose saved inference records match the current evaluation."""
import json
from dataclasses import replace
from pathlib import Path
from .backend import POSTTRAIN,registry

EVAL=POSTTRAIN/'runs/structural_eval_20261115'
def evaluated_registry(show_all=False):
    all_models=registry()
    selected={}
    grpo=POSTTRAIN/'runs/grpo_eval_20261120'
    if (grpo/'VERIFIED.json').exists() and json.loads((grpo/'VERIFIED.json').read_text())['all_completed']:
        record=json.loads((grpo/'plan.json').read_text())['checkpoints'][0]
        key=record['registry_key'];spec=all_models[key]
        assert spec.adapters==tuple(record['adapters']) and len(spec.adapters)==3
        assert record['step']==record['dev_best']['step']==300
        for p in (0,1):
            rows=[json.loads(s) for s in (grpo/'results/grpo_pt'/f'p{p}.jsonl').open()]
            assert len(rows)==36 and all(len(r['samples'])==4 and tuple(r['adapters'])==spec.adapters for r in rows)
        selected[key]=replace(spec,label='★ GRPO PT · step 300 · dev-best · 已完成统一评测')
    recent=POSTTRAIN/'runs/qwen3_queue_results_20261120'
    if (recent/'VERIFIED.json').exists() and json.loads((recent/'VERIFIED.json').read_text())['all_completed']:
        for record in json.loads((recent/'plan.json').read_text())['checkpoints']:
            key=record['registry_key'];cp=Path(record['checkpoint']);spec=all_models[key]
            assert spec.adapters==(str(cp),) and spec.base==record['base']
            assert Path(json.loads((cp.parent/'best.json').read_text())['path'])==cp
            for p in (0,1):
                rows=[json.loads(s) for s in (recent/'results'/record['key']/f'p{p}.jsonl').open()]
                assert len(rows)==36 and all(len(r['samples'])==4 and tuple(r['adapters'])==spec.adapters for r in rows)
            selected[key]=replace(spec,label=f'★ Qwen3 直接 PT · {record["label"]} · step {record["step"]} · dev-best · 已完成评测')
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
    latest_file = Path(__file__).with_name('latest_models.json')
    if latest_file.exists():
        for record in json.loads(latest_file.read_text()):
            ev = Path(record['evaluation_root']); cp = Path(record['checkpoint'])
            assert (ev/'FOX_PATCH_VERIFIED.json').exists() and (ev/'no_eos/COMPLETED.json').exists()
            assert Path(json.loads((cp.parent/'best.json').read_text())['path']) == cp
            spec = all_models[record['registry_key']]
            assert spec.adapters == tuple(record['adapters']) and spec.base == record['base']
            results = [json.loads(line) for line in (ev/'results'/record['key']/'p1.jsonl').open()]
            assert len(results) == 36 and all(len(r['samples']) == 4 and tuple(r['adapters']) == spec.adapters for r in results)
            selected[record['registry_key']] = spec
    display_order = [{'key': '0.6B-PT', 'label': 'Bricknet (0.6B) · 已完成评测'}, {'key': 'post-PT-checkpoint-000750', 'label': 'Bricknet + ourdata + M_only · 750 · 已完成评测'}, {'key': 'fullg-PT-checkpoint-000500', 'label': 'Bricknet + ourdata + Full_G · 500 · 已完成评测'}, {'key': 'nzd-masked-PT-2000', 'label': 'Bricknet + Bricknet_traindata_aug + Full_G · 2000 · 已完成评测'}, {'key': 'nzd-plain-PT-1750', 'label': 'Bricknet + Bricknet_traindata + Full_G · 1750 · 已完成评测'}, {'key': 'direct-M_only-checkpoint-000750', 'label': 'Qwen + ourdata + M_only · 750 · 已完成评测'}, {'key': 'direct-Full_G-checkpoint-000750', 'label': 'Qwen + ourdata + Full_G · 750 · 已完成评测'}, {'key': 'latest-nzd_v2_masked-checkpoint-002211', 'label': 'Qwen + Bricknet_traindata_aug + Full_G · 2211 · 已完成评测'}, {'key': 'latest-nzd_v2_plain-checkpoint-002211', 'label': 'Qwen + Bricknet_traindata + Full_G · 2211 · 已完成评测'}, {'key': 'queue-regularized_g-checkpoint-001500', 'label': 'Qwen + ourdata + Full_G + AdamW 0.05 · 1500 · 已完成评测'}, {'key': 'latest-regularized_g_wd010_do015-checkpoint-001000', 'label': 'Qwen + ourdata + Full_G + AdamW 0.10 · 1000 · dropout 0.15 · 已完成评测'}, {'key': 'grpo-PT-checkpoint-300', 'label': 'Bricknet + Bricknet_traindata_aug + Full_G + GRPO · 300 · 已完成评测'}, {'key': 'direct-BrickNet-checkpoint-002188', 'label': 'Qwen + 官方 paths 抽样70k + Full_G · 2188 · 已完成评测'}, {'key': 'queue-nzd_plain-checkpoint-002188', 'label': 'Qwen + nzd v4子集 plain + Full_G · 2188 · 已完成评测'}, {'key': 'queue-nzd_masked-checkpoint-002188', 'label': 'Qwen + nzd v4子集 masked + Full_G · 2188 · 已完成评测'}]
    selected = {r["key"]: replace(selected[r["key"]], label=r["label"]) for r in display_order if r["key"] in selected}
    if show_all:selected.update({k:v for k,v in all_models.items() if k not in selected and v.mode != 'BASE'})
    return selected

def evaluation_summary():
    try:return Path(__file__).with_name('results_summary.md').read_text(encoding='utf-8')
    except OSError:return '评测总表暂不可用。'
