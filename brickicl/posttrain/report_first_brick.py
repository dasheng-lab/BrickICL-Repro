"""Audit and summarize the unified 18-case first-brick experiment."""
import json,hashlib,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from brickicl.eval.report import cluster_ci
from brickicl.posttrain.paths import model_label
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/history_first_brick06'))
    args = parser.parse_args(argv)
    ROOT=args.run_dir;OUT=ROOT/'reports';OUT.mkdir(parents=True,exist_ok=True)

    LABELS={'frozen_pt':'官方 PT','post_pt':'M-only PT · 1750','fullg_pt':'Full-G PT · 500',
            'frozen_sft':'官方 SFT','post_sft':'M-only SFT · 2188','fullg_sft':'Full-G SFT · 250'}
    METRICS=['node_f1','pose_f1','strict_pose_f1','exact','parse_ok','length_abs_error','eos_emitted','hit_token_cap',
             'valid_collision_free','generation_collision_free','pose_pref','leg_is_long']
    def read(p):return [json.loads(s) for s in p.open(encoding='utf-8')]
    def avg(v):return float(np.mean(v))
    def ci(vals,groups):
        vals=np.asarray(vals,float);groups=np.asarray(groups);keys=np.unique(groups)
        sums=np.array([vals[groups==k].sum() for k in keys]);counts=np.array([(groups==k).sum() for k in keys])
        pick=np.random.default_rng(0).integers(len(keys),size=(2000,len(keys)))
        estimates=sums[pick].sum(axis=1)/counts[pick].sum(axis=1)
        return [float(x) for x in np.percentile(estimates,[2.5,97.5])]
    assert np.allclose(ci([-2,0,1,7],['a','b','b','c']),cluster_ci([-2,0,1,7],['a','b','b','c']),atol=1e-12)
    protocol=json.loads((ROOT/'protocol.json').read_text());plan=json.loads((ROOT/'generation_plan.json').read_text())
    LABELS={a:model_label(a,json.loads((ROOT/f'provenance.{a}.json').read_text())['adapters']) for a in LABELS}
    plan_by_id={key:g for g in plan for key in g['ids']}
    inputs={p:{r['id']:r for r in read(ROOT/f'p{p}_triads.jsonl')} for p in (0,1)}
    assert set(inputs[0])==set(inputs[1]) and len(inputs[0])==54
    for key,a in inputs[0].items():
        b=inputs[1][key];assert b['n_prefix']==a['n_prefix']+1
        assert b['prefix'].startswith(a['prefix']) and len(b['prefix'].splitlines())-len(a['prefix'].splitlines())==2
        for k in ('short','long'):assert a['prefix']+a['target_'+k]==b['prefix']+b['target_'+k]
    report={'protocol':protocol,'settings':{},'first_brick_effect':{},'audit':{},'source_results':{}}
    raw={};record_total=0
    for prime in (0,1):
        models={};raw[prime]={}
        for arm in LABELS:
            records=read(ROOT/f'p{prime}.{arm}.jsonl')
            assert len(records)==54==len({r['id'] for r in records})
            assert {r['id'] for r in records}==set(inputs[prime])
            by_source=defaultdict(dict)
            for r in records:
                inp=inputs[prime][r['id']];g=plan_by_id[r['id']]
                assert r['prime']==prime and r['r']==4 and r['n_prefix']==inp['n_prefix']
                assert r['seed']==g['seed'] and r['max_new_tokens']==g['max_new_tokens'] and len(r['samples'])==4
                for sample in r['samples']:
                    indices=None if sample['collisions'] is None else [i for i in sample['collisions'] if i>=inp['n_prefix']]
                    expected_free=sample['parse_ok'] and indices==[]
                    if 'generation_collision_free' in sample:assert sample['generation_collision_free']==expected_free
                    sample['generation_collision_free']=expected_free
                assert abs(r['log_odds']-(r['lp_long']['logp']-r['lp_short']['logp']))<1e-8
                vals={m:avg([s[m] for s in r['samples']]) for m in METRICS}
                vals['log_odds']=r['log_odds'];vals['signed_length_bias']=avg([s['n_generated']-inp['k_'+inp['expected']] for s in r['samples']])
                by_source[r['source_key']][r['condition']]=vals;record_total+=len(r['samples'])
            assert len(by_source)==18 and all(set(v)=={'original','stretched','ablated'} for v in by_source.values())
            metrics=METRICS+['log_odds','signed_length_bias'];keys=sorted(by_source)
            result={'sources':18,'inputs':54,'generations':216,'conditions':{},'overall':{},'delta':{}}
            for c in ['original','stretched','ablated']:
                result['conditions'][c]={m:avg([v[c][m] for v in by_source.values()]) for m in metrics}
            result['overall']={m:avg([v[c][m] for v in by_source.values() for c in ['original','stretched','ablated']]) for m in metrics}
            for m in metrics:
                vals=[by_source[k]['stretched'][m]-by_source[k]['original'][m] for k in keys]
                result['delta'][m]={'mean':avg(vals),'ci95':ci(vals,keys)}
            models[arm]=result;raw[prime][arm]=by_source
        report['settings'][str(prime)]={'label':'给第一块' if prime else '不给第一块','models':models}
        report['source_results'][str(prime)]=raw[prime]
    assert record_total==2592
    for arm in LABELS:
        effects={};keys=sorted(raw[0][arm])
        for c in ['original','stretched','ablated']:
            effects[c]={}
            for m in ['pose_f1','strict_pose_f1','exact','parse_ok','length_abs_error']:
                vals=[raw[1][arm][k][c][m]-raw[0][arm][k][c][m] for k in keys]
                effects[c][m]={'given_minus_not_given':avg(vals),'ci95':ci(vals,keys)}
        report['first_brick_effect'][arm]=effects
    report['audit']={'total_generations':record_total,'identical_sources':True,'byte_identical_complete_references':True,
        'same_seeds_batch_composition_and_caps':True,'same_six_checkpoints':True,
        'max_between_prime_log_odds_difference':max(abs(raw[0][a][k][c]['log_odds']-raw[1][a][k][c]['log_odds']) for a in LABELS for k in raw[0][a] for c in ['original','stretched','ablated'])}
    (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    intro=['### 统一测试：18个固定案例，六个模型','',
     '两部分使用完全相同的狐狸、出租车等18个来源，固定加长幅度 r=4；只改变是否预给缺失子装配的第一块。两种设置都已经提供主体及可见参照，均属于局部补全。',
     '每个来源测试正常、加长、删除参照三条件，每条件4次；每个模型每部分216次生成。模型间、两种设置间共用批次、随机种子与token上限；不使用额外caption或完整模型示范。','']
    body=[]
    for prime in (0,1):
        body += [f'#### {"给第一块" if prime else "不给第一块"}','',
          '已提供目标首块，只评模型继续生成的剩余部分。' if prime else '目标首块也由模型生成，需要自行接入正确位置。','',
          '| 模型 · checkpoint | Δlog-odds | 正常 pose F1 | 加长 pose F1 | 删除参照 pose F1 | 补全完整解析率（3条件平均） |',
          '|---|---:|---:|---:|---:|---:|']
        for arm,label in LABELS.items():
            s=report['settings'][str(prime)]['models'][arm];c=s['conditions']
            body.append(f'| {label} | {s["delta"]["log_odds"]["mean"]:+.2f} | {c["original"]["pose_f1"]:.4f} | {c["stretched"]["pose_f1"]:.4f} | {c["ablated"]["pose_f1"]:.4f} | {s["overall"]["parse_ok"]:.1%} |')
        body.append('')
    notes=['**读表：** 在同一部分比较模型，在上下两部分比较首块提示的效果。Δlog-odds衡量参照加长后对长答案的偏好变化；pose F1衡量实际生成的零件、颜色和位姿匹配；解析率只检查补全序列能否完整解析，不等于无碰撞或语义正确。','',
     '**为什么两部分的Δlog-odds可能几乎一样？** 首块是长、短参考答案共有的前缀。把它从答案移到输入后，这段共同概率在长减短时抵消。因此首块提示的效果主要看实际生成F1、解析率和停止行为，不能只看Δlog-odds。','',
     '“删除参照”移除其他重复副本，保留主体；短答案是诊断参考，并非残余输入唯一能推出的答案。全部案例是之前已经探索过的固定案例，不当作全新的泛化测试。',
     '权重仍由开发集选择；表中步骤从本轮provenance读取。本表不混入旧的独立从零生成率；旧实验保存在归档中。','',
     '[打开统一六模型3D对照](http://127.0.0.1:18765/unified_compare.html?source=114&prime=0)']
    summary='\n'.join(intro+body+notes)+'\n'
    (OUT/'GRADIO_SUMMARY.zh-CN.md').write_text(summary,encoding='utf-8')
    detail=['# 统一首块提示评测','']+intro+body+notes+['','## 核验与完整指标','',
     '共2592次生成记录，其中1296次“给第一块”结果通过输入、权重、种子、token上限核对后复用；1296次“不给第一块”重新生成。',
     '“给第一块”的已知部分不计入待生成目标F1。删除参照条件的短目标只作诊断；不把给定首块自动算作生成正确。',
     '同一来源的两种设置使用相同绝对token上限（沿用旧“给第一块”的上限），而不是随剩余目标件数增减预算。',
     'metrics.json保留分条件node/pose/strict pose F1、exact、解析率、EOS、碰撞率、长度误差、逐来源结果及首块提示的配对变化。95% CI按18个来源bootstrap，2000次，seed=0。',
     '运行目录由 --run-dir 指定（默认 runs/history_first_brick06）。未重新训练或按测试分数选择checkpoint。','',
     '## 给第一块对加长补全的影响','', '| 模型 | pose F1变化（给−不给） | 95% CI |','|---|---:|---|']
    for arm,label in LABELS.items():
        e=report['first_brick_effect'][arm]['stretched']['pose_f1'];detail.append(f'| {label} | {e["given_minus_not_given"]:+.4f} | [{e["ci95"][0]:+.4f}, {e["ci95"][1]:+.4f}] |')
    (OUT/'REPORT.zh-CN.md').write_text('\n'.join(detail)+'\n',encoding='utf-8')
    print(json.dumps({'audit':report['audit'],'tables':{p:{a:{'delta_log_odds':v['delta']['log_odds']['mean'],'stretched_pose_f1':v['conditions']['stretched']['pose_f1'],'parse_rate':v['overall']['parse_ok']} for a,v in s['models'].items()} for p,s in report['settings'].items()}},indent=2))

if __name__ == '__main__':
    main()
