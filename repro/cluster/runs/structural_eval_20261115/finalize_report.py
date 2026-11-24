import json
from pathlib import Path
root=Path(__file__).resolve().parent
metrics=json.loads((root/'metrics.json').read_text())
summary=(root/'results_summary.md').read_text()
summary='\n'.join(s for s in summary.splitlines() if not s.startswith('碰撞网格覆盖不足的行'))
missing=[f'{m["model"]["label"]} {m["physical"]["collision_coverage"]}/64' for m in metrics.values() if m['physical']['collision_coverage']<64]
if missing:summary+='\n\n无碰撞长度按完成碰撞检查的样本平均。'+ '、'.join(missing)+'；其余模型均为 64/64。未检查样本缺少对应碰撞网格。\n'
summary+='\n[已完成的训练 loss 曲线](http://127.0.0.1:18765/training_structural/)\n'
(root/'results_summary.md').write_text(summary)
report=['# 结构修改后训练的新评测结果','',summary,'','## 权重与复现','',
        '| 模型 | 开发集最佳 step | 开发集 loss |','|---|---:|---:|']
for arm in ('post_pt','post_sft','fullg_pt','fullg_sft'):
    m=metrics[arm]['model'];report.append(f'| {m["label"]} | {m["step"]} | {m["dev_best"]["dev_mean"]:.8f} |')
report+=['','不同监督目标的开发 loss 不能直接比较。四组均按各自开发集选择权重，未根据测试结果更换 checkpoint。','',
 '统一补全共 2,304 次自由生成，另有 512 次从零生成。所有八个模型共享新两条件批次和种子。采样温度 1.0、top-k 20、top-p 0.95；补全预算沿用旧绝对 token 上限，独立生成预算为 4,096 tokens，评分截到 100 块。','',
 '网站另保留四条固定描述的展示样本，以及三个 PT 的给首块回退补全。展示和回退结果不混入主表；回退增加了重试预算。','',
 'protocol.json 保存每个模型的 adapter 链、权重 SHA256 和批次计划。p0/p1 的 *.scored.jsonl 保留逐样本输出、位姿评分、停止行为和碰撞检查。*.free100.scored.jsonl 保存独立搭建的逐样本长度。SCORER_VALIDATED.json 记录与当前已发布评分实现的交叉核验。','',
 '这 18 个来源是此前已探索的固定案例，不能据此宣称未见结构上的普遍泛化能力。主表 pose F1 允许可解析前缀得分，并按目标长度截断，所以超长或未完整解析的输出也可能获得较高 F1。3D viewer 保留完整可解析输出及停止状态。','',
 '记录目录为 /home/sunshq114514/brickICL/runs/structural_eval_20261115/。']
(root/'REPORT.zh-CN.md').write_text('\n'.join(report)+'\n')
