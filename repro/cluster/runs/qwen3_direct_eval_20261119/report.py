import csv,json,shutil
from common import *

def main():
    assert json.loads((ROOT/'VERIFIED.json').read_text())['all_completed']
    specs=plan()['checkpoints'];metrics={s['arm']:json.loads((folder(s)/'metrics.json').read_text()) for s in specs}
    old=json.loads((PREVIOUS/'metrics.json').read_text());combined=old|metrics
    ordering=['direct_bricknet','direct_m','direct_g','frozen_pt','post_pt','fullg_pt','plain','masked','frozen_sft','post_sft','fullg_sft']
    lines=['### Qwen3 直接 PT 与既有模型：统一生成评测','',
       '新增三组从原始 Qwen3-0.6B 直接训练，各 7 万条，LoRA r=64，不叠加 BrickNet adapter。所有训练权重按开发集选取，表中 step 为 dev-best。旧模型结果保留，采样协议相同。','',
       '“BrickNet 数据”指本次从官方训练路径抽样的 7 万条子集；“官方 BrickNet PT”是已发布的完整预训练权重，两者训练预算不同。','',
       '固定 18 个历史案例（来自官方 validation），仅正常／加长，每条件 4 次；每个模型每种首块设置 144 次补全，不提供 caption 或完整作品示范，不使用回退。','']
    rows=[]
    for prime in ('1','0'):
        lines+=['#### '+('给第一块' if prime=='1' else '不给第一块'),'',
          '| 模型 · step | 采样 Δlog-odds | 正常 pose F1 | 加长 pose F1 | 独立有效长度 | 独立无碰撞长度 |',
          '|---|---:|---:|---:|---:|---:|']
        for arm in ordering:
            m=combined[arm];s=m['model'];c=m['completion'][prime];p=m['physical'];label=s['label']
            if 'step' in s:label+=f' · {s["step"]}'
            if arm in ('post_pt','fullg_pt','post_sft','fullg_sft'):label='BrickNet 初始化 · '+label
            lines.append(f'| {label} | {c["delta"]:+.3f} | {c["f1"]["original"]:.4f} | {c["f1"]["stretched"]:.4f} | {p["valid_length"]:.2f} | {p["collision_free_length"]:.2f} |')
            rows.append({'arm':arm,'model':label,'prime':int(prime),'step':s.get('step',0),'delta_log_odds':c['delta'],
              'f1_normal':c['f1']['original'],'f1_stretched':c['f1']['stretched'],'expected_valid_length':p['valid_length'],
              'expected_collision_free_length':p['collision_free_length'],'collision_coverage':p['collision_coverage']})
        lines.append('')
    lines+=['Pose F1：取生成的前目标长度块，按类型／位姿做一对一匹配，忽略颜色；给首块时只评剩余部分。', '',
       '采样 Δlog-odds：每来源分别将正常／加长 4 次采样的长特征命中 k 变为 ln[(k+0.5)/(4-k+0.5)]，加长减正常，再平均 18 个来源。','',
       '独立生成：每模型另外 64 次，最多生成 4096 token，按前 100 块评估。PT 只给起始符 a；旧 SFT 使用官方描述。有效长度截止首次解析失败，无碰撞长度再截止首次碰撞。两种首块设置共用这两列。', '',
       '官方 SFT 的碰撞网格覆盖为 63/64，无碰撞长度只对可检查样本取平均；其它覆盖见 CSV。','',
       '[打开 Qwen3 直接 PT 六模型 3D 对照](http://127.0.0.1:18765/qwen3_direct_compare.html?source=114&prime=1) · [此前六模型对照](http://127.0.0.1:18765/unified_compare.html?source=114&prime=1)']
    if metrics['direct_m']['physical']['valid_length']==0:
        lines+=['','本次 M-only 的 64 次独立生成均在第一块前解析失败，因此两种长度期望都为 0；在给定 LEGO 上下文时仍可获得非零补全 F1。Viewer 保留这些失败输出。']
    summary='\n'.join(lines)+'\n';(ROOT/'results_summary.md').write_text(summary)
    dest=ROOT/'delivery';dest.mkdir(exist_ok=True);(dest/'RESULTS.zh-CN.md').write_text(summary)
    with (dest/'evaluation_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic(ROOT/'metrics.json',metrics);atomic(dest/'metrics.json',metrics)
    print('Report: three new + eight existing models',flush=True)
if __name__=='__main__':main()
