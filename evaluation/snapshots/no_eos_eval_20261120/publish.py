import csv,fcntl,json,os,shutil,signal,subprocess,time,urllib.request
from pathlib import Path
from statistics import mean
from common import ROOT,REPO,read,sha,atomic

SERVICE=Path('/home/sunshq114514/brickicl-gradio-service');TARGET=SERVICE/'brickicl/playground/results_summary.md';URL='http://127.0.0.1:17860'
def get(route):
    with urllib.request.urlopen(URL+route,timeout=10) as f:return json.load(f)
def process():
    ids=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if p.stat().st_uid==os.getuid() and b'brickicl.playground.app' in args and b'--port' in args and args[args.index(b'--port')+1]==b'17860':ids.append(int(p.name))
        except (OSError,ValueError,IndexError):pass
    assert len(ids)==1,ids;return ids[0]
def start():
    with (SERVICE/'logs/playground.no-eos.log').open('ab') as log:
        return subprocess.Popen(['bash','brickicl/playground/cluster_service.sh'],cwd=SERVICE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
def stop(pid):
    os.kill(pid,signal.SIGTERM)
    for _ in range(60):
        if not Path(f'/proc/{pid}').exists():return
        time.sleep(.5)
    raise RuntimeError('Gradio did not stop')
def replace(src,dest):
    tmp=dest.with_name(dest.name+'.no-eos-tmp');shutil.copy2(src,tmp);tmp.replace(dest)
def main():
    assert (ROOT/'SCORING_COMPLETED.json').exists()
    plan=json.loads((ROOT/'plan.json').read_text());out=ROOT/'delivery';out.mkdir(exist_ok=True)
    rows=[];metrics={}
    for spec in plan['models']:
        dest=ROOT/'results'/spec['key'];m=json.loads((dest/'metrics.json').read_text());assert m['eos_suppression_verified']
        raw=read(dest/'free_no_eos.jsonl');scored=read(dest/'free_no_eos.scored.jsonl')
        assert len(raw)==len(scored)==64
        for a,b in zip(raw,scored):
            assert a['id']==b['id'] and a['text']==b['full_text']
            assert len(a['generated_token_ids'])==4096 and not set(a['forbidden_eos_ids'])&set(a['generated_token_ids'])
        known=[r['collision_free_length'] for r in scored if r['collision_free_length'] is not None]
        assert mean(r['valid_length'] for r in scored)==m['no_eos_valid_length']
        assert (mean(known) if known else None)==m['no_eos_collision_free_length'] and len(known)==m['collision_coverage']
        row=dict(spec['published_row']);row.update(no_eos_valid_length=m['no_eos_valid_length'],
            no_eos_collision_free_length=m['no_eos_collision_free_length'],no_eos_collision_coverage=m['collision_coverage'])
        rows.append(row);metrics[spec['key']]=m
    atomic(ROOT/'VERIFIED.json',{'models':len(rows),'samples':len(rows)*64,'all_eos_suppressed':True,'all_completed':True})
    lines=['## 给第一块 · PT 统一评测','',
        '按“官方 → BrickNet 起点后训练 → Qwen3 起点后训练 → RL”排列。本阶段起点与开发集最佳 step 保留。最后四列是独立生成，分别允许／禁止 EOS；其余补全指标仍使用原协议。','']
    groups=list(dict.fromkeys(r['group'] for r in rows))
    for i,group in enumerate(groups,1):
        lines += [f'### {i}. {group}','','| 模型 | 本阶段训练起点 | step | 采样 Δlog-odds | 正常 F1 | 加长 F1 | 有效长度 | 无碰撞长度 | 禁 EOS 有效长度 | 禁 EOS 无碰撞长度 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['group']!=group:continue
            f=r['no_eos_collision_free_length'];free='未知' if f is None else f'{f:.2f}'
            lines.append(f'| {r["model"]} | {r["initialization"]} | {r["step"] if r["step"] is not None else "发布权重"} | {r["delta_log_odds"]:+.3f} | {r["f1_normal"]:.4f} | {r["f1_stretched"]:.4f} | {r["valid_length"]:.2f} | {r["collision_free_length"]:.2f} | {r["no_eos_valid_length"]:.2f} | {free} |')
        lines.append('')
    lines+=['**独立生成口径**：两种条件各64次，只输入a，T=1、top-k20、top-p=.95、batch32、随机种子0/32。长度列为64次生成的平均值（经验期望），均按前100块评分。“禁EOS”将全部EOS token的采样概率置零，不使用EOS停止，仍有4096 token硬上限；不是删除已生成的EOS或无限生成。','',
        '有效长度截止首次解析失败，无碰撞长度另截止首次碰撞。新增两列用于检查提前结束的影响，不代表原模型自然生成能力自动变好；若首行就不合法，强制继续也不会挽回前缀得分。','',
        '**补全口径**：给第一块，18个历史validation来源，正常/加长各4次。采样Δlog-odds使用完整输出长特征命中、+0.5平滑；pose F1按剩余目标长度截断、不计颜色。无caption、无整件示范、无回退。','',
        'nzd v2 增强2000从官方BrickNet PT出发，并作为GRPO300的起点。当前v4子集的Qwen3组从原始Qwen3训练；v2重训完成后另评估并明确版本。','',
        '[GRPO渲染对照](http://127.0.0.1:18765/grpo_compare.html?source=114&prime=1) · [正则化／nzd渲染对照](http://127.0.0.1:18765/qwen3_regularized_compare.html?source=114&prime=1)']
    missing={k:m['collision_coverage'] for k,m in metrics.items() if m['collision_coverage']<64}
    if missing:lines+=['','禁EOS碰撞网格覆盖不足：'+str(missing)+'；无碰撞长度只对可检查样本求平均。']
    summary='\n'.join(lines)+'\n';(out/'PT_RESULTS_NO_EOS.zh-CN.md').write_text(summary);atomic(out/'metrics.json',metrics)
    with (out/'PT_RESULTS_NO_EOS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if (ROOT/'PUBLISHED.json').exists():return
    assert get('/health')['service_id']=='sunshq-gradio-live'
    config=get('/config');expected=next(c['props'] for c in config['components'] if c['props'].get('label')=='01 / 选择模型')['choices']
    assert len(expected)==12 and [r[1] for r in expected]==[r['registry_key'] for r in rows]
    assert get('/gradio_api/queue/status')['queue_size']==0
    shutil.copy2(TARGET,ROOT/'prior_results_summary.md');pid=process();child=None
    try:
        replace(out/'PT_RESULTS_NO_EOS.zh-CN.md',TARGET);stop(pid);child=start()
        for _ in range(60):
            try:
                config=get('/config');markdown=[r['props'].get('value','').strip() for r in config['components'] if r['type']=='markdown']
                assert summary.strip() in markdown
                dropdown=next(c['props'] for c in config['components'] if c['props'].get('label')=='01 / 选择模型')
                assert dropdown['choices']==expected;break
            except (OSError,AssertionError,ValueError,KeyError,StopIteration):time.sleep(1)
        else:raise RuntimeError('No-EOS columns not confirmed live')
        atomic(ROOT/'PUBLISHED.json',{'models':12,'samples':768,'columns':['禁 EOS 有效长度','禁 EOS 无碰撞长度'],'summary_sha256':sha(TARGET),'unix_time':time.time()})
    except BaseException:
        replace(ROOT/'prior_results_summary.md',TARGET)
        if child and child.poll() is None:child.terminate();child.wait(timeout=30)
        if not Path(f'/proc/{pid}').exists():start()
        raise
    print('Published both no-EOS columns',flush=True)
if __name__=='__main__':main()
