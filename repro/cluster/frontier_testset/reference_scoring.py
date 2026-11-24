import argparse,json,math,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
import numpy as np
from scipy.optimize import linear_sum_assignment
from bricknet import parse_sample,tree_to_graph,decode_graph,score_text,serialize_tree
from brickicl.eval.match import same_pose

ROOT=Path('runs/structural_eval_20261115')
ORDER=('frozen_pt','post_pt','fullg_pt','plain','masked','frozen_sft','post_sft','fullg_sft')
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def atomic(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(p)
def realize(s):
    p=parse_sample(s)
    return p.tree,np.asarray(decode_graph(tree_to_graph(p.tree))),p.error
def stack_count(tree,root,pid):
    if len(tree.parts)<=root:return 0
    branch={root}
    for i in range(root+1,len(tree.parts)):
        if tree.edges[i-1].parent in branch:branch.add(i)
    return sum(tree.parts[i].part_id==pid for i in branch)
def pose(g,gt,reference,ref_t):
    match=np.zeros((len(g),len(reference)),dtype=np.int8)
    for i,a in enumerate(g):
        for j,b in enumerate(reference):
            if a.part_id==b.part_id and same_pose(a.part_id,gt[i],ref_t[j]):match[i,j]=1
    hits=int(match[linear_sum_assignment(-match)].sum()) if match.size else 0
    return 2*hits/(len(g)+len(reference))
def score_row(item):
    row,rec=item
    short,_,se=realize(row['prefix']+row['target_short'])
    long,_,le=realize(row['prefix']+row['target_long'])
    assert se is None and le is None
    extra=Counter(p.part_id for p in long.parts)-Counter(p.part_id for p in short.parts)
    assert len(extra)==1 and sum(extra.values())==4
    pid=next(iter(extra));origin=row['n_prefix']-row['prime'];threshold=stack_count(long,origin,pid)
    ref,ref_t,err=realize(row['prefix']+row['target_'+row['expected']]);assert err is None
    cut=row['n_prefix'];R=ref.parts[cut:];rt=ref_t[cut:]
    result=dict(rec);samples=[]
    for s in rec['samples']:
        text=row['prefix']+s['text']
        tree,mats,error=realize(text)
        G=tree.parts[cut:];gt=mats[cut:]
        pf=pose(G[:len(R)],gt[:len(R)],R,rt)
        full_f1=pose(G,gt,R,rt)
        want=Counter(p.part_id for p in R);got=Counter(p.part_id for p in G)
        node_f1=2*sum((want&got).values())/(len(G)+len(R))
        # score_text appends a terminal newline; first-brick scoring and the viewer parse raw text.
        # Check collisions on exactly the parseable tree that F1 and the viewer use.
        _,_,collisions=score_text(serialize_tree(tree))
        added=None if collisions is None else [i for i in collisions if i>=cut]
        samples.append(s|{'pose_f1':pf,'pose_f1_untruncated':full_f1,'node_f1':node_f1,
            'n_generated':len(G),'target_parts':len(R),'parse_ok':error is None,'parse_error':error,
            'exact':pf==1, 'exact_old':error is None and len(G)==len(R) and pf==1,
            'long_feature':stack_count(tree,origin,pid)>=threshold,
            'long_feature_count':stack_count(tree,origin,pid),'long_feature_threshold':threshold,
            'collisions':collisions,'generated_collision_indices':added,
            'generation_collision_free':error is None and added==[]})
    result['samples']=samples
    return result
def physical_row(rec):
    lines=rec['text'].rstrip('\n').split('\n');capped=len(lines)>199
    text='\n'.join(lines[:199])
    n,invalid,collisions=score_text(text)
    valid=n if invalid is None else invalid
    free=min([valid]+collisions) if collisions is not None else None
    parsed=parse_sample(text)
    return rec|{'full_text':rec['text'],'text':text,'capped':capped,'n_actions':n,'invalid':invalid,'collisions':collisions,
                'valid_length':valid,'collision_free_length':free,'parsed_parts':len(parsed.tree.parts),'parse_error':parsed.error}
def summarize_completion(records):
    sources=defaultdict(dict);f1=defaultdict(list);parse=0;all_n=0
    for r in records:
        samples=r['samples'];c=r['condition']
        sources[r['source_key']][c]=sum(s['long_feature'] for s in samples)
        f1[c].extend(s['pose_f1'] for s in samples)
        parse+=sum(s['parse_ok'] for s in samples);all_n+=len(samples)
    assert len(sources)==100 and all(set(s)=={'original','stretched'} for s in sources.values())
    logit=lambda k:math.log((k+.5)/(4-k+.5))
    vals=[logit(s['stretched'])-logit(s['original']) for s in sources.values()]
    estimates=np.asarray(vals)[np.random.default_rng(0).integers(len(sources),size=(2000,len(sources)))].mean(1)
    return {'delta':mean(vals),'delta_ci95':np.percentile(estimates,[2.5,97.5]).tolist(),
            'f1':{c:mean(v) for c,v in f1.items()},'parse':parse/all_n,'samples':all_n,'source_hits':dict(sources)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    inputs={p:{r['id']:r for r in read(ROOT/f'p{p}_inputs.jsonl')} for p in (0,1)}
    models=protocol['models'];metrics={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for arm in ('post_pt','post_sft','fullg_pt','fullg_sft','frozen_pt','frozen_sft','plain','masked'):
            marker=ROOT/f'GENERATED.{arm}.json'
            while not marker.exists():
                if not args.watch:raise RuntimeError(f'Generation pending for {arm}')
                time.sleep(20)
            result={'model':models[arm],'completion':{}}
            for prime in (0,1):
                recs=read(ROOT/f'p{prime}.{arm}.jsonl')
                assert len(recs)==36 and {r['id'] for r in recs}==set(inputs[prime])
                scored=list(pool.map(score_row,[(inputs[prime][r['id']],r) for r in recs],chunksize=1))
                (ROOT/f'p{prime}.{arm}.scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
                result['completion'][str(prime)]=summarize_completion(scored)
            rs=read(ROOT/f'{arm}.free.jsonl');assert len(rs)==64
            phys=list(pool.map(physical_row,rs,chunksize=4))
            (ROOT/f'{arm}.free100.scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in phys))
            known=[r['collision_free_length'] for r in phys if r['collision_free_length'] is not None]
            result['physical']={'samples':64,'valid_length':mean(r['valid_length'] for r in phys),
                'collision_free_length':mean(known) if known else None,'collision_coverage':len(known),
                'parse_rate':mean(r['invalid'] is None for r in phys)}
            metrics[arm]=result;atomic(ROOT/f'metrics.{arm}.json',result)
            print('SCORED',arm,result['completion']['1']['f1'],result['physical'],flush=True)
    atomic(ROOT/'metrics.json',metrics)
    lines=['### 新版统一评测：18 个固定案例，8 个模型','',
           'M-only 和 Full-G 使用结构修改数据后训练的开发集最佳权重。全部八个模型重新推理；正常／加长两种条件，每条件 4 次，每个模型每种首块设置 144 次。无 caption、无完整示范、无回退采样。','']
    for prime in (0,1):
        lines+=['#### '+('给第一块' if prime else '不给第一块'),'',
            '| 模型 · dev-best | 采样 Δlog-odds | 正常 pose F1 | 加长 pose F1 | 有效长度 | 无碰撞长度 |',
            '|---|---:|---:|---:|---:|---:|']
        for arm in ORDER:
            m=metrics[arm];spec=m['model'];label=spec['label']+(f' · {spec["step"]}' if 'step' in spec else '')
            c=m['completion'][str(prime)];p=m['physical']
            free='—' if p['collision_free_length'] is None else f'{p["collision_free_length"]:.1f}'
            lines.append(f'| {label} | {c["delta"]:+.3f} | {c["f1"]["original"]:.4f} | {c["f1"]["stretched"]:.4f} | {p["valid_length"]:.1f} | {free} |')
        lines.append('')
    lines+=['**Pose F1** 按标准答案长度取生成的前 n_t 块，再做型号与位姿的一对一匹配，颜色不参与，位置容差 3 LDU。给首块时只评剩余部分。Viewer 展示完整可解析输出，超长部分仍可见。','',
        '**采样 Δlog-odds** 用完整输出统计长特征命中数 k，以 ln[(k+0.5)/(4-k+0.5)] 计算每来源的加长减正常，再平均 18 个来源。每条件仅 4 次采样。','',
        '**有效长度／无碰撞长度** 来自每模型另外 64 次从零生成，输出截至 100 块后评分。PT 只给起始符 a，SFT 使用前 64 条官方验证描述，每条一次。表中两种首块设置共用这两列。','',
        '旧表含删除条件的批次已重新组织，本表所有八模型共用新的两条件批次、种子和 token 上限。所有权重按开发集选取，测试集用于报告。','',
        '[打开新版六模型 3D 对照](http://127.0.0.1:18765/unified_compare.html?source=114&prime=1) · [对照／增强 PT](http://127.0.0.1:18765/nzd_order_compare.html?source=258&prime=1) · [PT 回退采样](http://127.0.0.1:18765/rollback_compare.html)']
    missing={a:m['physical']['collision_coverage'] for a,m in metrics.items() if m['physical']['collision_coverage']<64}
    if missing:lines+=['','碰撞网格覆盖不足的行，无碰撞长度仅对已检查样本取平均。覆盖样本数 '+str(missing)+'。']
    (ROOT/'results_summary.md').write_text('\n'.join(lines)+'\n')
    atomic(ROOT/'SCORING_COMPLETED.json',{'state':'completed','models':8,'completion_samples':2304,'physical_samples':512})
if __name__=='__main__':main()
