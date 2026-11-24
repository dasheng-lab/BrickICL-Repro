"""Stage fresh result boards and self-contained MPDs without altering the live catalog."""
import copy,hashlib,importlib.util,json,shutil,time
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr,score_text
from brickicl.viz.lazy_catalog import _write_json,write_catalog
from run_inference import ROOT,read,atomic

LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
STAGE=ROOT/'viewer_stage'
PREFIX='structural_eval_20261115'
SIX=('frozen_pt','post_pt','fullg_pt','frozen_sft','post_sft','fullg_sft')
NEW=('post_pt','post_sft','fullg_pt','fullg_sft')
def main():
    for marker in ('SCORING_COMPLETED.json','EXTRA_COMPLETED.json','SCORER_VALIDATED.json'):
        assert (ROOT/marker).exists(),marker
    protocol=json.loads((ROOT/'protocol.json').read_text());models=protocol['models']
    labels={a:r['label']+(f' · {r["step"]}' if 'step' in r else '') for a,r in models.items()}
    old=json.loads((LIVE/'unified_compare.json').read_text())
    (STAGE/'models').mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('current_viewer_pack',LIVE/'pack.py')
    pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
    lib=pack.build_index();entries=[]
    def add(rel,text,cut,task,group,title,**meta):
        parsed=parse_sample(text);parts=parsed.tree.parts
        lines=graph_to_ldr(tree_to_graph(parsed.tree)).splitlines() if parts else []
        mpd,missing=pack.embed([l for l in lines if l.startswith('1 ')],lib)
        assert not missing,(rel,missing)
        file=PREFIX+'/'+rel;dest=STAGE/'models'/file;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(mpd)
        entry={'file':file,'path':text,'n_parts':len(parts),'parse_error':parsed.error,'heldout_from':cut,
               'task':task,'group':group,'title':title,'size_kb':round(dest.stat().st_size/1024),
               'evaluation_run':ROOT.name}|meta
        entries.append(entry)
        return file
    boards={kind:{'schema':1,'arms':{a:labels[a] for a in arms},'samples':4,'source_count':18,
                       'source_order':old['source_order'],'sources':{},'evaluation_run':ROOT.name}
            for kind,arms in [('unified',SIX),('nzd',('plain','masked'))]}
    inputs={}
    scored={}
    for p in (0,1):
        inputs[p]={r['id']:r for r in read(ROOT/f'p{p}_inputs.jsonl')}
        scored[p]={a:{r['id']:r for r in read(ROOT/f'p{p}.{a}.scored.jsonl')} for a in models}
        task='completion_given_head' if p else 'completion_no_head'
        for sid in old['source_order']:
            src=old['sources'][sid];group=('给第一块' if p else '不给第一块')+' · '+src['label']
            for board in boards.values():
                source=board['sources'].setdefault(sid,{'label':src['label'],'caption':src['caption'],'primes':{}})
                source['primes'][str(p)]={'prime':p,'conditions':{}}
            for condition,cn in [('original','正常'),('stretched','加长')]:
                row=inputs[p][f'val:{sid}:{condition}'];cut=row['n_prefix'];k=row['k_'+row['expected']]
                rel=f'completion/p{p}/{sid}/{condition}'
                inp=add(rel+'_input.mpd',row['prefix'],cut,task,group,cn+' · 共同输入',is_input=True,
                        source=int(sid),prime=p,condition=condition,caption=src['caption'],collisions=row['prefix_collisions'])
                ref=add(rel+'_reference.mpd',row['prefix']+row['target_'+row['expected']],cut,task,group,cn+' · 标准答案',
                        is_reference=True,source=int(sid),prime=p,condition=condition,caption=src['caption'])
                common={'label':cn,'prefix_parts':cut,'target_parts':k,'input_sha256':row['prefix_sha256'],
                        'input':inp,'reference':ref,'outputs':{}}
                for a in models:
                    record=scored[p][a][row['id']]
                    out=[]
                    for i,s in enumerate(record['samples'],1):
                        metrics={x:s[x] for x in ('pose_f1','node_f1','parse_ok','n_generated','exact','eos_emitted')}
                        note=f'{cn} · 样本 {i}/4。位姿 F1 按答案长度截断、不计颜色；显示完整可解析输出。'
                        file=add(rel+f'_{a}_s{i}.mpd',row['prefix']+s['text'],cut,task,group,f'{cn} · {labels[a]} · 样本 {i}',
                            arm=a,source=int(sid),prime=p,condition=condition,sample_index=i-1,
                            caption=src['caption'],note=note,collisions=s['collisions'],evaluation_metrics=metrics,
                            adapters=models[a]['adapters'],checkpoint=models[a].get('checkpoint'))
                        assert entries[-1]['n_parts']==cut+s['n_generated']
                        out.append({'file':file}|metrics)
                    common['outputs'][a]=out
                for kind,board in boards.items():
                    c=copy.deepcopy(common);c['outputs']={a:common['outputs'][a] for a in board['arms']}
                    board['sources'][sid]['primes'][str(p)]['conditions'][condition]=c
            print('PACKED',p,sid,flush=True)
    rollback={'arms':{a:labels[a] for a in ('frozen_pt','post_pt','fullg_pt')},'sources':{},'evaluation_run':ROOT.name}
    for a in rollback['arms']:
        rows=read(ROOT/f'rollback.{a}.jsonl');assert len(rows)==32
        for r in rows:
            row=inputs[1][r['id']];sid=row['source_key'].split(':')[-1];condition=row['condition'];i=r['sample_index']
            s=r['sample'];k=row['k_'+row['expected']];cut=row['n_prefix'];cn='正常' if condition=='original' else '加长'
            base=scored[1][a][r['id']]['samples'][i]
            file=add(f'rollback/{sid}/{condition}_{a}_s{i+1}.mpd',row['prefix']+s['text'],cut,'rollback_pt',
                f'PT 回退 · {old["sources"][sid]["label"]} · {cn}',f'{labels[a]} · 样本 {i+1}',arm=a,source=int(sid),
                prime=1,condition=condition,sample_index=i,collisions=s['collisions'],
                note='逐块检查解析和新增碰撞。位姿 F1 按答案长度截断，不计颜色。',
                evaluation_metrics={x:s[x] for x in ('pose_f1','parse_ok','eos_emitted')},
                rollback={x:s[x] for x in ('termination','rejections','backtracks','attempted_tokens')},
                adapters=models[a]['adapters'],checkpoint=models[a].get('checkpoint'))
            source=rollback['sources'].setdefault(sid,{'label':old['sources'][sid]['label'],'conditions':{}})
            ref=boards['unified']['sources'][sid]['primes']['1']['conditions'][condition]
            case=source['conditions'].setdefault(condition,{'prefix_parts':cut,'target_parts':k,
                'prefix_collisions':row['prefix_collisions'],'reference':ref['reference'],'results':{x:[None]*4 for x in rollback['arms']}})
            fields=('pose_f1','n_generated','parse_ok','collisions','eos_emitted')
            data={x:s[x] for x in fields+('termination','rejections','backtracks','attempted_tokens','retained_tokens')}
            data['events']=[e for e in s['events'] if e['event']!='accept']
            case['results'][a][i]={'file':file,'baseline_file':ref['outputs'][a][i]['file'],'sample':data,
                                   'baseline':{x:base[x] for x in fields}}
    # The named caption gallery remains four examples, two samples per SFT model.
    names={'fox':'狐狸','car':'汽车','chair':'椅子','robot':'机器人'}
    for a in ('frozen_sft','post_sft','fullg_sft'):
        for r in read(ROOT/f'gallery.{a}.jsonl'):
            text=r['text']
            if not text.endswith('\n'):text+='\n'
            parsed=parse_sample(text);i=r['sample']
            add(f'gallery/{r["id"]}_{a}_s{i+1}.mpd',text,0,'from_scratch',f'SFT 从零生成 · {names[r["id"]]}',
                f'{labels[a]} · 样本 {i+1}',arm=a,caption=r['caption'],from_scratch=True,is_primary_generation_test=True,
                input_mode='caption',input_prompt=r['caption']+'\n',generation_task=r['id'],sample_index=i,
                collisions=r['collisions'],adapters=models[a]['adapters'],checkpoint=models[a].get('checkpoint'),
                note=f'固定描述第 {i+1}/2 次，原始采样结果。最多展示前 100 块；与统计表的 64 条描述是不同展示批次。',
                generation_metadata={'parse_ok':parsed.error is None,'eos_emitted':r['eos_emitted'],'capped':r['capped']})
    for a in ('frozen_pt','post_pt','fullg_pt'):
        for r in read(ROOT/f'{a}.free100.scored.jsonl')[:4]:
            text=r['text'];text=text if text.endswith('\n') else text+'\n';i=r['id']
            add(f'gallery/unconditional_{a}_s{i+1}.mpd',text,0,'from_scratch','PT 从零生成 · 前四次采样',
                f'{labels[a]} · 样本 {i+1}',arm=a,from_scratch=True,is_primary_generation_test=True,
                input_mode='unconditional',input_prompt='a',sample_index=i,collisions=r['collisions'],
                adapters=models[a]['adapters'],checkpoint=models[a].get('checkpoint'),
                note='64 次独立搭建中的前四次，最多展示 100 块。',generation_metadata={'eos_emitted':r['eos_emitted'],'capped':r['capped']})
    # Replace the outdated training-data examples (including deleted-reference examples) with actual current records.
    training=Path('runs/structural_20261114')
    chosen={};need={}
    for r in read(training/'selection_manifest.jsonl'):
        if r['mode']!='PT':continue
        key=(r['version'],r['condition'])
        chosen.setdefault(key,[])
        if len(chosen[key])<2 and all(x['source_key']!=r['source_key'] for x in chosen[key]):
            chosen[key].append(r);need[r['source_pool'],r['version'],r['source_row']]=r
    for pool in ('sft','pt'):
        for version in ('v2','v3'):
            for i,line in enumerate((training/pool/f'{version}_train.jsonl').open()):
                if (pool,version,i) not in need:continue
                r=json.loads(line);cn='原样' if r['condition']=='original' else '结构修改'
                group=f'结构修改训练数据 · {version}';label=f'{cn} · {r["source_key"]}'
                for kind,text in [('input',r['prompt']),('reference',r['prompt']+r['completion'])]:
                    add(f'data/{r["id"]}_{kind}.mpd',text,r['n_prefix'],'posttrain_icl',group,label+(' · 输入' if kind=='input' else ' · 参考'),
                        dataset_version=int(version[-1]),is_reference=kind=='reference',caption=r['caption'],
                        note='实际活动训练记录。目标 M 高亮显示。',collisions=[],condition=r['condition'])
    assert len(entries)==2596,len(entries)
    _write_json(STAGE/'models/manifest.json',entries)
    write_catalog(STAGE/'models',entries)
    _write_json(STAGE/'unified_compare.json',boards['unified'])
    _write_json(STAGE/'nzd_order_compare.json',boards['nzd'])
    _write_json(STAGE/'rollback_conditions.json',rollback)
    for name in ('index.html','unified_compare.html','nzd_order_compare.html','rollback_compare.html','colors.ldr'):
        shutil.copyfile(LIVE/name,STAGE/name)
    vendor=STAGE/'vendor-three-170'
    if not vendor.exists():vendor.symlink_to(LIVE/'vendor-three-170')
    page=STAGE/'unified_compare.html';html=page.read_text()
    html=html.replace('同一批18个固定案例 · 全部0.6B · ','结构修改后训练新版 · 同一批18个固定案例 · 全部0.6B · ')
    old_line="score.textContent='位姿 F1 '+s.pose_f1.toFixed(3)+' · 零件 F1 '+s.node_f1.toFixed(3);"
    assert old_line in html
    html=html.replace(old_line,"score.textContent='位姿 F1（按答案长度截断） '+s.pose_f1.toFixed(3);")
    page.write_text(html)
    page=STAGE/'rollback_compare.html';html=page.read_text().replace('位姿 F1（不计颜色）','位姿 F1（截断、不计颜色）')
    page.write_text(html)
    atomic(ROOT/'VIEWER_STAGED.json',{'models':len(entries),'evaluation':ROOT.name,
        'replace_tasks':['completion_no_head','completion_given_head','rollback_pt','from_scratch','posttrain_icl'],
        'model_run':'structural_20261114','deletion_condition_count':0})
    print('STAGED VIEWER',len(entries),flush=True)
if __name__=='__main__':main()
