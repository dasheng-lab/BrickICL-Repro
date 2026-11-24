import copy,importlib.util,json,shutil
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr
from brickicl.viz.lazy_catalog import _write_json,write_catalog
from common import *

LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
STAGE=ROOT/'viewer_stage'

def main():
    assert json.loads((ROOT/'VERIFIED.json').read_text())['all_completed']
    specs={s['arm']:s for s in plan()['checkpoints']};old=json.loads((LIVE/'unified_compare.json').read_text())
    arms={'frozen_pt':'官方 BrickNet PT','post_pt':'BrickNet 初始化 · M-only · 750','fullg_pt':'BrickNet 初始化 · Full-G · 500'}
    arms.update({a:specs[a]['label']+f' · {specs[a]["step"]}' for a in ('direct_bricknet','direct_m','direct_g')})
    board={'schema':1,'arms':arms,'samples':4,'source_count':18,'source_order':old['source_order'],'sources':{},'evaluation_run':ROOT.name}
    module=importlib.util.spec_from_file_location('viewer_pack',LIVE/'pack.py');pack=importlib.util.module_from_spec(module);module.loader.exec_module(pack)
    lib=pack.build_index();entries=[];(STAGE/'models').mkdir(parents=True,exist_ok=True)
    def add(rel,text,cut,**meta):
        parsed=parse_sample(text);lines=graph_to_ldr(tree_to_graph(parsed.tree)).splitlines() if parsed.tree.parts else []
        mpd,missing=pack.embed([l for l in lines if l.startswith('1 ')],lib);assert not missing,(rel,missing)
        file=ROOT.name+'/'+rel;dest=STAGE/'models'/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(mpd)
        entry={'file':file,'path':text,'n_parts':len(parsed.tree.parts),'parse_error':parsed.error,'heldout_from':cut,
               'evaluation_run':ROOT.name,'size_kb':round(dest.stat().st_size/1024)}|meta
        entries.append(entry);return file
    for prime in (0,1):
        inputs={r['id']:r for r in read(ROOT/f'p{prime}_inputs.jsonl')}
        scored={a:{r['id']:r for r in read(folder(s)/f'p{prime}.scored.jsonl')} for a,s in specs.items()}
        for sid in old['source_order']:
            source=old['sources'][sid];target=board['sources'].setdefault(sid,{'label':source['label'],'caption':source['caption'],'primes':{}})
            target['primes'][str(prime)]={'prime':prime,'conditions':{}}
            for condition,cn in [('original','正常'),('stretched','加长')]:
                row=inputs[f'val:{sid}:{condition}'];case=copy.deepcopy(source['primes'][str(prime)]['conditions'][condition])
                assert case['input_sha256']==row['prefix_sha256'] and case['prefix_parts']==row['n_prefix']
                case['target_parts']=scored['direct_m'][row['id']]['samples'][0]['target_parts']
                case['outputs']={a:case['outputs'][a] for a in ('frozen_pt','post_pt','fullg_pt')}
                for arm,spec in specs.items():
                    outputs=[]
                    for i,s in enumerate(scored[arm][row['id']]['samples']):
                        metrics={k:s[k] for k in ('pose_f1','node_f1','parse_ok','n_generated','exact','eos_emitted')}
                        file=add(f'completion/p{prime}/{sid}/{condition}_{arm}_s{i+1}.mpd',row['prefix']+s['text'],row['n_prefix'],
                            task='completion_given_head' if prime else 'completion_no_head',
                            group='Qwen3 直接 PT · '+source['label'],title=f'{cn} · {arms[arm]} · 样本 {i+1}',
                            arm=arm,source=int(sid),prime=prime,condition=condition,sample_index=i,
                            caption=source['caption'],collisions=s['collisions'],evaluation_metrics=metrics,
                            adapters=spec['adapters'],base=spec['base'],checkpoint=spec['checkpoint'],
                            note='原始 Qwen3 直接 PT，按开发集选权重；没有加载 BrickNet adapter。位姿 F1 按答案长度截断，显示完整可解析输出。')
                        assert entries[-1]['n_parts']==row['n_prefix']+s['n_generated']
                        outputs.append({'file':file}|metrics)
                    case['outputs'][arm]=outputs
                target['primes'][str(prime)]['conditions'][condition]=case
    for arm,spec in specs.items():
        rows=read(folder(spec)/'free.scored.jsonl')
        for i,r in enumerate(rows[:4]):
            add(f'independent/{arm}_s{i+1}.mpd',r['text'],0,task='from_scratch',group='Qwen3 直接 PT · 无条件生成',
                title=f'{arms[arm]} · 样本 {i+1}',arm=arm,from_scratch=True,input_mode='unconditional',input_prompt='a',
                sample_index=i,collisions=r['collisions'],adapters=spec['adapters'],base=spec['base'],checkpoint=spec['checkpoint'],
                note='64 次独立生成中的前四次；仅给起始符 a，最多展示100块。',
                generation_metadata={'eos_emitted':r['eos_emitted'],'capped':r['capped']},
                evaluation_metrics={'valid_length':r['valid_length'],'collision_free_length':r['collision_free_length']})
    assert len(entries)==876
    _write_json(STAGE/'models/manifest.json',entries);write_catalog(STAGE/'models',entries)
    old_manifest=json.loads((LIVE/'models/manifest.json').read_text());old_catalog=json.loads((LIVE/'models/catalog.json').read_text())
    new_catalog=json.loads((STAGE/'models/catalog.json').read_text())
    assert not set(g['id'] for g in new_catalog['groups']) & set(g['id'] for g in old_catalog['groups'])
    _write_json(STAGE/'models/catalog.json',{'schema':2,'groups':new_catalog['groups']+old_catalog['groups']})
    _write_json(STAGE/'models/manifest.json',entries+old_manifest)
    _write_json(STAGE/'qwen3_direct_compare.json',board)
    html=(LIVE/'unified_compare.html').read_text()
    html=html.replace('unified_compare.json','qwen3_direct_compare.json').replace('统一测试 · 六模型同输入对照','Qwen3 直接 PT · 六模型对照')
    html=html.replace('统一测试：同一案例，六个模型','Qwen3 直接 PT：与 BrickNet 初始化同台比较')
    html=html.replace('结构修改后训练新版 · 同一批18个固定案例','Qwen3 直接 PT 与 BrickNet 对照 · 同一批18个固定案例')
    html=html.replace('<a href="./?v=fox-compare">返回完整 Viewer</a>','<a href="./?v=fox-compare">返回完整 Viewer</a><a href="unified_compare.html?source=114&prime=1">此前模型对照</a>')
    (STAGE/'qwen3_direct_compare.html').write_text(html)
    index=(LIVE/'index.html').read_text();anchor='<a id="foxCompareLink"'
    assert anchor in index
    index=index.replace(anchor,'<a id="directQwenLink" href="qwen3_direct_compare.html?source=114&amp;prime=1" style="margin:0 12px 10px;color:var(--accent);font-weight:600">Qwen3 直接 PT · 新三组与基线对照 ↗</a>\n  '+anchor,1)
    (STAGE/'index.html').write_text(index)
    atomic(ROOT/'VIEWER_STAGED.json',{'new_models':len(entries),'total_models':len(entries)+len(old_manifest),
        'new_outputs':864,'independent_gallery':12,'board_arms':list(arms),'live_manifest_sha256':sha(LIVE/'models/manifest.json'),
        'live_index_sha256':sha(LIVE/'index.html'),'old_two_copy':sum(r.get('task')=='two_copy' for r in old_manifest)})
    print('STAGED',len(entries),flush=True)
if __name__=='__main__':main()
