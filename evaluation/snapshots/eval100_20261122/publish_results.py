"""Publish only a complete, checksum-verified 100-case run; preserve historical raw evaluations."""
import copy,hashlib,importlib.util,json,os,re,shutil,time
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr
from brickicl.viz.lazy_catalog import _write_json,write_catalog
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer');PLAY=Path('/home/sunshq114514/brickicl-gradio-service/brickicl/playground');STAGE=ROOT/'viewer_stage';DELIVERY=ROOT/'delivery'
def j(p):return json.loads(p.read_text())
def read(p):return [json.loads(s) for s in p.open() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (ROOT/'COMPLETED.json').exists();p=j(ROOT/'plan.json');assert j(ROOT/'VISUAL_APPROVED.json')['approved'] and sha(ROOT/'p1_inputs.jsonl')==p['input_sha256']
 metrics={};inputs={r['id']:r for r in read(ROOT/'p1_inputs.jsonl')};preview=j(LIVE/'eval100_review.json')
 for spec in p['models']:
  dest=ROOT/'results'/spec['arm'];v=j(dest/'VERIFIED.json');assert v['sources']==100 and v['input_sha256']==p['input_sha256']
  for file,digest in v['output_sha256'].items():assert sha(dest/file)==digest
  m=j(dest/'metrics.json');assert m['completion']['1']['samples']==800 and m['no_eos']['eos_suppression_verified'];metrics[spec['arm']]=m
 before=j(LIVE/'models/manifest.json');oldcat=j(LIVE/'models/catalog.json');labels={r['arm']:re.sub(r'^(PT|SFT) · ','',r['group']) for r in before if r['task']=='from_scratch'}
 replaced=[r for r in before if r.get('arm') in metrics and r['task'] in ('completion_given_head','from_scratch')]
 entries=[copy.deepcopy(r) for r in before if r not in replaced];source_order=preview['source_order'];generated=[];mapping={};lookup={};(STAGE/'models').mkdir(parents=True,exist_ok=True)
 mod=importlib.util.spec_from_file_location('pack',LIVE/'pack.py');pack=importlib.util.module_from_spec(mod);mod.loader.exec_module(pack);lib=pack.build_index()
 def add(file,text,cut,spec,**meta):
  text=text if text.endswith('\n') else text+'\n';parsed=parse_sample(text);lines=graph_to_ldr(tree_to_graph(parsed.tree)).splitlines() if parsed.tree.parts else [];mpd,missing=pack.embed([s for s in lines if s.startswith('1 ')],lib);assert not missing,(file,missing)
  dest=STAGE/'models'/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(mpd)
  entry={'file':file,'path':text,'heldout_from':cut,'n_parts':len(parsed.tree.parts),'parse_error':parsed.error,'arm':spec['arm'],'base':spec['base'],'adapters':spec['adapters'],'checkpoint':spec.get('checkpoint'),'evaluation_run':ROOT.name,'size_kb':round(dest.stat().st_size/1024),'input_revision':'100case_constructed_20261122'}|meta
  entries.append(entry);generated.append(entry);return entry
 for spec in p['models']:
  arm=spec['arm'];label=labels[arm];dest=ROOT/'results'/arm;rs={r['id']:r for r in read(dest/'p1.scored.jsonl')}
  for sid in source_order:
   source=preview['sources'][sid]
   for condition,cn in [('original','正常'),('stretched','加长')]:
    key=('synthetic:' if int(sid)>=1000 else 'val:')+sid+':'+condition;row=inputs[key]
    for i,s in enumerate(rs[key]['samples']):
     met={k:s[k] for k in ('pose_f1','node_f1','parse_ok','n_generated','exact','eos_emitted')}
     entry=add(f'{ROOT.name}/completion/{arm}/{sid}/{condition}_s{i+1}.mpd',row['prefix']+s['text'],row['n_prefix'],spec,task='completion_given_head',group=label,model_label=label,title=f'{source["label"]} · {cn} · 样本 {i+1}/4',source=int(sid),prime=1,condition=condition,sample_index=i,caption=source['caption'],collisions=s['collisions'],evaluation_metrics=met,source_record=str(dest/'p1.scored.jsonl'),note='100案例统一评测；给第一块，正常/加长各4次，无caption、无回退。18原案例+82新构造作品。')
     lookup[arm,sid,condition,i]={'file':entry['file'],**met,'parse_ok':entry['parse_error'] is None,'raw_parse_ok':s['parse_ok']}
  for mode,cn,path in [('natural','允许 EOS','free.scored.jsonl'),('no_eos','禁止 EOS','free_no_eos.scored.jsonl')]:
   for i,r in enumerate(read(dest/path)[:4]):
    entry=add(f'{ROOT.name}/from_scratch/{arm}/{mode}_s{i+1}.mpd',r['text'],0,spec,task='from_scratch',group='PT · '+label,title=f'{cn} · 样本{i+1}/4',from_scratch=True,input_mode='unconditional',input_prompt='a',caption='',sample_index=i,eos_condition=mode,collisions=r['collisions'],source_record=str(dest/path),evaluation_metrics={'valid_length':r['valid_length'],'collision_free_length':r['collision_free_length']},generation_metadata={'eos_emitted':r['eos_emitted'],'capped':r['capped']},note='本轮重新无条件生成，只给a。各64次中固定前4次，最多展示100块。')
    lookup[arm,'scratch',mode,i]={'file':entry['file']}
  print('PACKED',arm,flush=True)
 for r in replaced:
  key=(r['arm'],str(r['source']),r['condition'],r['sample_index']) if r['task']=='completion_given_head' else (r['arm'],'scratch',r['eos_condition'],r['sample_index']);mapping[r['file']]=lookup[key]['file']
 task_order=list(dict.fromkeys(r['task'] for r in before));arm_order=[s['arm'] for s in p['models']]+['frozen_sft','post_sft','fullg_sft']
 def rank(r):
  t=r['task'];a=r.get('arm')
  return (task_order.index(t),arm_order.index(a) if t in ('completion_given_head','from_scratch') else 0)
 entries.sort(key=rank)
 for r in entries:
  dest=STAGE/'models'/r['file']
  if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);os.link(LIVE/'models'/r['file'],dest)
 _write_json(STAGE/'models/manifest.json',entries);write_catalog(STAGE/'models',entries);cat=j(STAGE/'models/catalog.json');cat['aliases']={**oldcat.get('aliases',{}),**mapping};_write_json(STAGE/'models/catalog.json',cat)
 assert all(g['count'] in (800,144) for g in cat['groups'] if g['task']=='completion_given_head')
 # Update every existing board, expanding PT-only boards to all 100. Historical SFT comparisons keep their original 18 sources.
 boards=[]
 for file in LIVE.glob('*compare.json'):
  b=j(file)
  if not isinstance(b.get('arms'),dict) or not b.get('sources') or not b.get('source_order') or not set(b['arms'])&set(metrics):continue
  if not all('primes' in value for value in b['sources'].values()):continue
  full=set(b['arms'])<=set(metrics);ids=source_order if full else b['source_order']
  for sid in ids:
   src=preview['sources'][sid];s=b['sources'].setdefault(sid,{'label':src['label'],'caption':src['caption'],'primes':{}});cset=s['primes'].setdefault('1',{'prime':1,'conditions':{}})['conditions']
   for cond,cn in [('original','正常'),('stretched','加长')]:
    key=('synthetic:' if int(sid)>=1000 else 'val:')+sid+':'+cond;r=inputs[key];case=cset.setdefault(cond,{})
    case.update(label=cn,input=src['conditions'][cond]['input'],reference=src['conditions'][cond]['reference'],prefix_parts=r['n_prefix'],target_parts=r['k_'+r['expected']],input_sha256=r['prefix_sha256']);outputs=case.setdefault('outputs',{})
    for arm in b['arms']:
     if arm in metrics:outputs[arm]=[lookup[arm,sid,cond,i] for i in range(4)]
  b['source_order']=ids;b['source_count']=len(ids);b['evaluation_run']=ROOT.name;_write_json(STAGE/file.name,b);boards.append(file.name)
  htmlfile=LIVE/file.with_suffix('.html').name
  if full and htmlfile.exists():
   html=htmlfile.read_text().replace('18个固定案例','100个固定案例（18原案例＋82新构造）').replace('18个来源','100个来源');(STAGE/htmlfile.name).write_text(html)
 DELIVERY.mkdir(exist_ok=True);_write_json(DELIVERY/'metrics.json',metrics)
 groups=[('1. 官方模型',['frozen_pt']),('2. Bricknet posttrain',['post_pt','fullg_pt','masked','plain']),('3. Qwen posttrain',['direct_m','direct_g','nzd_v2_masked','nzd_v2_plain','regularized_g','regularized_g_wd010_do015']),('4. RL posttrain',['grpo_pt']),('5. 历史训练配方补充',['direct_bricknet','nzd_plain','nzd_masked'])]
 lines=['## PT 结果：100 个固定案例','','本轮100作品=18个原案例＋82个从空白构造的新作品。新增为12类程序化搭建规则下的不同几何设计，包含8种重复部件零件组合与上下两种朝向；没有把重采样、换色计为新作品。这是混合测试分布，不是100个自然官方validation作品。所有模型保持原开发集最佳checkpoint。','']
 for title,arms in groups:
  lines+=['### '+title,'','| 模型 + 数据 + 目标 | step | Δlog-odds | 正常 F1 | 加长 F1 | 有效长度 | 无碰撞长度 | 禁EOS有效长度 | 禁EOS无碰撞长度 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for arm in arms:
   m=metrics[arm];s=m['model'];c=m['completion']['1'];ph=m['physical'];ne=m['no_eos'];values=[c['delta'],c['f1']['original'],c['f1']['stretched'],ph['valid_length'],ph['collision_free_length'],ne['valid_length'],ne['collision_free_length']]
   lines.append('| '+' | '.join([s['recipe'],str(s.get('step','发布权重')),f'{values[0]:+.3f}',f'{values[1]:.4f}',f'{values[2]:.4f}']+[f'{x:.2f}' if x is not None else '—' for x in values[3:]])+' |')
  lines.append('')
 lines+=['**补全**：给第一块，正常/加长各4次，每模型800次。T=1、top-k20、top-p=.95，无caption、无回退。F1仍按剩余目标长度截断、型号与位姿匹配且忽略颜色；Δlog-odds仍使用长特征命中数与+0.5平滑，按100作品平均。','','**独立生成**：本轮重新生成允许/禁止EOS各64次，仅给a，4096 token上限，前100块评分。长度单位为积木；缺失碰撞网格的样本不计入无碰撞均值。','','**GRPO**：此行为可读的旧r1/step300。新WJY v2.1/step400权重仍不可读，因此未伪装成新GRPO结果。','','[100案例输入与标准答案](/viewer/eval100_review.html) · [最新Qwen组与官方PT对比](/viewer/latest_compare.html?source=114&prime=1) · [完整3D Viewer](/viewer/)']
 for arm,m in metrics.items():
  for mode in ('physical','no_eos'):
   if m[mode]['collision_coverage']!=64:lines.append(f'碰撞覆盖：{labels[arm]} / {mode} = {m[mode]["collision_coverage"]}/64。')
 summary='\n'.join(lines)+'\n';(DELIVERY/'results_summary.md').write_text(summary)
 backup=ROOT/'web_before_results';backup.mkdir(exist_ok=True)
 targets=[LIVE/'models/manifest.json',LIVE/'models/catalog.json',PLAY/'backend.py',PLAY/'results_summary.md',REPO/'README_MODELS_DATA_RESULTS.zh-CN.md']+[LIVE/n for n in boards]
 for path in targets:
  dest=backup/path.relative_to(Path('/home/sunshq114514'));dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(path,dest)
 for entry in generated:
  dest=LIVE/'models'/entry['file'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(STAGE/'models'/entry['file'],dest)
 for src in (STAGE/'models').rglob('*.json'):
  dest=LIVE/'models'/src.relative_to(STAGE/'models');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 for src in STAGE.iterdir():
  if src.is_file():shutil.copy2(src,LIVE/src.name)
 (PLAY/'results_summary.md').write_text(summary)
 backend=(PLAY/'backend.py').read_text();oldpath='runs/viewer_parse_fox_fix_20261121/p1_inputs.jsonl';assert oldpath in backend;backend=backend.replace(oldpath,'runs/eval100_20261122/p1_inputs.jsonl');compile(backend,str(PLAY/'backend.py'),'exec');(PLAY/'backend.py').write_text(backend)
 doc=REPO/'README_MODELS_DATA_RESULTS.zh-CN.md';text=doc.read_text();start=text.index('## 6. 与Gradio一致的结果');end=text.index('\n## 7.',start);text=text[:start]+'## 6. 与Gradio一致的结果\n\n'+summary+text[end:]
 text=text.replace('当前只展示给第一块：18个官方validation来源，正常/加长各4次，144次补全/模型。测试文件在 `runs/structural_eval_20261115/p1_inputs.jsonl`','当前只展示给第一块：100个来源（18原validation作品＋82新构造作品），正常/加长各4次，800次补全/模型。测试文件在 `runs/eval100_20261122/p1_inputs.jsonl`').replace('再平均18个来源','再平均100个来源')
 text+='\n## 100案例新版评测（2026-11-22）\n\n数据、程序与原始记录：`'+str(ROOT)+'`。新构造样本未用于任何训练。本轮15个PT模型各800次补全及允许/禁止EOS各64次已重新运行。Viewer给第一块按模型分组，PT每模型800项；3个历史SFT仍保留旧18案例结果，未参与本表。原18案例表及样本保留在旧run目录与本次web_before_results备份中。新增82个是12类规则、8种重复零件组合下的从空白搭建作品，统计上不是82种独立设计风格；正常/加长条件不重复计为独立作品。\n\n访问 `/viewer/eval100_review.html` 可逐案例核对输入和标准答案；“自建数据→测试数据”现有100作品、400项输入/参考。指标总表见上文第6节。\n';doc.write_text(text)
 _write_json(ROOT/'PUBLISHED.json',{'state':'published','sources':100,'models':15,'new_render_entries':len(generated),'total_entries':len(entries),'boards':boards,'unix_time':time.time()});print('Published 100-case metrics and viewer',flush=True)
if __name__=='__main__':main()
