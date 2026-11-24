import copy,hashlib,importlib.util,json,os,shutil
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr
from brickicl.viz.lazy_catalog import _write_json,write_catalog
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer');STAGE=ROOT/'viewer_stage'
LABELS={'nzd_v2_plain':'Qwen3 直接 PT · 对照 · 2211','nzd_v2_masked':'Qwen3 直接 PT · 增强 · 2211','regularized_g_wd010_do015':'Full-G 正则化 · WD0.10 / dropout0.15 · 1000'}
def j(p):return json.loads(p.read_text())
def read(p):return [json.loads(s) for s in p.open() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 entries=copy.deepcopy(j(LIVE/'models/manifest.json'));original_files={r['file'] for r in entries};catalog=j(LIVE/'models/catalog.json');added=[]
 assert not any(r.get('arm') in LABELS for r in entries)
 old=j(LIVE/'unified_compare.json');order=['114']+[s for s in old['source_order'] if s!='114'];board={'schema':1,'arms':{'frozen_pt':'官方 PT',**LABELS},'samples':4,'source_count':18,'source_order':order,'sources':{},'evaluation_run':ROOT.name}
 module=importlib.util.spec_from_file_location('viewer_pack',LIVE/'pack.py');pack=importlib.util.module_from_spec(module);module.loader.exec_module(pack);lib=pack.build_index()
 (STAGE/'models').mkdir(parents=True,exist_ok=True)
 def add(rel,text,cut,spec,**meta):
  display=text if text.endswith('\n') else text+'\n';parsed=parse_sample(display)
  lines=graph_to_ldr(tree_to_graph(parsed.tree)).splitlines() if parsed.tree.parts else []
  mpd,missing=pack.embed([s for s in lines if s.startswith('1 ')],lib);assert not missing,(rel,missing)
  file=ROOT.name+'/'+rel;p=STAGE/'models'/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(mpd)
  r={'file':file,'path':display,'n_parts':len(parsed.tree.parts),'parse_error':parsed.error,'heldout_from':cut,'arm':spec['arm'],'base':spec['base'],'adapters':spec['adapters'],'checkpoint':spec['checkpoint'],'size_kb':round(p.stat().st_size/1024),'input_revision':'fox_duplicate_removed_20261121','evaluation_run':ROOT.name}|meta
  entries.append(r);added.append(r);return file
 for m in j(ROOT/'delivery/metrics.json'):
  spec=m['model'];arm=spec['arm'];label=LABELS[arm];ev=Path(m['evaluation_root']);inputs={r['id']:r for r in read(ev/'p1_inputs.jsonl')};scored={r['id']:r for r in read(ev/'results'/arm/'p1.scored.jsonl')}
  for sid in order:
   source=old['sources'][sid];dst=board['sources'].setdefault(sid,{'label':source['label'],'caption':source['caption'],'primes':{'1':{'prime':1,'conditions':{}}}})
   for cond,cn in [('original','正常'),('stretched','加长')]:
    row=inputs[f'val:{sid}:{cond}'];case=dst['primes']['1']['conditions'].get(cond)
    if case is None:
     case=copy.deepcopy(source['primes']['1']['conditions'][cond]);assert case['input_sha256']==row['prefix_sha256'];case['outputs']={'frozen_pt':case['outputs']['frozen_pt']};dst['primes']['1']['conditions'][cond]=case
    case['outputs'][arm]=[]
    for i,s in enumerate(scored[row['id']]['samples']):
     metrics={k:s[k] for k in ('pose_f1','node_f1','parse_ok','n_generated','exact','eos_emitted')}
     file=add(f'completion/{arm}/{sid}/{cond}_s{i+1}.mpd',row['prefix']+s['text'],row['n_prefix'],spec,
       task='completion_given_head',group=label,model_label=label,title=f'{source["label"]} · {cn} · 样本 {i+1}/4',source=int(sid),prime=1,condition=cond,sample_index=i,caption=source['caption'],collisions=s['collisions'],evaluation_metrics=metrics,
       note='开发集最佳checkpoint；18个固定案例，各条件4次。狐狸输入已去重；无回退。',source_record=str(ev/'results'/arm/'p1.scored.jsonl'))
     assert added[-1]['n_parts']==row['n_prefix']+s['n_generated']
     case['outputs'][arm].append({'file':file}|metrics)
  for mode,cn,path in [('natural','允许 EOS',ev/'results'/arm/'free.scored.jsonl'),('no_eos','禁止 EOS',ev/'no_eos/results'/arm/'free_no_eos.scored.jsonl')]:
   rows=read(path);assert len(rows)==64
   for i,r in enumerate(rows[:4]):
    assert r['adapters']==spec['adapters'] and not r.get('caption') and r['text'].startswith('a')
    add(f'from_scratch/{arm}/{mode}_s{i+1}.mpd',r['text'],0,spec,task='from_scratch',group='PT · '+label,title=f'{cn} · 样本{i+1}/4',from_scratch=True,input_mode='unconditional',input_prompt='a',caption='',sample_index=i,eos_condition=mode,collisions=r['collisions'],
      generation_metadata={'eos_emitted':r['eos_emitted'],'capped':r['capped']},evaluation_metrics={'valid_length':r['valid_length'],'collision_free_length':r['collision_free_length']},source_record=str(path),note=f'输入仅起始符a，无自然语言或积木参照。{cn}的64次评测前4次，未挑选成功样本。最多100块。')
 assert len(added)==456
 tasks=list(dict.fromkeys(r['task'] for r in entries));entries.sort(key=lambda r:tasks.index(r['task']))
 for r in entries:
  if r['file'] not in original_files:continue
  p=STAGE/'models'/r['file'];p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():os.link(LIVE/'models'/r['file'],p)
 _write_json(STAGE/'models/manifest.json',entries);write_catalog(STAGE/'models',entries)
 cat=j(STAGE/'models/catalog.json');cat['aliases']=catalog.get('aliases',{});_write_json(STAGE/'models/catalog.json',cat)
 assert len([g for g in cat['groups'] if g['task']=='completion_given_head'])==18
 assert all(g['count']==144 for g in cat['groups'] if g['task']=='completion_given_head')
 assert len([g for g in cat['groups'] if g['task']=='posttrain_icl'])==5
 _write_json(STAGE/'latest_compare.json',board)
 html=(LIVE/'unified_compare.html').read_text().replace('unified_compare.json','latest_compare.json').replace('六模型','四模型').replace('六个模型','四个模型').replace('统一测试：同一案例','最新训练结果：同一案例')
 (STAGE/'latest_compare.html').write_text(html)
 index=(LIVE/'index.html').read_text();anchor='<a id="grpoCompareLink"';assert anchor in index
 index=index.replace(anchor,'<a id="latestCompareLink" href="latest_compare.html?source=114&amp;prime=1" style="margin:0 12px 10px;color:var(--accent);font-weight:600">最新对照／增强／正则化 · 渲染对照 ↗</a>\n  '+anchor,1)
 (STAGE/'index.html').write_text(index)
 _write_json(ROOT/'VIEWER_STAGED.json',{'new_entries':len(added),'total_entries':len(entries),'aliases':len(cat['aliases']),'source_manifest_sha256':sha(LIVE/'models/manifest.json'),'source_index_sha256':sha(LIVE/'index.html'),'new_files':[r['file'] for r in added]})
 print('Viewer staged:',len(entries),flush=True)
if __name__=='__main__':main()
