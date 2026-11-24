import copy,hashlib,importlib.util,json,os,shutil
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr
from brickicl.viz.lazy_catalog import _write_json,write_catalog
ROOT=Path(__file__).resolve().parent;LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
def j(p):return json.loads(p.read_text())
def main():
 backup=ROOT/'viewer_before';backup.mkdir(exist_ok=True)
 old=j(backup/'manifest.json') if (backup/'manifest.json').exists() else j(LIVE/'models/manifest.json');catalog=j(backup/'catalog.json') if (backup/'catalog.json').exists() else j(LIVE/'models/catalog.json')
 for name in ('manifest.json','catalog.json'):
  if not (backup/name).exists():shutil.copy2(LIVE/'models'/name,backup/name)
 cases=j(LIVE/'unified_compare.json')['sources'];new={str(s['source']):s for s in j(ROOT/'synthetic_sources.json')};rows=[json.loads(s) for s in (ROOT/'p1_inputs.jsonl').open()]
 mod=importlib.util.spec_from_file_location('pack',LIVE/'pack.py');pack=importlib.util.module_from_spec(mod);mod.loader.exec_module(pack);lib=pack.build_index()
 entries=copy.deepcopy(old);board={'source_order':['114']+[str(r['source_key'].split(':')[-1]) for r in rows if r['condition']=='original' and r['source_key']!='val:114'],'sources':{}}
 for r in entries:
  if r['task']=='posttrain_icl' and r['group'].startswith('测试数据'):r['group']='测试数据（100 个固定案例）';r['dataset_category']=r['group']
 for row in rows:
  sid=row['source_key'].split(':')[-1];source=cases.get(sid,new.get(sid));case=board['sources'].setdefault(sid,{'label':source['label'],'caption':source['caption'],'origin':'original_validation' if sid in cases else 'constructed','conditions':{}});condition=row['condition'];case['conditions'][condition]={}
  for kind,text in [('input',row['prefix']),('reference',row['prefix']+row['target_'+row['expected']])]:
   existing=next((r for r in entries if r['task']=='posttrain_icl' and str(r.get('source'))==sid and r.get('condition')==condition and r.get('is_input',False)==(kind=='input') and (r.get('is_input') or r.get('is_reference'))),None)
   if existing:file=existing['file'];assert existing['path'].rstrip('\n')==text.rstrip('\n')
   else:
    parsed=parse_sample(text);assert parsed.error is None
    lines=graph_to_ldr(tree_to_graph(parsed.tree)).splitlines();mpd,missing=pack.embed([s for s in lines if s.startswith('1 ')],lib);assert not missing
    file=f'{ROOT.name}/data/{sid}_{condition}_{kind}.mpd';dest=LIVE/'models'/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(mpd)
    entries.append({'file':file,'path':text,'n_parts':len(parsed.tree.parts),'parse_error':None,'collisions':[],'heldout_from':row['n_prefix'],'task':'posttrain_icl','group':'测试数据（100 个固定案例）','dataset_category':'测试数据（100 个固定案例）','title':source['label']+' · '+('正常' if condition=='original' else '加长')+' · '+('给首块输入' if kind=='input' else '目标参考'),'source':int(sid),'condition':condition,'prime':1,'is_input':kind=='input','is_reference':kind=='reference','caption':source['caption'],'origin':'constructed_eval_only','source_record':str(ROOT/'p1_inputs.jsonl'),'note':'从空白搭建的新测试作品；不用于训练。输入和标准答案已通过解析、连通与碰撞检查。','size_kb':round(dest.stat().st_size/1024)})
   case['conditions'][condition][kind]=file
 _write_json(LIVE/'models/manifest.json',entries);write_catalog(LIVE/'models',entries);cat=j(LIVE/'models/catalog.json');cat['aliases']=catalog['aliases'];_write_json(LIVE/'models/catalog.json',cat)
 _write_json(LIVE/'eval100_review.json',board)
 html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>100案例测试集检查</title><style>body{margin:0;background:#f1f3f6;color:#172b43;font:14px Arial,sans-serif}header{background:white;padding:16px 24px}h1{font-size:21px;margin:0 0 12px}select,button{font:inherit;padding:8px;border:1px solid #cad4e2;border-radius:7px;background:white}main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}.card{background:white;border:1px solid #dbe2ee;border-radius:10px;overflow:hidden}h2{font-size:16px;margin:12px}iframe{width:100%;height:540px;border:0}p{color:#5c6c80}a{color:#2563eb}</style><header><h1>100案例测试集：正常与加长</h1><select id="source"></select> <button id="mode">切换输入 / 标准答案</button> <a href="./">返回Viewer</a><p id="caption"></p><p>18个原测试作品 + 82个从空白构造的新作品；每作品只有一组正常/加长测试。绿色为待补部分。当前展示标准答案，不是模型生成。</p></header><main><section class="card"><h2>正常</h2><iframe id="original"></iframe></section><section class="card"><h2>加长（各支脚多4块）</h2><iframe id="stretched"></iframe></section></main><script>
let data,kind='reference';const select=document.getElementById('source');function render(){let s=data.sources[select.value];document.getElementById('caption').textContent=s.caption+' · '+(kind==='reference'?'标准答案':'给首块输入');for(let c of ['original','stretched']){let f=document.getElementById(c);f.dataset.ready='false';f.dataset.file=s.conditions[c][kind];f.src='./?embed=1#'+encodeURIComponent(f.dataset.file)}history.replaceState(null,'','?source='+select.value)}window.addEventListener('message',e=>{if(e.origin!==location.origin||e.data?.type!=='bricknet-viewer-ready')return;for(let f of document.querySelectorAll('iframe'))if(f.contentWindow===e.source&&f.dataset.file===e.data.file)f.dataset.ready='true'});select.onchange=render;document.getElementById('mode').onclick=()=>{kind=kind==='reference'?'input':'reference';render()};fetch('eval100_review.json').then(r=>r.json()).then(d=>{data=d;for(let id of d.source_order){let o=document.createElement('option');o.value=id;o.textContent=d.sources[id].label;select.append(o)}select.value=new URLSearchParams(location.search).get('source')||'114';render()});</script></html>'''
 (LIVE/'eval100_review.html').write_text(html);_write_json(ROOT/'DATA_PREVIEW_PUBLISHED.json',{'entries':len(entries),'new_data_entries':len(entries)-len(old),'sources':100,'url':'/viewer/eval100_review.html','metrics_not_changed':True});print('100-case input/reference viewer published',flush=True)
if __name__=='__main__':main()
