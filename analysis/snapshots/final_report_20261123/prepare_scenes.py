import copy,hashlib,importlib.util,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from bricknet import parse_sample,tree_to_graph,graph_to_ldr,decode_graph
from brickicl.eval.match import same_pose
from brickicl.viz.lazy_catalog import _write_json,write_catalog
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];REPORT=ROOT/'report';LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer');WEB=LIVE/'report_figures_20261123';EVAL=REPO/'runs/eval100_official_dfs_20261122'
def read(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def matching_reference(tree,cut):
 target=set(range(cut,len(tree.parts)));kids=defaultdict(list)
 for e in tree.edges:kids[e.parent].append(e.child)
 mats=np.asarray(decode_graph(tree_to_graph(tree)));want=Counter(tree.parts[i].part_id for i in target);mt=np.linalg.inv(mats[cut])[None]@mats[list(sorted(target))];refs=[]
 for root in range(cut):
  nodes=[];todo=[root]
  while todo:i=todo.pop();nodes.append(i);todo.extend(kids[i])
  if set(nodes)&target or Counter(tree.parts[i].part_id for i in nodes)!=want:continue
  rt=np.linalg.inv(mats[root])[None]@mats[nodes];match=np.array([[tree.parts[a].part_id==tree.parts[b].part_id and same_pose(tree.parts[a].part_id,ma,mb) for b,mb in zip(nodes,rt)] for a,ma in zip(sorted(target),mt)],dtype=np.int8)
  if int(match[linear_sum_assignment(-match)].sum())==len(target):refs.extend(nodes)
 return sorted(set(refs))
def main():
 WEB.mkdir(exist_ok=True);(WEB/'models').mkdir(exist_ok=True);(REPORT/'assets/renders').mkdir(parents=True,exist_ok=True);(REPORT/'assets/ldraw').mkdir(exist_ok=True)
 for p in LIVE.iterdir():
  if p.name in ['models','index.html','report_figures_20261123','official_dfs_review','grpo_old18']:continue
  dst=WEB/p.name
  if not dst.exists():dst.symlink_to(p,target_is_directory=p.is_dir())
 html=(LIVE/'index.html').read_text().replace('const dir = new THREE.Vector3(0.75, 0.55, 1).normalize();',"const cameraDirection = state.entry?.report_camera_dir || [0.75,0.55,1];\n  const dir = new THREE.Vector3(...cameraDirection).normalize();")
 html=html.replace('  fitCamera();\n  if (window.parent', '  fitCamera();\n  window.reportReadyFile=entry.file;\n  if (window.parent');(WEB/'index.html').write_text(html)
 mod=importlib.util.spec_from_file_location('pack',LIVE/'pack.py');pack=importlib.util.module_from_spec(mod);mod.loader.exec_module(pack);lib=pack.build_index();entries=[];provenance=[]
 rows={r['id']:r for r in read(EVAL/'p1_inputs.jsonl')};cache={a:{r['id']:r for r in read(EVAL/'results'/a/'p1.scored.jsonl')} for a in ['frozen_pt','post_pt','grpo_pt']}
 def add(name,text,cut,collisions=[],recolor=None,**meta):
  text=text if text.endswith('\n') else text+'\n';p=parse_sample(text);lines=[s for s in graph_to_ldr(tree_to_graph(p.tree)).splitlines() if s.startswith('1 ')]
  if recolor:
   changed=[]
   for i,line in enumerate(lines):
    fields=line.split();fields[1]=str(recolor.get(i,71));changed.append(' '.join(fields))
   lines=changed
  mpd,missing=pack.embed(lines,lib);assert not missing;file=name+'.mpd';(WEB/'models'/file).write_text(mpd);shutil.copy2(WEB/'models'/file,REPORT/'assets/ldraw'/file)
  entry={'file':file,'path':text,'n_parts':len(p.tree.parts),'parse_error':p.error,'heldout_from':cut,'collisions':collisions,'task':'posttrain_icl','group':'Report figure sources','title':name,'report_camera_dir':[-.75,.55,-1] if 'fox' in name else [0.75,.55,1],**meta};entries.append(entry);provenance.append({'scene':name,'file':file,'source':meta})
 for condition in ['original','stretched']:
  r=rows[f'val:114:{condition}'];text=r['prefix']+r['target_'+r['expected']];tree=parse_sample(text).tree;cut=r['n_prefix']-1;refs=matching_reference(tree,cut);assert refs,(condition,'no matching visible module');colors={i:14 for i in refs};colors.update({i:3 for i in range(cut,len(tree.parts))});add('pipeline_fox_'+condition,text,cut,recolor=colors,reference_indices=refs,target_indices=list(range(cut,len(tree.parts))),ground_truth=True,source_id=r['id'])
 for sid,sample,label in [(114,2,'fox'),(497,0,'spacecraft')]:
  r=rows[f'val:{sid}:stretched'];add(label+'_target',r['prefix']+r['target_long'],r['n_prefix'],ground_truth=True,source_id=r['id'])
  for arm in ['frozen_pt','post_pt','grpo_pt']:
   s=cache[arm][r['id']]['samples'][sample];add(label+'_'+arm,r['prefix']+s['text'],r['n_prefix'],s['collisions'],source_id=r['id'],sample_index=sample,arm=arm,metrics={k:s[k] for k in ['pose_f1','pose_f1_untruncated','n_generated','target_parts','parse_ok','generation_collision_free']},ground_truth=False)
 # Four real, heterogeneous benchmark cases for the appendix. All are ground-truth constructions.
 for sid in [20,489,499,417]:
  for condition in ['original','stretched']:
   r=rows[f'val:{sid}:{condition}'];add(f'dataset_{sid}_{condition}',r['prefix']+r['target_'+r['expected']],r['n_prefix'],source_id=r['id'],ground_truth=True)
 _write_json(WEB/'models/manifest.json',entries);write_catalog(WEB/'models',entries)
 (REPORT/'assets/render_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2));print('Prepared',len(entries),'actual viewer scenes',flush=True)
if __name__=='__main__':main()
