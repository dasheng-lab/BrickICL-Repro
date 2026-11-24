import json,shutil
from collections import defaultdict
from pathlib import Path
from bricknet import parse_sample,tree_to_graph,graph_to_ldr
from brickicl.viz.pack_probe import viewer_pack
from brickicl.viz.lazy_catalog import write_catalog
root=Path('runs/structural_20261114')
rows=[json.loads(l) for l in (root/'smoke_rows.jsonl').open()]
groups=defaultdict(list)
for r in rows:groups[r['source_key'],r['version'],r['group']['index']].append(r)
out=root/'examples';site=out/'site';(site/'models').mkdir(parents=True,exist_ok=True)
existing=Path('runs/dfs_posttrain_20261114/caption_audit/site')
shutil.copyfile(existing/'index.html',site/'index.html')
shutil.copyfile(existing/'colors.ldr',site/'colors.ldr')
vendor=site/'vendor-three-170'
if not vendor.exists():vendor.symlink_to(Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer/vendor-three-170'))
pack=viewer_pack();index=pack.build_index();manifest=[];chosen=[]
for key,rs in groups.items():
    normal=next((r for r in rs if r['condition']=='original'),None)
    changed=next((r for r in rs if r['edit']['kind']=='tip_extension'),None) or next((r for r in rs if r['edit']['kind']=='stack_insert'),None)
    if not normal or not changed:continue
    number=len(chosen)+1
    for state,r in [('before',normal),('after',changed)]:
        tree=parse_sample(r['full_dfs_path']).tree
        lines=[l for l in graph_to_ldr(tree_to_graph(tree)).splitlines() if l.startswith('1 ')]
        mpd,missing=pack.embed(lines,index);assert not missing
        file=f'{number}-{state}.mpd';(site/'models'/file).write_text(mpd)
        manifest.append({'file':file,'task':'examples','group':'Subassembly edits','title':str(key)+' '+state,
                         'n_parts':len(tree.parts),'path':r['full_dfs_path'],'caption':r['caption'],'parse_error':None,'collisions':[]})
    chosen.append({'number':number,'source':normal['source_key'],'version':normal['version'],
        'before_parts':normal['selected_parts'],'after_parts':changed['selected_parts'],
        'before_geometry':normal['geometry_id'],'after_geometry':changed['geometry_id'],'edit':changed['edit'],
        'before_caption':normal['caption'],'after_caption':changed['caption']})
    if len(chosen)==3:break
assert any(r['edit']['kind']=='tip_extension' for r in chosen)
write_catalog(site/'models',manifest)
(out/'examples.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
print(json.dumps([{k:r[k] for k in ('number','source','version','before_parts','after_parts','edit')} for r in chosen]))
