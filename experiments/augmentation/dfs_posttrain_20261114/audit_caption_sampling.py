import hashlib,json,random,re,shutil
from collections import Counter,defaultdict
from pathlib import Path
from bricknet import parse_sample,graph_to_ldr,tree_to_graph
from brickicl.posttrain.reference_last import connected_subsets
from brickicl.viz.pack_probe import viewer_pack
from brickicl.viz.lazy_catalog import write_catalog

root=Path('runs/dfs_posttrain_20261114')
out=root/'caption_audit'
out.mkdir(exist_ok=True)
indices={v:[json.loads(x) for x in (root/f'cache/PT_G/{v}/train.index.jsonl').open()] for v in ('v2','v3')}
coverage={}
draws_by_mode={}
for mode in ('PT','SFT'):
    draws=Counter((x['dataset'],x['row_index']) for line in (root/f'{mode}_schedule.jsonl').open() for x in json.loads(line))
    draws_by_mode[mode]=draws
    coverage[mode]={}
    for v,rows in indices.items():
        selected={i:n for (version,i),n in draws.items() if version==v}
        all_families={r['source_family'] for r in rows}
        seen_families={rows[i]['source_family'] for i in selected}
        coverage[mode][v]={'pool_rows':len(rows),'draws':sum(selected.values()),'unique_rows':len(selected),
            'row_coverage':len(selected)/len(rows),'all_families':len(all_families),'seen_families':len(seen_families),
            'max_repeats_per_row':max(selected.values()),'repeated_draws':sum(selected.values())-len(selected),
            'unique_caption_methods':dict(Counter(rows[i]['caption_method'] for i in selected))}
    coverage[mode]['total_unique']=len(draws)
    coverage[mode]['optimizer_steps']=2188
    coverage[mode]['actual_draws']=sum(draws.values())
(out/'sampling_coverage.json').write_text(json.dumps(coverage,indent=2))
# Stable hash-ranked examples, one per category, chosen from rows that the SFT schedule actually uses.
specs={
 'v2_official':lambda v,r:v=='v2' and r['caption_method']=='official_caption_with_verified_structure',
 'v2_recolored':lambda v,r:v=='v2' and r['palette_id'] and r['caption_method']=='official_caption_with_verified_edit_instructions',
 'v2_stretched':lambda v,r:v=='v2' and r['r']>0 and not r['palette_id'] and r['caption_method']=='official_caption_with_verified_edit_instructions',
 'v2_deleted':lambda v,r:v=='v2' and r['deleted_copy_count']>0 and not r['cropped_to_path_budget'],
 'v3_pt_structural':lambda v,r:v=='v3' and r['source_split']=='pt' and not r['cropped_to_path_budget'],
 'v3_cropped':lambda v,r:v=='v3' and r['cropped_to_path_budget'],
}
chosen={}
row_stats={}
for v in ('v2','v3'):
    distinct_captions=set();variants=Counter();families=Counter()
    for i,line in enumerate((root/f'corpus/{v}/train.jsonl').open()):
        r=json.loads(line)
        distinct_captions.add(hashlib.sha256(r['caption'].encode()).digest())
        variants[(r['source_key'],r['condition'],r['r'],r['palette_id'])]+=1
        families[r['source_family']]+=1
        if (v,i) not in draws_by_mode['SFT']:continue
        rank=hashlib.sha256(('audit-20261114:'+r['id']).encode()).hexdigest()
        for name,test in specs.items():
            if test(v,r) and (name not in chosen or rank<chosen[name][0]):
                chosen[name]=(rank,r,v,i)
    row_stats[v]={'distinct_captions':len(distinct_captions),'source_condition_edit_variants':len(variants),
                 'families':len(families),'most_rows_per_family':max(families.values())}
(out/'redundancy.json').write_text(json.dumps(row_stats,indent=2))
live=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
site=out/'site';(site/'models').mkdir(parents=True,exist_ok=True)
html=(live/'index.html').read_text()
pos=html.rfind('</script>')
html=html[:pos]+"""
window.auditState=state;
window.auditRotate=()=>{const offset=camera.position.clone().sub(controls.target);offset.applyAxisAngle(new THREE.Vector3(0,1,0),Math.PI);camera.position.copy(controls.target).add(offset);controls.update();requestRender();};
"""+html[pos:]
(site/'index.html').write_text(html)
shutil.copyfile(live/'colors.ldr',site/'colors.ldr')
vendor=site/'vendor-three-170'
if not vendor.exists():vendor.symlink_to(live/'vendor-three-170',target_is_directory=True)
pack=viewer_pack();part_index=pack.build_index()
manifest=[];audits=[]
for number,(name,(_,r,v,i)) in enumerate(chosen.items(),1):
    p=parse_sample(r['full_dfs_path']);m=parse_sample(r['prompt']+r['completion'])
    assert p.error is None and m.error is None
    inv=Counter(r['full_dfs_path'].splitlines()[j].split(' ',1)[1] for j in [0,*range(1,len(r['full_dfs_path'].splitlines()),2)])
    colors=Counter()
    for s,n in inv.items():colors[s.rsplit(' | ',1)[1]]+=n
    assert sum(inv.values())==r['selected_parts']==len(p.tree.parts)
    assert int(re.search(r'Construct a connected LEGO assembly of (\d+) parts',r['caption'])[1])==len(p.tree.parts)
    for color,n in colors.items():
        assert f'{color} ({n} {"part" if n==1 else "parts"})' in r['caption']
    graph=tree_to_graph(m.tree)
    groups=connected_subsets(r['target_positions'],graph.edges)+connected_subsets(r['evidence_positions'],graph.edges)
    motif_invs=[Counter(m.tree.parts[k] for k in g) for g in groups]
    assert len(groups)==r['target_copy_count']+r['evidence_copy_count']
    assert all(len(g)==r['motif_size'] for g in groups)
    assert all(x==motif_invs[0] for x in motif_invs)
    lines=[s for s in graph_to_ldr(tree_to_graph(p.tree)).splitlines() if s.startswith('1 ')]
    mpd,missing=pack.embed(lines,part_index)
    assert not missing,missing
    file=f'case-{number}.mpd'
    (site/'models'/file).write_text(mpd)
    manifest.append({'file':file,'task':'caption_audit','group':'Caption review','title':name,'n_parts':len(p.tree.parts),
                     'path':r['full_dfs_path'],'caption':r['caption'],'parse_error':None,'collisions':[]})
    audits.append({'number':number,'category':name,'version':v,'row_index':i,'id':r['id'],'source':r['source_key'],
        'family':r['source_family'],'condition':r['condition'],'r':r['r'],'palette_id':r['palette_id'],
        'cropped':r['cropped_to_path_budget'],'before_parts':r['preselection_parts'],'parts':r['selected_parts'],
        'copies':len(groups),'parts_per_copy':r['motif_size'],'colors':dict(colors),'caption':r['caption'],
        'caption_method':r['caption_method'],'scheduled_sft_draws':draws_by_mode['SFT'][(v,i)],
        'mechanical_checks':'total, colors, repeated-copy inventories and counts match','file':file})
write_catalog(site/'models',manifest)
(out/'selected.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
print(json.dumps({'coverage':coverage,'redundancy':row_stats,'selected':[{'number':r['number'],'category':r['category'],'source':r['source'],'parts':r['parts']} for r in audits]},ensure_ascii=False,indent=2))
