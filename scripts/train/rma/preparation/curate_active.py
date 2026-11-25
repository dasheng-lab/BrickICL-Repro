"""Select complete-motif v3 rows and a separately screened caption subset."""
import hashlib,json,lzma,re
from collections import Counter,defaultdict,deque
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from bricknet import load_catalog,parse_sample,tree_to_graph
from brickicl.sampling.build import read_graphs
from brickicl.posttrain.reference_last import connected_subsets

SOURCE=Path('runs/dfs_posttrain_20261114')
OUT=Path('runs/dfs_curated_20261114')
DATA=Path('data/bricknet/dataset')
LIMIT=25000
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(s):
    s=s.lower().replace('gray','grey').replace('-',' ')
    return re.sub(r'\s+',' ',s)
SOURCE_FIXES={
 42776:'An elongated LEGO platform with dark-green sides and rounded ends, raised above a black base by transparent supports. Two parallel rows of black and transparent square elements run along its upper surface.'
}
ROW_FIXES={
 'pt-165433-m0-w4-deleted-r0-p0':'A bright-light-yellow LEGO assembly with curved outer segments arranged around a blue central core and small multicolored round details near the center.',
 'v3-pt-165352-t0-w5-original-r0-p0':'An open black Technic frame with a central box-like section, extended beams, and upright supports. This is a connected fragment of a larger assembly.',
 'v3-pt-77664-t1-w4-original-r0-p0':'A transparent-yellow LEGO form with a rounded, roughly barrel-shaped outline, curved outer surfaces, and flat plates across the top.'
}

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'DATA_READY.json').exists():print('curated data already ready');return
    cat=load_catalog()
    caps=defaultdict(list)
    for line in lzma.open(DATA/'captions_sft.jsonl.xz','rt'):
        r=json.loads(line);caps[int(r['id'])].append(r['caption'].strip())
    # Ground color screening in the unedited source to avoid interpreting an editing instruction as a contradiction.
    color_words=('black','white','grey','blue','red','green','yellow','orange','brown','pink','purple','tan','aqua','azure','lavender','lime')
    rejected=Counter();base_caps={}
    for i,g in enumerate(read_graphs(DATA/'sft.npz.xz')):
        if i not in caps:continue
        if any(c not in cat.code_to_color for c in g.colors):
            rejected['source_has_unknown_color']+=1
            continue
        colors=' '.join(normalized(cat.code_to_color[c]) for c in g.colors)
        options=[]
        for text in caps[i]:
            words=text.split()
            if not 18<=len(words)<=150 or '\n-' in text or '\n*' in text:
                rejected['length_or_list']+=1;continue
            clean=normalized(text)
            if re.search(r'\b(possibly|perhaps|might|could|unclear|hard to tell)\b',clean):
                rejected['uncertain_description']+=1;continue
            claims={c for c in color_words if re.search(r'\b'+c+r'\b',clean)}
            available={c for c in color_words if re.search(r'\b'+c+r'\b',colors)}
            if claims-available:
                rejected['color_conflict']+=1;continue
            if any(f'{shade} grey' in clean and f'{shade} ' not in colors for shade in ('light','dark')):
                rejected['grey_shade_conflict']+=1;continue
            options.append(text)
        if options:base_caps[i]=min(options,key=lambda s:(len(s.split()),digest(s)))
    base_caps.update(SOURCE_FIXES)
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    def reviewed_caption(r):
        spec=r['caption'][r['caption'].index('Construct a connected LEGO assembly'):]
        if r['id'] in ROW_FIXES:
            return ROW_FIXES[r['id']]+'\n'+spec,'manual_two_view_review'
        if r['source_split']!='sft' or r['cropped_to_path_budget'] or r['source'] not in base_caps:
            return None,None
        base=base_caps[r['source']]
        original=r['caption']
        if 'Build the following edited version: ' in original:
            edit=original.split('Build the following edited version: ',1)[1].split('\nFinal structural specification:',1)[0]
            text='Starting design: '+base+'\nBuild the following edited version: '+edit+'\nFinal structural specification: '+spec
        else:text=base+'\n'+spec
        return text,('manual_source_review' if r['source'] in SOURCE_FIXES else 'official_caption_auto_screened')
    # Compact metadata is enough for selection. Preserve all family/condition strata first.
    meta=[]
    for i,line in enumerate((SOURCE/'corpus/v3/train.jsonl').open()):
        r=json.loads(line);caption,kind=reviewed_caption(r)
        key=(r['condition'],r['r'],r['palette_id'],tuple(r['target_copy_indices']),r['selected_parts'])
        meta.append({'i':i,'family':r['source_family'],'variant':key,'rank':digest('active:'+r['id']),
                     'caption':caption is not None,'manual':r['id'] in ROW_FIXES})
    families=defaultdict(lambda:defaultdict(list))
    for r in meta:families[r['family']][r['variant']].append(r)
    queues={}
    for family,variants in families.items():
        buckets=[]
        for key,rs in sorted(variants.items()):
            buckets.append(deque(sorted(rs,key=lambda r:(not r['manual'],not r['caption'],r['rank']))))
        ordered=[]
        while any(buckets):
            for b in buckets:
                if b:ordered.append(b.popleft())
        # Put human-reviewed rows first inside the family, with the rest retaining condition rotation.
        ordered.sort(key=lambda r:not r['manual'])
        queues[family]=deque(ordered)
    selected=set()
    order=sorted(queues,key=lambda f:digest('family:'+f))
    while len(selected)<min(LIMIT,len(meta)):
        for f in order:
            if queues[f]:selected.add(queues[f].popleft()['i'])
            if len(selected)==min(LIMIT,len(meta)):break
    assert {meta[i]['family'] for i in selected}==set(families)
    (OUT/'v3_active_indices.json').write_text(json.dumps(sorted(selected)))
    counts=Counter();quality=Counter();details=defaultdict(Counter);manifest=[];family_sets=defaultdict(set)
    for v in ('v2','v3'):
        for split in ('train','dev','test'):
            dirs={}
            for arm in ('PT_M','PT_G','SFT_M','SFT_G'):
                d=OUT/'cache'/arm/v;d.mkdir(exist_ok=True,parents=True)
                dirs[arm]=d
            old_index={a:[json.loads(s) for s in (SOURCE/f'cache/{a}/{v}/{split}.index.jsonl').open()] for a in ('PT_M','PT_G')}
            old_tokens={a:np.memmap(SOURCE/f'cache/{a}/{v}/{split}.u32',mode='r',dtype='<u4') for a in old_index}
            streams={a:((d/f'{split}.u32').open('wb'),(d/f'{split}.index.jsonl').open('w')) for a,d in dirs.items()}
            offsets=Counter()
            caption_file=OUT/f'captions_{v}_{split}.jsonl'
            with caption_file.open('w') as cf:
                try:
                    for i,line in enumerate((SOURCE/f'corpus/{v}/{split}.jsonl').open()):
                        if v=='v3' and split=='train' and i not in selected:continue
                        r=json.loads(line)
                        assert r['selected_parts']<=100
                        # The subset retains complete M, every reference copy, and all attachment anchors.
                        tree=parse_sample(r['prompt']+r['completion']).tree
                        target=set(r['target_positions']);evidence=set(r['evidence_positions'])
                        assert len(target)==r['k'] and len(evidence)==r['evidence_copy_count']*r['motif_size']
                        groups=connected_subsets(target,tree_to_graph(tree).edges)+connected_subsets(evidence,tree_to_graph(tree).edges)
                        assert len(groups)==r['target_copy_count']+r['evidence_copy_count']
                        invs=[Counter(tree.parts[j] for j in g) for g in groups]
                        assert all(len(g)==r['motif_size'] for g in groups) and all(x==invs[0] for x in invs)
                        for g in groups:
                            members=set(g)
                            anchors=[e.parent if e.child in members else e.child for e in tree.edges
                                     if (e.child in members)!=(e.parent in members)]
                            assert anchors and all(p<r['n_prefix'] and p not in target for p in anchors)
                        items={}
                        for arm in old_index:
                            idx=old_index[arm][i];assert idx['id']==r['id']
                            ids=old_tokens[arm][idx['offset']:idx['offset']+idx['length']].tolist()
                            items[arm]=(ids,idx['prefix_length'],'path_only')
                        cap,kind=reviewed_caption(r)
                        if cap:
                            head=tok.encode(cap,add_special_tokens=False)+tok.encode('\n',add_special_tokens=False)
                            for scope in ('M','G'):
                                ids,n,_=items['PT_'+scope]
                                items['SFT_'+scope]=(head+ids,len(head)+(n if scope=='M' else 0),kind)
                            if any(len(x[0])>6400 for a,x in items.items() if a.startswith('SFT')):
                                for a in ('SFT_M','SFT_G'):del items[a]
                                quality['caption_overlength']+=1
                            else:
                                cf.write(json.dumps({'id':r['id'],'source_key':r['source_key'],'caption':cap,'quality':kind,
                                                     'original_row_index':i,'version':v,'split':split})+'\n')
                                quality[f'{split}/{kind}']+=1
                        else:quality[f'{split}/no_semantic_caption']+=1
                        for arm,(ids,n,kind) in items.items():
                            binary,index=streams[arm]
                            row={k:r[k] for k in ('id','source_key','source_family','condition','k','selected_parts')}
                            row.update(offset=offsets[arm],length=len(ids),prefix_length=n,target_length=len(ids)-n,
                                       caption_method=kind,original_row_index=i)
                            binary.write(np.asarray(ids,dtype='<u4').tobytes());index.write(json.dumps(row)+'\n')
                            offsets[arm]+=len(ids);counts[f'{arm}/{v}/{split}']+=1
                            if split=='train':details[arm][str(r['selected_parts'])]+=1
                            family_sets[(arm,split)].add(r['source_family'])
                        if split=='train':
                            manifest.append({'version':v,'id':r['id'],'original_row_index':i,'family':r['source_family'],
                                             'parts':r['selected_parts'],'condition':r['condition'],'cropped':r['cropped_to_path_budget'],
                                             'has_screened_caption':'SFT_M' in items})
                finally:
                    for b,index in streams.values():b.close();index.close()
            print(v,split,{a:counts[f'{a}/{v}/{split}'] for a in streams},flush=True)
    for arm in streams:
        for a,b in (('train','dev'),('train','test'),('dev','test')):
            assert not family_sets[(arm,a)]&family_sets[(arm,b)]
        cache=OUT/'cache'/arm
        (cache/'READY.json').write_text(json.dumps({'status':'ready','cross_split_overlap':0,
            'artifact_sha256':{str(p.relative_to(cache)):file_sha(p) for p in cache.glob('*/*') if p.is_file()}},indent=2))
    (OUT/'selection_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    report={'status':'ready','counts':dict(counts),'caption_quality':dict(quality),'caption_screening_rejects':dict(rejected),
        'max_parts':100,'v3_active_limit':LIMIT,'v3_families':len(families),'crop_penalty':False,
        'related_subassemblies_checked':'all target/reference inventories, counts, connectivity and anchors preserved',
        'length_histograms':{k:dict(v) for k,v in details.items()},
        'family_counts':{f'{a}/{s}':len(f) for (a,s),f in family_sets.items()},
        'caption_limitations':'automatic language/color screening is not exhaustive visual verification; three exact rows and one source have human two-view revisions',
        'source_ready_sha256':file_sha(SOURCE/'corpus/READY.json')}
    (OUT/'DATA_READY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('counts','caption_quality','v3_families','family_counts')}),flush=True)
if __name__=='__main__':main()
