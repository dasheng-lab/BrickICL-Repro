import json,lzma,re
from collections import Counter,defaultdict
from pathlib import Path
from bricknet import load_catalog
from brickicl.sampling.build import read_graphs
from curate_active import SOURCE_FIXES,ROW_FIXES,normalized,digest

ROOT=Path('runs/dfs_70k_20261114')
DATA=Path('baseline/bricknet/data/dataset')
def load_bases():
    saved=ROOT/'screened_bases.json'
    if saved.exists():return {int(k):v for k,v in json.loads(saved.read_text()).items()}
    cat=load_catalog();caps=defaultdict(list);reject=Counter();bases={}
    for line in lzma.open(DATA/'captions_sft.jsonl.xz','rt'):
        r=json.loads(line);caps[int(r['id'])].append(r['caption'].strip())
    words=('black','white','grey','blue','red','green','yellow','orange','brown','pink','purple','tan','aqua','azure','lavender','lime')
    for i,g in enumerate(read_graphs(DATA/'sft.npz.xz')):
        if i not in caps:continue
        if any(c not in cat.code_to_color for c in g.colors):
            reject['unknown_source_color']+=1;continue
        colors=' '.join(normalized(cat.code_to_color[c]) for c in g.colors)
        available={c for c in words if re.search(r'\b'+c+r'\b',colors)}
        options=[]
        for text in caps[i]:
            if not 18<=len(text.split())<=150 or '\n-' in text or '\n*' in text:
                reject['length_or_list']+=1;continue
            clean=normalized(text)
            if re.search(r'\b(possibly|perhaps|might|could|unclear|hard to tell)\b',clean):
                reject['uncertainty']+=1;continue
            claims={c for c in words if re.search(r'\b'+c+r'\b',clean)}
            if claims-available:
                reject['color_conflict']+=1;continue
            if any(f'{s} grey' in clean and f'{s} ' not in colors for s in ('light','dark')):
                reject['shade_conflict']+=1;continue
            options.append(text)
        if options:bases[i]=min(options,key=lambda s:(len(s.split()),digest(s)))
    bases.update(SOURCE_FIXES)
    saved.write_text(json.dumps(bases))
    (ROOT/'caption_screening.json').write_text(json.dumps({'sources_with_caption':len(bases),'rejections':dict(reject)},indent=2))
    return bases

def caption_for(row,bases):
    spec=row['caption'][row['caption'].index('Construct a connected LEGO assembly'):]
    if row['id'] in ROW_FIXES:return ROW_FIXES[row['id']]+'\n'+spec,'manual_two_view_review'
    if row['source_split']!='sft' or row['cropped_to_path_budget'] or row['source'] not in bases:
        return None,None
    base=bases[row['source']]
    if 'Build the following edited version: ' in row['caption']:
        edits=row['caption'].split('Build the following edited version: ',1)[1].split('\nFinal structural specification:',1)[0]
        text='Starting design: '+base+'\nBuild the following edited version: '+edits+'\nFinal structural specification: '+spec
    else:text=base+'\n'+spec
    return text,('manual_source_review' if row['source'] in SOURCE_FIXES else 'official_caption_auto_screened')
