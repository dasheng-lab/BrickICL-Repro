"""Build separate PT whole-G and caption-only SFT whole-G caches; preserve old experiments."""
import hashlib
import json
import lzma
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from bricknet import parse_sample
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('runs/reference_last_06'))
    parser.add_argument('--out',type=Path,default=Path('runs/fullg_06'))
    args = parser.parse_args(argv)
    ROOT=args.out;SOURCE=args.data
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'READY.json').exists():raise RuntimeError('full-G data already exists; do not overwrite')
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    caps=defaultdict(list)
    for line in lzma.open('baseline/bricknet/data/dataset/captions_sft.jsonl.xz','rt'):
        r=json.loads(line);caps[int(r['id'])].append(r['caption'].strip())
    nl=tok.encode('\n',add_special_tokens=False);assert len(nl)==1
    counts=Counter();tokens=Counter();sources=defaultdict(set);families=defaultdict(set);skips=Counter()

    def caption(row):
        choices=caps[row['source']]
        base=choices[row['sampling_seed']%len(choices)]
        edits=[]
        if row.get('r',0):
            edits.append(f'Lengthen the matching repeated subassemblies by inserting {row["r"]} additional copies of the stacked element into each subassembly. Keep the rest of the body unchanged.')
        if row.get('deleted_copy_count',0):
            remaining=row['evidence_copy_count']+row['target_copy_count']
            edits.append(f'Remove {row["deleted_copy_count"]} of those repeated subassemblies completely, leaving {remaining} matching copies.')
        if row.get('palette_id',0):
            # State the actual edited part/color inventory of one target copy, not an invented color name.
            lines=row['completion'].splitlines()
            inventory=Counter(line.split(' ',1)[1] for line in lines[0::2])
            divisor=row['target_copy_count']
            assert all(n%divisor==0 for n in inventory.values())
            specification='; '.join(f'{n//divisor} x {name}' for name,n in sorted(inventory.items()))
            edits.append('Recolor the matching repeated subassemblies consistently. Each copy has the following part and color inventory after the edits: '+specification+'. These colors override the initial description only for those repeated subassemblies.')
        if not edits:return base,'author_caption'
        return ('Start from this initial LEGO design: '+base+'\nCreate a modified complete model with these changes: '+' '.join(edits)), 'author_caption_with_explicit_edits'

    for dataset in ['v2','v3']:
        for split in ['train','dev','test']:
            rows=[json.loads(s) for s in (SOURCE/dataset/f'{split}.jsonl').open()]
            for mode in ['PT','SFT']:
                dest=ROOT/mode/dataset;dest.mkdir(parents=True,exist_ok=True)
                key=f'{mode}/{dataset}/{split}';offset=0;seen=set()
                with (dest/f'{split}.u32').open('wb') as binary,(dest/f'{split}.index.jsonl').open('w') as index,(dest/f'{split}.jsonl').open('w') as human:
                    batch=[]
                    def write_batch(batch):
                        global offset
                        if not batch:return
                        all_g=tok([g for row,g,p,kind in batch],add_special_tokens=False)['input_ids']
                        prompts=tok([p for row,g,p,kind in batch],add_special_tokens=False)['input_ids']
                        for (row,g,p,kind),gt,pt in zip(batch,all_g,prompts):
                            if mode=='PT':
                                assert p=='a' and gt[:len(pt)]==pt
                                ids=gt+[tok.eos_token_id];prefix=len(pt)
                            else:
                                prefix=len(pt)+1;ids=pt+nl+gt+[tok.eos_token_id]
                            if len(ids)>6400:skips[key+':overlength']+=1;continue
                            assert ids[-1]==tok.eos_token_id and prefix<len(ids)
                            digest=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest()
                            if digest in seen:skips[key+':duplicate']+=1;continue
                            seen.add(digest)
                            binary.write(np.asarray(ids,dtype='<u4').tobytes())
                            record={k:row.get(k) for k in ['id','source_key','source_family','condition','layout','k']}
                            record.update(offset=offset,length=len(ids),prefix_length=prefix,target_length=len(ids)-prefix,
                                          full_graph_parts=row['n_prefix']+row['k'],prompt_kind=kind,loss_scope='whole_G_plus_EOS')
                            index.write(json.dumps(record)+'\n')
                            human.write(json.dumps(record|{'prompt':p,'path':g,'original_missing_parts':row['k']})+'\n')
                            offset+=len(ids);counts[key]+=1;tokens[key+':all']+=len(ids);tokens[key+':supervised']+=len(ids)-prefix
                            sources[f'{mode}/{split}'].add(row['source_key']);families[f'{mode}/{split}'].add(row['source_family'])
                    for row in rows:
                        if mode=='SFT' and (row['source_split']!='sft' or row['source'] not in caps):
                            skips[key+':no_author_caption']+=1;continue
                        g=row['prompt']+row['completion'];assert g.startswith('a ')
                        parsed=parse_sample(g);assert parsed.error is None and len(parsed.tree.parts)==row['n_prefix']+row['k']
                        p,kind=('a','node_identifier_only') if mode=='PT' else caption(row)
                        batch.append((row,g,p,kind))
                        if len(batch)==128:write_batch(batch);batch=[]
                    write_batch(batch)
                print(key,counts[key],tokens[key+':supervised'],'supervised tokens',flush=True)
    for mode in ['PT','SFT']:
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            assert not sources[f'{mode}/{a}']&sources[f'{mode}/{b}']
            assert not families[f'{mode}/{a}']&families[f'{mode}/{b}']
        checksums={str(p.relative_to(ROOT/mode)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/mode).glob('*/*') if p.is_file()}
        report={'status':'ready','mode':mode,'cross_split_overlap':0,'artifact_sha256':checksums,
                'counts':{k:v for k,v in counts.items() if k.startswith(mode+'/')},'tokens':{k:v for k,v in tokens.items() if k.startswith(mode+'/')},
                'source_counts':{k:len(v) for k,v in sources.items() if k.startswith(mode+'/')},
                'objective':'all G tokens plus EOS; PT supplies a only, SFT supplies only author caption with explicit edit instructions if needed',
                'source_corpus_ready_sha256':hashlib.sha256((SOURCE/'READY.json').read_bytes()).hexdigest(),
                'captions':'only official SFT-pool captions; no fabricated captions for PT-pool sources',
                'limitations':'caption edits are rule-based instructions, not manually verified semantic captions; PT/SFT data pools differ due caption availability'}
        (ROOT/mode/'READY.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'READY.json').write_text(json.dumps({'status':'ready','counts':dict(counts),'tokens':dict(tokens),'skips':dict(skips),
        'source_counts':{k:len(v) for k,v in sources.items()},'cross_split_overlap':0},indent=2)+'\n')

if __name__ == '__main__':
    main()
