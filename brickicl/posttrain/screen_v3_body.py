"""Exclude nominally complex bodies dominated by repeated paving/base bricks."""
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
from bricknet import parse_sample
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('runs/posttrain_icl_v3_local'))
    args = parser.parse_args(argv)
    root=args.data
    metadata=json.loads((root/'metadata.json').read_text())
    if metadata.get('body_dominance_screen'):raise RuntimeError('already screened')
    counts,removed,totals=Counter(),Counter(),Counter()
    sources,families=defaultdict(set),defaultdict(set)
    for split in ('train','dev','test'):
        path=root/f'{split}.jsonl';temp=path.with_suffix('.body.tmp')
        with path.open() as stream,temp.open('w') as output:
            for line in stream:
                row=json.loads(line)
                parsed=parse_sample(row['prompt']+row['completion'])
                assert parsed.error is None
                motif=set(row['evidence_positions'])|set(row['target_positions'])
                body=[p.part_id for i,p in enumerate(parsed.tree.parts) if i not in motif]
                assert len(body)==row['body_parts']
                maximum=max(Counter(body).values())/len(body)
                if maximum>.45:
                    removed[split]+=1;continue
                row['body_max_part_type_fraction']=maximum
                output.write(json.dumps(row,ensure_ascii=False)+'\n')
                counts[split]+=1;counts[f'{split}:{row["condition"]}']+=1
                sources[split].add(row['source_key']);families[split].add(row['source_family'])
                totals[f'{split}:supervised_tokens']+=row['target_tokens_with_eos']
        os.replace(temp,path)
    metadata['counts_before_body_screen']=metadata['counts']
    metadata.update(counts=dict(counts),source_counts={k:len(v) for k,v in sources.items()},
                    family_counts={k:len(v) for k,v in families.items()},token_totals=dict(totals),
                    body_dominance_screen={'threshold':.45,'removed':dict(removed)})
    metadata['quality_gates']['max_body_part_type_fraction']=.45
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'sources':metadata['source_counts'],'removed':dict(removed)},indent=2),flush=True)

if __name__ == '__main__':
    main()
