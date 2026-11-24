"""Update counts after pure row filtering; token/EOS checks were run on the superset."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('runs/posttrain_icl_v3_local'))
    args = parser.parse_args(argv)
    root=args.data
    p=root/'training_profile.json'
    profile=json.loads(p.read_text())
    assert profile['loss_mask_boundary_eos_padding_checks'].startswith('passed')
    rows=[json.loads(s) for s in (root/'train.jsonl').open()]
    assert all('body_max_part_type_fraction' in r for r in rows)
    target=[r['target_tokens_with_eos'] for r in rows]
    context=[r['tokens_with_eos']-r['target_tokens_with_eos'] for r in rows]
    profile.update(rows=len(rows),sources=len({r['source_key'] for r in rows}),
        conditions=dict(Counter(r['condition'] for r in rows)),tokens_per_epoch_including_context=sum(context)+sum(target),
        supervised_tokens_per_epoch_including_eos=sum(target),
        prompt_tokens_p50_p95_max=np.quantile(context,[.5,.95,1]).tolist(),
        target_tokens_p50_p95_max=np.quantile(target,[.5,.95,1]).tolist(),
        target_parts_min_p50_p95_max=np.quantile([r['k'] for r in rows],[0,.5,.95,1]).tolist(),
        supervised_fraction_of_all_tokens=sum(target)/(sum(target)+sum(context)),
        loss_mask_boundary_eos_padding_checks='passed on all rows before a pure row-removal body-quality screen; retained rows unchanged')
    p.write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(profile,indent=2),flush=True)

if __name__ == '__main__':
    main()
