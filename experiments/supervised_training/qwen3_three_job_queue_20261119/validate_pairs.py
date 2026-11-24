"""Check frozen subset identity, train/dev separation and sampled 3D equivalence."""
import json,random
from pathlib import Path
import numpy as np
from bricknet import parse_sample,tree_to_graph,decode_graph
from prepare import ROOT,atomic,sha

def main():
    ready=json.loads((ROOT/'DATA_READY.json').read_text());selection=[json.loads(s) for s in (ROOT/'selection.jsonl').open()]
    assert ready['rows_per_arm']==len(selection)==70000
    inds={arm:{split:[json.loads(s) for v in ('v2','v3') for s in (ROOT/f'cache/{arm}/{v}/{split}.index.jsonl').open()] for split in ('train','dev')} for arm in ('plain','masked')}
    assert len(inds['plain']['dev'])==len(inds['masked']['dev'])==256
    sources={r['source_key'] for r in inds['plain']['train']};held={r['source_key'] for r in inds['plain']['dev']}
    assert len(sources)==70000 and not sources&held
    for split in ('train','dev'):
        p=inds['plain'][split];m=inds['masked'][split]
        assert [(r['id'],r['source_key'],r['nodes'],r['origin']) for r in p]==[(r['id'],r['source_key'],r['nodes'],r['origin']) for r in m]
        assert all(r['prefix_length']==1 and r['length']<=6400 and r['nodes']<=100 for r in p+m)
    schedules=[json.loads(s) for s in (ROOT/'schedule.jsonl').open()]
    refs=[(r['dataset'],r['row_index']) for b in schedules for r in b]
    assert len(refs)==len(set(refs))==70000
    # Known-family audit uses the existing structural-source fingerprints; unknown families are not claimed checked.
    known={}
    old=ROOT.parent/'structural_20261114'
    for v in ('v2','v3'):
        for split in ('train','dev','test'):
            for s in (old/f'cache/PT_G/{v}/{split}.index.jsonl').open():
                r=json.loads(s);known[r['source_key']]=r['source_family']
    nzd=Path('/home/nzd0420/brickICL/runs/nzd-supertry')
    all_dev={f'{r["split"]}:{r["source"]}' for r in map(json.loads,(nzd/'dev_v4/plain.jsonl').open())}
    overlap={known[s] for s in sources if s in known}&{known[s] for s in all_dev if s in known}
    checks=[];rng=random.Random(19)
    picked=rng.sample([r for r in selection if r['origin']=='generated_pair'],64)
    with (nzd/'train_v4_mix/plain.jsonl').open('rb') as pf,(nzd/'train_v4_mix/masked.jsonl').open('rb') as mf:
        for r in picked:
            pf.seek(r['plain_offset']);mf.seek(r['masked_offset']);p=json.loads(pf.readline());m=json.loads(mf.readline())
            pt=parse_sample(p['path']);mt=parse_sample(m['path']);assert pt.error is None and mt.error is None
            assert len(pt.tree.parts)==len(mt.tree.parts)==p['nodes']
            pm=np.asarray(decode_graph(tree_to_graph(pt.tree)));mm=np.asarray(decode_graph(tree_to_graph(mt.tree)))
            pidx={node:i for i,node in enumerate(p['order'])};midx={node:i for i,node in enumerate(m['order'])}
            assert pidx.keys()==midx.keys()
            for node,i in pidx.items():
                j=midx[node];assert pt.tree.parts[i].part_id==mt.tree.parts[j].part_id
                assert np.allclose(pm[i],mm[j],atol=1e-4,rtol=0),(r['source_key'],node)
            checks.append(r['line'])
    result={'rows_per_arm':70000,'unique_sources':70000,'dev_pairs':256,'train_dev_source_overlap':0,
            'paired_schedule_no_replacement':True,'geometry_pairs_verified':len(checks),'geometry_source_lines':checks,
            'known_family_overlap_count':len(overlap),'known_family_overlap':sorted(overlap),
            'data_ready_sha256':sha(ROOT/'DATA_READY.json')}
    atomic(ROOT/'DATA_VALIDATED.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
