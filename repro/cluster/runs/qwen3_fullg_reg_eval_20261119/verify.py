"""Verify copied results and every newly completed checkpoint against its raw records."""
import argparse,math
from statistics import mean
from common import *
from reference_scoring import summarize_completion

def main():
    p=plan();validate_protocol(p);checked=[]
    inputs={prime:{r['id']:r for r in read(ROOT/f'p{prime}_inputs.jsonl')} for prime in (0,1)}
    for spec in p['checkpoints']:
        dest=folder(spec)
        if not (dest/'metrics.json').exists():continue
        m=json.loads((dest/'metrics.json').read_text());assert m['model']==spec
        marker=json.loads((dest/'GENERATED.json').read_text())
        for name,digest in marker['files_sha256'].items():assert sha(dest/name)==digest
        for prime in (0,1):
            raw=read(dest/f'p{prime}.jsonl');scored=read(dest/f'p{prime}.scored.jsonl')
            assert len(raw)==len(scored)==36
            assert {r['id'] for r in raw}==set(inputs[prime])
            for r,s in zip(raw,scored):
                assert r['id']==s['id'] and r['adapters']==spec['adapters']
                row=inputs[prime][r['id']];group=p['protocol']['batch_plan'][r['plan_group']]
                assert r['seed']==group['seed'] and r['max_new_tokens']==group['max_new_tokens']
                assert r['prefix_sha256']==hashlib.sha256(row['prefix'].encode()).hexdigest()
                assert r['condition'] in ('original','stretched') and len(r['samples'])==len(s['samples'])==4
                for a,b in zip(r['samples'],s['samples']):assert a['text']==b['text'] and 0<=b['pose_f1']<=1
            assert summarize_completion(scored)==m['completion'][str(prime)]
        raw=read(dest/'free.jsonl');phys=read(dest/'free.scored.jsonl')
        assert len(raw)==len(phys)==64 and [r['id'] for r in raw]==list(range(64))
        for r,s in zip(raw,phys):
            assert r['adapters']==spec['adapters'] and r['text']==s['full_text']
            assert 0<=s['valid_length']<=100
            assert s['collision_free_length'] is None or 0<=s['collision_free_length']<=s['valid_length']
        known=[r['collision_free_length'] for r in phys if r['collision_free_length'] is not None]
        assert mean(r['valid_length'] for r in phys)==m['physical']['valid_length']
        assert len(known)==m['physical']['collision_coverage']
        assert (mean(known) if known else None)==m['physical']['collision_free_length']
        checked.append(spec['key'])
    result={'checked_curve_points':len(checked),'checked_saved_checkpoints':sum(not k.endswith('_000000') for k in checked),
            'expected_saved_checkpoints':1,'all_completed':len(checked)==1,'checked_keys':checked,'unix_time':time.time()}
    atomic(ROOT/'VERIFIED.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        while not (ROOT/'SCORING_COMPLETED.json').exists():
            if (ROOT/'INFERENCE_FAILED.json').exists() or (ROOT/'SCORING_FAILED.json').exists():
                raise RuntimeError('Evaluation failed; full verification pending.')
            time.sleep(30)
    main()
