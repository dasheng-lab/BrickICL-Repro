import fcntl,json,time,traceback
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from common import ROOT,read,sha,atomic,jsonl
from reference_scoring import physical_row

def main():
    lock=(ROOT/'scoring.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text());assert sha(ROOT/'reference_scoring.py')==plan['scoring_sha256']
    with ProcessPoolExecutor(max_workers=4) as pool:
        while True:
            pending=0
            for spec in plan['models']:
                dest=ROOT/'results'/spec['key']
                if (dest/'metrics.json').exists():continue
                pending+=1
                if not (dest/'GENERATED.json').exists():continue
                marker=json.loads((dest/'GENERATED.json').read_text());assert sha(dest/'free_no_eos.jsonl')==marker['raw_sha256']
                raw=read(dest/'free_no_eos.jsonl');assert len(raw)==64 and [r['id'] for r in raw]==list(range(64))
                for r in raw:
                    assert r['adapters']==spec['adapters'] and r['tokens']==len(r['generated_token_ids'])==4096
                    assert not set(r['forbidden_eos_ids'])&set(r['generated_token_ids'])
                scored=list(pool.map(physical_row,raw,chunksize=4))
                for r in scored:
                    assert 0<=r['valid_length']<=100
                    assert r['collision_free_length'] is None or 0<=r['collision_free_length']<=r['valid_length']
                    r.pop('generated_token_ids',None)
                known=[r['collision_free_length'] for r in scored if r['collision_free_length'] is not None]
                m={'model':spec,'samples':64,'no_eos_valid_length':mean(r['valid_length'] for r in scored),
                    'no_eos_collision_free_length':mean(known) if known else None,'collision_coverage':len(known),
                    'parse_rate':mean(r['invalid'] is None for r in scored),'eos_suppression_verified':True}
                jsonl(dest/'free_no_eos.scored.jsonl',scored);atomic(dest/'metrics.json',m);print('SCORED',spec['key'],m['no_eos_valid_length'],m['no_eos_collision_free_length'],flush=True)
            if not pending:break
            if (ROOT/'INFERENCE_FAILED.json').exists():raise RuntimeError('No-EOS inference failed')
            time.sleep(15)
    atomic(ROOT/'SCORING_COMPLETED.json',{'models':1,'samples':64,'state':'completed'})
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'SCORING_FAILED.json',{'traceback':traceback.format_exc()});raise
