import fcntl,json,os,time,traceback
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from common import *
from reference_scoring import score_row,physical_row,summarize_completion

def main():
    lock=(ROOT/'scoring.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=plan();validate_protocol(p)
    inputs={prime:{r['id']:r for r in read(ROOT/f'p{prime}_inputs.jsonl')} for prime in (0,1)}
    with ProcessPoolExecutor(max_workers=6) as pool:
        while True:
            remaining=[]
            for spec in p['checkpoints']:
                dest=folder(spec)
                if (dest/'metrics.json').exists():continue
                remaining.append(spec)
                if not (dest/'GENERATED.json').exists():continue
                marker=json.loads((dest/'GENERATED.json').read_text())
                for name,digest in marker['files_sha256'].items():assert sha(dest/name)==digest
                atomic(ROOT/'scoring_status.json',{'state':'running','key':spec['key'],'updated_unix':time.time()})
                result={'model':spec,'completion':{}}
                for prime in (0,1):
                    recs=read(dest/f'p{prime}.jsonl')
                    assert len(recs)==36 and {r['id'] for r in recs}==set(inputs[prime])
                    scored=list(pool.map(score_row,[(inputs[prime][r['id']],r) for r in recs],chunksize=1))
                    jsonl(dest/f'p{prime}.scored.jsonl',scored);result['completion'][str(prime)]=summarize_completion(scored)
                recs=read(dest/'free.jsonl');assert len(recs)==64
                phys=list(pool.map(physical_row,recs,chunksize=4));jsonl(dest/'free.scored.jsonl',phys)
                known=[r['collision_free_length'] for r in phys if r['collision_free_length'] is not None]
                result['physical']={'samples':64,'valid_length':mean(r['valid_length'] for r in phys),
                    'collision_free_length':mean(known) if known else None,'collision_coverage':len(known),
                    'parse_rate':mean(r['invalid'] is None for r in phys)}
                result['scored_unix']=time.time();atomic(dest/'metrics.json',result)
                print('SCORED',spec['key'],result['completion']['1']['f1'],result['physical'],flush=True)
            if not remaining:break
            if (ROOT/'INFERENCE_FAILED.json').exists():
                raise RuntimeError('Inference failed; see INFERENCE_FAILED.json. Scoring can be resumed.')
            time.sleep(15)
    atomic(ROOT/'SCORING_COMPLETED.json',{'state':'completed','models':3,'unix_time':time.time()})
    atomic(ROOT/'scoring_status.json',{'state':'completed','updated_unix':time.time()})

if __name__=='__main__':
    try:main()
    except Exception:
        atomic(ROOT/'SCORING_FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
