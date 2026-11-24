import importlib.util,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from score_results import read,score_row,summarize_completion,ROOT
path=Path('/home/nzd0420/brickICL/nzd-supertry/step11_report_metrics/first_brick.py')
spec=importlib.util.spec_from_file_location('published_first_brick',path)
published=importlib.util.module_from_spec(spec);spec.loader.exec_module(published)
checks=[]
with ProcessPoolExecutor(max_workers=4) as pool:
    for arm,p in [('frozen_pt',0),('frozen_pt',1),('frozen_sft',1)]:
        rows={r['id']:r for r in read(ROOT/f'p{p}_inputs.jsonl')}
        records=[r for r in read(Path('runs/history_first_brick06')/f'p{p}.{arm}.jsonl') if r['id'] in rows]
        expected=published.score(rows,published.references(rows,p),records,truncate=True)
        ours=summarize_completion(list(pool.map(score_row,[(rows[r['id']],r) for r in records])))
        assert abs(expected['delta']-ours['delta'])<1e-12
        assert all(abs(expected['f1'][c]-ours['f1'][c])<1e-12 for c in ('original','stretched'))
        checks.append({'arm':arm,'prime':p,'delta':ours['delta'],'f1':ours['f1']})
(ROOT/'SCORER_VALIDATED.json').write_text(json.dumps({'matches_published_implementation':True,'checks':checks},indent=2))
print(json.dumps(checks))
