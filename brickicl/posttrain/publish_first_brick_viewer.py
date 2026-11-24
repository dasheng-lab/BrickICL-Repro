"""Publish a completed unified run to a local viewer directory, preserving backups."""
import argparse,json,shutil
from pathlib import Path
from brickicl.posttrain.paths import REPO
from brickicl.viz.lazy_catalog import _write_json,write_catalog

def publish(run_dir,viewer_dir,packed_dir):
    run_dir,viewer_dir,packed_dir=map(Path,(run_dir,viewer_dir,packed_dir))
    if not (run_dir/'COMPLETED.json').exists() or not (run_dir/'VIEWER_STAGED.json').exists():
        raise ValueError('Evaluation and packing must complete before publishing')
    entries=json.loads((run_dir/'viewer_entries.json').read_text());board=json.loads((run_dir/'unified_compare.json').read_text())
    if len(entries)!=2808 or len(board['sources'])!=18:raise ValueError('Incomplete unified run')
    models=viewer_dir/'models';manifest_path=models/'manifest.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    backup=run_dir/'viewer_before_unified';backup.mkdir(parents=True,exist_ok=True)
    for rel in ['index.html','unified_compare.html','unified_compare.json','fox_compare.html','fox_compare.json','fox_compare_primed.json','models/manifest.json','models/catalog.json']:
        src=viewer_dir/rel;dst=backup/rel
        if src.exists() and not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for e in entries:
        src=packed_dir/e['file'];dest=models/e['file']
        if not dest.resolve().is_relative_to(models.resolve()) or not src.is_file() or not src.stat().st_size:
            raise ValueError(f'Invalid or missing model: {e["file"]}')
        if src.resolve()!=dest.resolve():
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.publishing');shutil.copy2(src,tmp);tmp.replace(dest)
    keep=[]
    for e in old:
        if e.get('task') not in ('single_model','two_copy','posttrain_icl','from_scratch'):continue
        if e.get('is_primary_generation_test') is False or '额外诊断' in e.get('group','') or e.get('group','').startswith('诊断'):continue
        if e.get('task')=='two_copy' and e.get('evaluation_run')!='compact_double_copy':continue
        keep.append(e)
    merged=entries+keep
    if len(merged)!=len({e['file'] for e in merged}):raise ValueError('Duplicate model files')
    if sum(e.get('task')=='two_copy' for e in merged) not in (0,16):raise ValueError('Double-copy display must contain 0 or 16 samples')
    for task in ('completion_no_head','completion_given_head'):
        if sum(e['task']==task for e in entries)!=1404:raise ValueError('Incomplete prime setting')
    # Source-controlled HTML contains the demand renderer; generated data stays under models/.
    viewer_dir.mkdir(parents=True,exist_ok=True)
    template=REPO/'baseline/repro/viewer'
    for name in ('index.html','unified_compare.html','fox_compare.html'):
        src=template/name;dst=viewer_dir/name
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    colors=packed_dir.parent/'colors.ldr'
    if colors.exists() and colors.resolve()!=(viewer_dir/'colors.ldr').resolve():shutil.copy2(colors,viewer_dir/'colors.ldr')
    _write_json(manifest_path,merged);write_catalog(models,merged)
    _write_json(viewer_dir/'unified_compare.json',board)
    for prime,name in [(0,'fox_compare.json'),(1,'fox_compare_primed.json')]:
        _write_json(viewer_dir/name,{'arms':board['arms'],'model_size':'0.6B','samples':4,**board['sources']['114']['primes'][str(prime)]})
    result={'unified_views':len(entries),'sources_each_prime':18,'double_copy_views':sum(e.get('task')=='two_copy' for e in merged),
            'extra_diagnostics_visible':0,'total_visible_models':len(merged),'old_raw_files_preserved':True}
    _write_json(run_dir/'VIEWER_PUBLISHED.json',result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path('runs/history_first_brick06'))
    p.add_argument('--viewer-dir',type=Path,default=REPO/'baseline/repro/viewer')
    p.add_argument('--packed-dir',type=Path,default=REPO/'baseline/repro/viewer/models')
    a=p.parse_args(argv);print(json.dumps(publish(a.run_dir,a.viewer_dir,a.packed_dir),indent=2))

if __name__=='__main__':main()
