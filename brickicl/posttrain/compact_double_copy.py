"""Remove extra-diagnostic displays and keep four fixed double-copy cases, four samples each."""
import json,shutil
from pathlib import Path
from brickicl.viz import pack_probe as pp
from brickicl.viz.lazy_catalog import _write_json,write_catalog
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer-dir',type=Path,default=pp.VIEWER)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/history_first_brick06'))
    parser.add_argument('--legacy-eval',type=Path,default=Path('runs/posttrain_eval06'))
    args = parser.parse_args(argv)
    LIVE=args.viewer_dir
    OUT=args.run_dir;backup=OUT/'viewer_before_cleanup';backup.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','catalog.json']:
        if (LIVE/'models'/name).exists() and not (backup/name).exists():shutil.copy2(LIVE/'models'/name,backup/name)
    def read(p):return [json.loads(s) for s in p.open()]
    oldroot=args.legacy_eval
    inputs={r['source_key']:r for r in read(oldroot/'legacy_triads.jsonl') if r['condition']=='stretched'}
    results={r['source_key']:r for r in read(oldroot/'legacy.frozen_pt_double.jsonl') if r['condition']=='stretched'}
    selected=[(114,'狐狸'),(458,'出租车'),(15,'桌子／拱门'),(48,'电线杆')]
    packer=pp.Packer()
    for source,title in selected:
        key=f'val:{source}';r=inputs[key];out=results[key];assert len(out['samples'])==4
        for i,s in enumerate(out['samples']):
            packer.add_pair(f'compact/source_{source}_s{i+1}.mpd',r['demo'],r['prefix']+s['text'],r['n_prefix'],
                group=f'双模型 · #{source} {title} · 0.6B · 加长',title=f'官方 PT 双模型示范 · 样本 {i+1}/4',caption=title,
                note='四个来源预先固定；只展示加长条件的前4次实际采样，不按效果挑选。左侧是完整示范，右侧是待补模型；权重为官方PT，输入预给目标第一块。'
                     f' 位姿F1={s["pose_f1"]:.3f}；整段解析={s["parse_ok"]}；EOS={s["eos_emitted"]}。')
            e=packer.entries[-1];offset=e['demo_parts'];e['collisions']=None if s['collisions'] is None else [j+offset for j in s['collisions']]
            e.update(task='two_copy',source_key=key,selected_sample=i,evaluation_run='compact_double_copy',is_primary_generation_test=True)
    assert len(packer.entries)==16
    for e in packer.entries:
        src=pp.VIEWER/'models'/e['file'];dest=LIVE/'models'/e['file'];dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.new');shutil.copy2(src,temp);temp.replace(dest)
    old=json.loads((LIVE/'models/manifest.json').read_text()) if (LIVE/'models/manifest.json').exists() else [];kept=[];removed=0
    newer_primary=any(e.get('task')=='from_scratch' and e['file'].startswith('fullg_generation/') for e in old)
    for e in old:
        group=e.get('group','')
        extra=e.get('is_primary_generation_test') is False or '额外诊断' in group or group.startswith('诊断（非主测试）')
        duplicate_zero=newer_primary and e.get('task')=='from_scratch' and not e['file'].startswith('fullg_generation/')
        if e.get('task')=='two_copy' or extra or duplicate_zero:removed+=1;continue
        kept.append(e)
    merged=packer.entries+kept
    assert len({e['file'] for e in merged})==len(merged)
    assert sum(e.get('task')=='two_copy' for e in merged)==16
    assert all(e.get('is_primary_generation_test') is not False and '额外诊断' not in e.get('group','') for e in merged)
    _write_json(LIVE/'models/manifest.json',merged);write_catalog(LIVE/'models',merged)
    result={'double_copy_sources':[s for s,_ in selected],'double_copy_views':16,'extra_diagnostic_views_visible':0,
            'removed_or_replaced_views':removed,'total_visible_views':len(merged),'raw_files_preserved':True}
    (OUT/'viewer_cleanup.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
