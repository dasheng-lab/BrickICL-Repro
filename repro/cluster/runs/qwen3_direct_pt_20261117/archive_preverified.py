import json,shutil
from prepare import ROOT,atomic

archive=ROOT/'pre_verified_run'
if archive.exists():raise SystemExit('Archive already exists; no repeat changes.')
status_path=ROOT/'training/M_only/status.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {}
pid=status.get('pid')
if pid and __import__('pathlib').Path(f'/proc/{pid}/cmdline').exists():
    cmd=__import__('pathlib').Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'qwen3_direct_pt_20261117/train.py' not in cmd,'Training still active'
archive.mkdir()
for relative in ('training/M_only','SMOKE_PASSED.json','queue.log','QUEUE_STATUS.json','QUEUE_FAILED.json',
                 'SHARED_BASE_MISMATCH.json','BASE_DOWNLOAD_FAILED.json','base_download.log'):
    source=ROOT/relative
    if source.exists():shutil.move(str(source),str(archive/source.name))
atomic(archive/'ARCHIVE_REASON.json',{'reason':'Short run paused during base audit; official metadata later verified the base as correct. Restart for clean isolated-base provenance.',
                                  'last_status':status,'not_used_for_final_training':True})
print('Archived pre-verification run at',archive)
