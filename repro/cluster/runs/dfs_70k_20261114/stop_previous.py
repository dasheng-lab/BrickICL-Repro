import json,os,signal,time
from pathlib import Path
root=Path('/home/sunshq114514/brickICL/runs/dfs_curated_20261114')
def owned(pid,needle):
    p=Path(f'/proc/{pid}')
    return p.exists() and p.stat().st_uid==os.getuid() and needle in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
queue=1355368
if owned(queue,'bash runs/dfs_posttrain_20261114/run_curated.sh'):
    os.kill(queue,signal.SIGTERM)
for _ in range(900):
    latest=root/'training/PT_M/latest.json'
    if latest.exists():
        saved=json.loads(latest.read_text())
        cp=Path(saved['path'])
        if saved['optimizer_step']>=250 and (cp/'training_state.pt').is_file() and (cp/'provenance.json').is_file():
            break
    time.sleep(2)
else:raise RuntimeError('No complete checkpoint yet; old queue stopped, training left running')
pid=1355411
if owned(pid,'python -u runs/dfs_curated_20261114/train.py'):
    os.kill(pid,signal.SIGTERM)
for _ in range(60):
    if not Path(f'/proc/{pid}').exists():break
    time.sleep(.5)
else:raise RuntimeError('Training process did not exit')
record={'state':'superseded_after_checkpoint','checkpoint':saved,
        'reason':'User requested fewer v2, more v3, and approximately 70000 unique training records per arm',
        'old_queue_stopped':not owned(queue,'run_curated.sh'),'training_pid_exited':True}
(root/'SUPERSEDED.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
