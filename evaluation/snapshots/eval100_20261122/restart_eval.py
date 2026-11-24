import os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];target=b'runs/eval100_20261122/run_eval.py';found=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if p.stat().st_uid==os.getuid() and target in args:found.append(int(p.name))
 except OSError:pass
assert len(found)==1,found
pid=found[0];os.kill(pid,signal.SIGTERM)
for _ in range(60):
 if not Path(f'/proc/{pid}').exists():break
 time.sleep(.5)
else:raise RuntimeError('Own evaluation has not terminated; not starting a duplicate')
with (ROOT/'evaluation.log').open('ab') as f:
 f.write(b'\nRESUME: preserve allocator cache between batches; fixed 35% GPU memory cap unchanged. Existing atomic shards retained.\n')
 proc=subprocess.Popen(['bash',str(ROOT/'run_all.sh')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
(ROOT/'job.pid').write_text(str(proc.pid));print('Resumed cluster job',proc.pid,'from saved shards',flush=True)
