"""Queue after both existing v2 jobs (including their evaluations); never duplicate them."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from prepare import ROOT,atomic,sha

REPO=ROOT.parents[1];DEPENDENCY=REPO/'runs/qwen3_nzd_v2_20261120'
def main():
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    subprocess.run([sys.executable,str(ROOT/'prepare.py')],check=True)
    ready=json.loads((DEPENDENCY/'DATA_READY.json').read_text())
    assert ready['train_rows']==70722 and ready['dev_rows']==2133
    for arm in ('plain','masked'):
        for split in ('train','dev'):
            record=ready['source_files'][arm][split]
            assert record['matches_bricknet_run'] and sha(Path(record['path']))==record['sha256']
    atomic(ROOT/'SUBMITTED.json',{'state':'queued','order':['nzd v2 plain','nzd v2 masked','Full-G wd .10 / dropout .15'],
        'dependency':str(DEPENDENCY),'dependency_data_sha256':sha(DEPENDENCY/'DATA_READY.json'),
        'controller_pid':os.getpid(),'submitted_unix':time.time(),'training_started':False,
        'natural_eos_evaluation':True,'no_eos_evaluation':True})
    while not (DEPENDENCY/'COMPLETED.json').exists():
        if (DEPENDENCY/'FAILED.json').exists():raise RuntimeError('Existing v2 queue failed; later Full-G training not started')
        pidfile=DEPENDENCY/'queue.pid'
        if pidfile.exists():
            pid=int(pidfile.read_text());proc=Path(f'/proc/{pid}/cmdline')
            if not proc.exists() or b'qwen3_nzd_v2_20261120/run_queue.py' not in proc.read_bytes():
                raise RuntimeError('Existing v2 queue not active; later Full-G training not started')
        atomic(ROOT/'queue_status.json',{'state':'queued_after_nzd_v2','pid':os.getpid(),'updated_unix':time.time()})
        time.sleep(30)
    # The prior queue ends only after both natural-EOS and no-EOS tests are completed.
    assert json.loads((DEPENDENCY/'COMPLETED.json').read_text())['state']=='completed'
    atomic(ROOT/'queue_status.json',{'state':'preflight','pid':os.getpid(),'updated_unix':time.time()})
    if not (ROOT/'SMOKE_PASSED.json').exists():subprocess.run([sys.executable,'-u',str(ROOT/'check.py')],cwd=REPO,check=True)
    atomic(ROOT/'queue_status.json',{'state':'training','pid':os.getpid(),'updated_unix':time.time()})
    subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--config',str(ROOT/'training/Full_G_reg/config.json'),'--resume'],cwd=REPO,check=True)
    atomic(ROOT/'COMPLETED.json',json.loads((ROOT/'training/Full_G_reg/COMPLETED.json').read_text()))
    atomic(ROOT/'queue_status.json',{'state':'evaluating','pid':os.getpid(),'updated_unix':time.time()})
    subprocess.run([sys.executable,'-u',str(ROOT/'evaluate_after.py')],cwd=REPO,check=True)
    atomic(ROOT/'queue_status.json',{'state':'completed','updated_unix':time.time()})
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
