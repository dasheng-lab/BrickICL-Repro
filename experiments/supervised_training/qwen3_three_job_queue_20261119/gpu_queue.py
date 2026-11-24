"""Adopt the active Full-G job; then train/evaluate nzd arms serially without shadowing stdlib queue."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from prepare import ROOT,atomic
REPO=ROOT.parents[1];REG=REPO/'runs/qwen3_fullg_reg_20261119'
def active_pid(file,needle):
    if not file.exists():return False
    try:
        pid=int(file.read_text());cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return needle.encode() in cmd
    except (ValueError,OSError):return False
def state(job,stage):atomic(ROOT/'queue_status.json',{'state':'running','job':job,'stage':stage,'pid':os.getpid(),'updated_unix':time.time()})
def main():
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(ROOT/'SUBMITTED.json',{'submitted_unix':time.time(),'controller_pid':os.getpid(),'mode':'serial GPU queue, persistent on SSH disconnect',
        'jobs':[{'id':'Full_G_reg','training':str(REG),'evaluation':str(REPO/'runs/qwen3_fullg_reg_eval_20261119'),'adopt_existing':True},
                {'id':'plain','training':str(ROOT/'training/plain'),'evaluation':str(REPO/'runs/qwen3_nzd_plain_eval_20261119')},
                {'id':'masked','training':str(ROOT/'training/masked'),'evaluation':str(REPO/'runs/qwen3_nzd_masked_eval_20261119')}],
        'rows_per_job':70000,'steps_per_job':2188,'training_plus_dev_best_evaluation':True})
    # The existing training and its evaluator already own this GPU work; never duplicate them.
    while not (REG/'EVAL_COMPLETED.json').exists():
        if (REG/'FAILED.json').exists() or (REG/'EVAL_FAILED.json').exists():raise RuntimeError('Full-G job or evaluator failed; queue stopped')
        if not (REG/'COMPLETED.json').exists():
            state('Full_G_reg','training')
            if not active_pid(REG/'run.pid','qwen3_fullg_reg_20261119/run.py'):raise RuntimeError('Existing Full-G training runner stopped unexpectedly')
        else:
            state('Full_G_reg','evaluation')
            if not active_pid(REG/'evaluation.pid','qwen3_fullg_reg_20261119/evaluate_after.py'):raise RuntimeError('Existing Full-G evaluator stopped unexpectedly')
        time.sleep(20)
    while not (ROOT/'DATA_READY.json').exists():
        if (ROOT/'DATA_FAILED.json').exists():raise RuntimeError('nzd data preparation failed')
        state('plain','waiting_data');time.sleep(20)
    for arm in ('plain','masked'):
        state(arm,'training')
        with (ROOT/f'train.{arm}.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--config',str(ROOT/f'training/{arm}/config.json'),'--resume'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        state(arm,'evaluation')
        with (ROOT/f'eval.{arm}.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'evaluate.py'),arm],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic(ROOT/'COMPLETED.json',{'state':'completed','jobs':3,'training_runs':[str(REG),str(ROOT/'training/plain'),str(ROOT/'training/masked')],'unix_time':time.time()})
    atomic(ROOT/'queue_status.json',{'state':'completed','updated_unix':time.time()});print('All three jobs completed',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'QUEUE_FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
