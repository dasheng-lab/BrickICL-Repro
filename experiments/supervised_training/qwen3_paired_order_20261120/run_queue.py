import fcntl,json,os,subprocess,sys,time,traceback
from prepare import ROOT,atomic
REPO=ROOT.parents[1]
def main():
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(ROOT/'SUBMITTED.json',{'jobs':['plain','masked'],'state':'submitted','dataset':'exact nzd v2',
        'source_train_rows':70722,'source_dev_rows':2133,'steps_per_arm':2211,
        'initialization':'verified original Qwen3-0.6B, no BrickNet adapters','train_then_eval_each_arm':True,'unix_time':time.time()})
    atomic(ROOT/'queue_status.json',{'state':'preparing_data','pid':os.getpid()})
    subprocess.run([sys.executable,'-u',str(ROOT/'build_data.py')],cwd=REPO,check=True)
    for arm in ('plain','masked'):
        atomic(ROOT/'queue_status.json',{'state':'training','arm':arm,'pid':os.getpid()})
        with (ROOT/f'train.{arm}.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--config',str(ROOT/f'training/{arm}/config.json'),'--resume'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic(ROOT/'queue_status.json',{'state':'evaluating','arm':arm,'pid':os.getpid()})
        with (ROOT/f'eval.{arm}.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'evaluate.py'),arm],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic(ROOT/'COMPLETED.json',{'state':'completed','models':2,'dataset':'nzd v2','unix_time':time.time()})
    atomic(ROOT/'queue_status.json',{'state':'completed'})
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
