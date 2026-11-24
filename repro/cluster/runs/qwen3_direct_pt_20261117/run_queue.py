import fcntl,json,os,subprocess,sys,time,traceback
from prepare import ROOT,atomic

def main():
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'BASE_PROVENANCE.json').read_text())['verified']
    subprocess.run([sys.executable,str(ROOT/'prepare.py')],check=True)
    if not (ROOT/'SMOKE_PASSED.json').exists():subprocess.run([sys.executable,str(ROOT/'smoke.py')],check=True)
    for arm in ('M_only','Full_G','BrickNet'):
        while arm=='BrickNet' and not (ROOT/'OFFICIAL_READY.json').exists():
            if (ROOT/'OFFICIAL_FAILED.json').exists():raise RuntimeError('Official preprocessing failed; see OFFICIAL_FAILED.json')
            atomic(ROOT/'QUEUE_STATUS.json',{'state':'waiting_official_data','pid':os.getpid()});time.sleep(30)
        atomic(ROOT/'QUEUE_STATUS.json',{'state':'training','arm':arm,'pid':os.getpid()})
        subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--config',str(ROOT/f'training/{arm}/config.json'),'--resume'],check=True)
    result={a:json.loads((ROOT/f'training/{a}/COMPLETED.json').read_text()) for a in ('M_only','Full_G','BrickNet')}
    atomic(ROOT/'COMPLETED.json',result);atomic(ROOT/'QUEUE_STATUS.json',{'state':'completed'})
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'QUEUE_FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
