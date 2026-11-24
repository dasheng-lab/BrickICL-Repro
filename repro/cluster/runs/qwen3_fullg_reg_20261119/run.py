import fcntl,json,os,subprocess,sys,time,traceback
from prepare import ROOT,atomic

def main():
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    subprocess.run([sys.executable,str(ROOT/'prepare.py')],check=True)
    if not (ROOT/'SMOKE_PASSED.json').exists():subprocess.run([sys.executable,str(ROOT/'check.py')],check=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--config',str(ROOT/'training/Full_G_reg/config.json'),'--resume'],check=True)
    atomic(ROOT/'COMPLETED.json',json.loads((ROOT/'training/Full_G_reg/COMPLETED.json').read_text()))
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
