import fcntl,json,os,subprocess,sys,time,traceback
from common import *

def main():
    lock=(ROOT/'delivery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'VERIFIED.json').exists() or not json.loads((ROOT/'VERIFIED.json').read_text())['all_completed']:
        if (ROOT/'INFERENCE_FAILED.json').exists() or (ROOT/'SCORING_FAILED.json').exists():raise RuntimeError('Evaluation failed')
        time.sleep(15)
    browser_python=str(REPO/'runs/proposal_report_20261111/env/bin/python')
    for script,python in [('report.py',sys.executable),('pack_viewer.py',sys.executable),('stage_app.py',sys.executable),
                          ('verify_viewer.py',browser_python),('publish.py',sys.executable),('verify_live.py',sys.executable)]:
        atomic(ROOT/'delivery_status.json',{'state':'running','step':script,'pid':os.getpid(),'unix_time':time.time()})
        print('DELIVERY',script,flush=True)
        subprocess.run([python,'-u',str(ROOT/script)],cwd=REPO,check=True)
    atomic(ROOT/'delivery_status.json',{'state':'completed','unix_time':time.time()})
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'DELIVERY_FAILED.json',{'traceback':traceback.format_exc(),'unix_time':time.time()});raise
