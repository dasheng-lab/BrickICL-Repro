import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,REPO,atomic

def main():
    lock=(ROOT/'finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'SCORING_COMPLETED.json').exists():
        if (ROOT/'INFERENCE_FAILED.json').exists() or (ROOT/'SCORING_FAILED.json').exists():raise RuntimeError('No-EOS evaluation failed')
        time.sleep(15)
    subprocess.run([sys.executable,'-u',str(ROOT/'publish.py')],cwd=REPO,check=True)
    paused=json.loads((ROOT/'PAUSED_TRAINING.json').read_text());pid=paused['pid'];proc=Path(f'/proc/{pid}')
    if proc.exists():
        assert proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]==paused['proc_start_ticks']
        assert 'qwen3_nzd_v2_20261120/train.py' in proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
        os.kill(pid,signal.SIGCONT)
        paused.update(state='resumed_after_no_eos_evaluation',resumed_unix=time.time())
        atomic(ROOT/'PAUSED_TRAINING.json',paused);atomic(REPO/'runs/qwen3_nzd_v2_20261120/PRIORITY_PAUSE.json',paused)
    else:raise RuntimeError('Paused v2 trainer exited; inspect before resuming queue')
    atomic(ROOT/'COMPLETED.json',{'state':'completed','models':12,'samples':768,'published':True,'v2_training_resumed':True,'new_regularization_training_started':False})
    print('No-EOS evaluation and Gradio complete; original v2 training resumed.',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'FINISH_FAILED.json',{'traceback':traceback.format_exc()});raise
