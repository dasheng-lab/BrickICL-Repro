"""Publish complete results with backups and restore the previous deployment if verification fails."""
import hashlib,json,os,shutil,signal,subprocess,tarfile,time,urllib.request
from collections import Counter
from pathlib import Path
from brickicl.viz.lazy_catalog import _write_json,write_catalog

ROOT=Path('runs/structural_eval_20261115').resolve()
LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
SERVICE=Path('/home/sunshq114514/brickicl-gradio-service')
PLAY=SERVICE/'brickicl/playground'
STAGE=ROOT/'viewer_stage'
URL='http://127.0.0.1:17860'
APP_FILES=('app.py','backend.py','evaluated_models.py','results_summary.md')
BOARDS=('unified_compare.html','unified_compare.json','nzd_order_compare.json','rollback_compare.html','rollback_conditions.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(route):
    with urllib.request.urlopen(URL+route,timeout=5) as f:return json.load(f)
def replace_file(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.structural-tmp')
    shutil.copyfile(src,tmp);tmp.replace(dest)
def process():
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if p.stat().st_uid==os.getuid() and b'brickicl.playground.app' in args and b'--port' in args and args[args.index(b'--port')+1]==b'17860':
                found.append(int(p.name))
        except (OSError,ValueError,IndexError):pass
    assert len(found)==1,found
    return found[0]
def start():
    with (SERVICE/'logs/playground.structural.log').open('ab') as log:
        return subprocess.Popen(['bash','brickicl/playground/cluster_service.sh'],cwd=SERVICE,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
def stop(pid):
    os.kill(pid,signal.SIGTERM)
    for _ in range(60):
        if not Path(f'/proc/{pid}').exists():return
        time.sleep(.5)
    raise RuntimeError('Service did not stop gracefully')
def validate(summary,expected):
    for _ in range(60):
        try:
            health=get('/health');assert health['service_id']=='sunshq-gradio-live' and health['samples']==72
            config=get('/config')
            markdown=[c['props'].get('value','') for c in config['components'] if c['type']=='markdown']
            assert summary.strip() in [s.strip() for s in markdown if isinstance(s,str)]
            model=next(c['props'] for c in config['components'] if c['props'].get('label')=='01 / 选择模型')
            assert {c[1] for c in model['choices']}==set(expected)
            return health
        except (OSError,ValueError,KeyError,AssertionError,StopIteration):time.sleep(1)
    raise RuntimeError('Live Gradio config did not match the staged results')
def main():
    for file in ('SCORING_COMPLETED.json','EXTRA_COMPLETED.json','VIEWER_STAGED.json','VIEWER_VALIDATED.json','REGISTRY_VALIDATED.json'):
        assert (ROOT/file).exists(),file
    summary=(ROOT/'results_summary.md').read_text()
    stage_app=ROOT/'gradio_stage/brickicl/playground'
    (stage_app/'results_summary.md').write_text(summary)
    for file in APP_FILES[:3]:compile((stage_app/file).read_text(),str(stage_app/file),'exec')
    with tarfile.open(ROOT/'initial_sources.tgz') as archive:
        for file in APP_FILES:
            expected=hashlib.sha256(archive.extractfile('brickicl/playground/'+file).read()).hexdigest()
            assert sha(PLAY/file)==expected,(file,'concurrent application edit')
    assert get('/health')['service_id']=='sunshq-gradio-live'
    pid=process()
    for _ in range(60):
        queue=get('/gradio_api/queue/status')['queue_size']
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        memory=[int(line.split(',')[1].strip()) for line in query.stdout.splitlines() if line.split(',')[0].strip()==str(pid)]
        if queue==0 and (not memory or memory[0]<3000):break
        time.sleep(2)
    else:raise RuntimeError('Gradio still has active inference')
    backup=ROOT/'deployment_backup';backup.mkdir(exist_ok=True)
    for file in APP_FILES:
        dest=backup/'gradio'/file;dest.parent.mkdir(exist_ok=True);shutil.copy2(PLAY/file,dest)
    for file in BOARDS:
        dest=backup/'viewer'/file;dest.parent.mkdir(exist_ok=True);shutil.copy2(LIVE/file,dest)
    shutil.copy2(LIVE/'models/manifest.json',backup/'manifest.json')
    old=json.loads((LIVE/'models/manifest.json').read_text())
    new=json.loads((STAGE/'models/manifest.json').read_text())
    replaced=set(json.loads((ROOT/'VIEWER_STAGED.json').read_text())['replace_tasks'])
    preserved=[r for r in old if r.get('task') not in replaced]
    merged=new+preserved
    assert len(new)==2596
    assert sum(r.get('task')=='two_copy' for r in merged)==sum(r.get('task')=='two_copy' for r in old)==16
    before_sha=sha(LIVE/'models/manifest.json')
    shutil.copytree(STAGE/'models/structural_eval_20261115',LIVE/'models/structural_eval_20261115',dirs_exist_ok=True)
    assert all((LIVE/'models'/r['file']).is_file() for r in merged)
    assert sha(LIVE/'models/manifest.json')==before_sha
    child=None
    try:
        for file in BOARDS:replace_file(STAGE/file,LIVE/file)
        _write_json(LIVE/'models/manifest.json',merged);write_catalog(LIVE/'models',merged)
        for file in APP_FILES:replace_file(stage_app/file,PLAY/file)
        stop(pid);child=start()
        expected=json.loads((ROOT/'REGISTRY_VALIDATED.json').read_text())['models']
        health=validate(summary,expected)
        result={'deployed':True,'models':len(merged),'new_models':len(new),'model_choices':list(expected),
                'health':health,'summary_sha256':sha(PLAY/'results_summary.md'),'previous_deployment':str(backup)}
        (ROOT/'PUBLISHED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException:
        for file in BOARDS:replace_file(backup/'viewer'/file,LIVE/file)
        restore=json.loads((backup/'manifest.json').read_text())
        _write_json(LIVE/'models/manifest.json',restore);write_catalog(LIVE/'models',restore)
        for file in APP_FILES:replace_file(backup/'gradio'/file,PLAY/file)
        if child and child.poll() is None:child.terminate();child.wait(timeout=30)
        if not Path(f'/proc/{pid}').exists():start()
        raise
if __name__=='__main__':main()
