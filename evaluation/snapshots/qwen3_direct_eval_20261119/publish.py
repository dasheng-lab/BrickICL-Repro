import json,os,shutil,signal,subprocess,tarfile,time,urllib.request
from pathlib import Path
from brickicl.viz.lazy_catalog import _write_json,write_catalog
from common import *

LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
SERVICE=Path('/home/sunshq114514/brickicl-gradio-service');PLAY=SERVICE/'brickicl/playground'
FILES=('app.py','backend.py','evaluated_models.py','results_summary.md')
APP='http://127.0.0.1:17860'
def get(route):
    with urllib.request.urlopen(APP+route,timeout=10) as f:return json.load(f)
def process():
    pids=[]
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
            if path.stat().st_uid==os.getuid() and b'brickicl.playground.app' in args and b'--port' in args and args[args.index(b'--port')+1]==b'17860':pids.append(int(path.name))
        except (OSError,ValueError,IndexError):pass
    assert len(pids)==1,pids;return pids[0]
def stop(pid):
    os.kill(pid,signal.SIGTERM)
    for _ in range(60):
        if not Path(f'/proc/{pid}').exists():return
        time.sleep(.5)
    raise RuntimeError('Gradio did not stop')
def start():
    log=(SERVICE/'logs/playground.direct-qwen3.log').open('ab')
    return subprocess.Popen(['bash','brickicl/playground/cluster_service.sh'],cwd=SERVICE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
def replace(src,dest):
    tmp=dest.with_name(dest.name+'.direct-tmp');shutil.copy2(src,tmp);tmp.replace(dest)
def main():
    if (ROOT/'PUBLISHED.json').exists():print('Already published');return
    for name in ('VERIFIED.json','VIEWER_VALIDATED.json','REGISTRY_VALIDATED.json'):assert (ROOT/name).exists()
    assert json.loads((ROOT/'VERIFIED.json').read_text())['all_completed']
    staged=json.loads((ROOT/'VIEWER_STAGED.json').read_text());stage=ROOT/'viewer_stage'
    assert sha(LIVE/'models/manifest.json')==staged['live_manifest_sha256']
    assert sha(LIVE/'index.html')==staged['live_index_sha256']
    with tarfile.open(ROOT/'service_snapshot.tgz') as archive:
        for name in FILES:
            assert hashlib.sha256(archive.extractfile('brickicl/playground/'+name).read()).hexdigest()==sha(PLAY/name),name
    assert get('/health')['service_id']=='sunshq-gradio-live'
    assert get('/gradio_api/queue/status')['queue_size']==0
    pid=process();backup=ROOT/'deployment_backup';backup.mkdir(exist_ok=True)
    for name in FILES:shutil.copy2(PLAY/name,backup/name)
    shutil.copy2(LIVE/'index.html',backup/'index.html')
    shutil.copy2(LIVE/'models/manifest.json',backup/'manifest.json')
    prior=json.loads((backup/'manifest.json').read_text());merged=json.loads((stage/'models/manifest.json').read_text())
    assert sum(r.get('task')=='two_copy' for r in merged)==staged['old_two_copy']
    shutil.copytree(stage/'models'/ROOT.name,LIVE/'models'/ROOT.name,dirs_exist_ok=True)
    assert all((LIVE/'models'/r['file']).is_file() for r in merged)
    child=None
    try:
        _write_json(LIVE/'models/manifest.json',merged);write_catalog(LIVE/'models',merged)
        for name in ('index.html','qwen3_direct_compare.html','qwen3_direct_compare.json'):replace(stage/name,LIVE/name)
        for name in FILES:replace(ROOT/'gradio_stage/brickicl/playground'/name,PLAY/name)
        stop(pid);child=start()
        expected=json.loads((ROOT/'REGISTRY_VALIDATED.json').read_text())
        for attempt in range(60):
            try:
                health=get('/health');assert health['service_id']=='sunshq-gradio-live' and health['samples']==72
                config=get('/config');drop=next(c['props'] for c in config['components'] if c['props'].get('label')=='01 / 选择模型')
                assert {r[1] for r in drop['choices']}==set(expected['models']) and drop['value']==expected['default']
                summary=(ROOT/'results_summary.md').read_text().strip()
                assert summary in [c['props'].get('value','').strip() for c in config['components'] if c['type']=='markdown']
                break
            except (OSError,StopIteration,AssertionError,KeyError,ValueError):time.sleep(1)
        else:raise RuntimeError('Live service did not match staged config')
        atomic(ROOT/'PUBLISHED.json',{'published':True,'new_models':876,'total_models':len(merged),'default_models':len(expected['models']),
            'health':health,'backup':str(backup),'summary_sha256':sha(PLAY/'results_summary.md'),'unix_time':time.time()})
        print('Published Qwen3 direct PT',flush=True)
    except BaseException:
        for name in FILES:replace(backup/name,PLAY/name)
        replace(backup/'index.html',LIVE/'index.html');_write_json(LIVE/'models/manifest.json',prior);write_catalog(LIVE/'models',prior)
        if child and child.poll() is None:child.terminate();child.wait(timeout=30)
        if not Path(f'/proc/{pid}').exists():start()
        raise
if __name__=='__main__':main()
