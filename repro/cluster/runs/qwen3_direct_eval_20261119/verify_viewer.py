"""Use the repository's existing cluster-side browser QA runtime."""
import json,os,threading,urllib.parse
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright
from common import ROOT,atomic

STAGE=ROOT/'viewer_stage';LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
RUNTIME=ROOT.parent/'proposal_report_20261111';BR=RUNTIME/'browser'
class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
    def translate_path(self,path):
        rel=urllib.parse.unquote(urllib.parse.urlsplit(path).path).lstrip('/')
        if not rel or rel.endswith('/'):rel+='index.html'
        candidate=(STAGE/rel).resolve()
        if not candidate.is_relative_to(STAGE.resolve()):raise ValueError(rel)
        return str(candidate if candidate.is_file() else LIVE/rel)

def main():
    manifest=json.loads((STAGE/'models/manifest.json').read_text());new=[r for r in manifest if r.get('evaluation_run')==ROOT.name]
    assert len(new)==876 and all((STAGE/'models'/r['file']).is_file() for r in new)
    assert not any(r.get('condition') in ('ablated','deleted') for r in new)
    assert 'training_qwen3_direct/' not in (STAGE/'qwen3_direct_compare.html').read_text()
    (STAGE/'audit_single.html').write_text('''<!doctype html><iframe id="frame" style="width:95vw;height:95vh;border:0"></iframe>
<script>let file=new URLSearchParams(location.search).get('file');document.getElementById('frame').src='./?embed=1#'+encodeURIComponent(file);
addEventListener('message',e=>{if(e.data?.type==='bricknet-viewer-ready')window.readyFile=e.data.file});</script>''')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}/';errors=[];checks=[]
    board=json.loads((STAGE/'qwen3_direct_compare.json').read_text())
    assert len(board['sources'])==18 and len(board['arms'])==6
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(BR/'chromium'),headless=True,
            env=dict(os.environ,LD_LIBRARY_PATH=f"{BR}:{BR/'lib'}:{RUNTIME/'nspr-focal/usr/lib/x86_64-linux-gnu'}",FONTCONFIG_FILE=str(BR/'fonts.conf'),FONTCONFIG_PATH=str(BR)),
            args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1500,'height':1250});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(base+'qwen3_direct_compare.html?source=114&prime=1',wait_until='domcontentloaded')
        page.wait_for_function("document.querySelectorAll('iframe[data-ready=true]').length===6",timeout=120000)
        for sid,p,condition,sample in [('114','1','stretched',0),('458','0','original',2),('114','1','original',1)]:
            page.locator('#source').select_option(sid);page.locator('#prime').select_option(p)
            page.locator(f'[data-condition="{condition}"]').click();page.locator('#sample').select_option(str(sample))
            expected=[outputs[sample]['file'] for outputs in board['sources'][sid]['primes'][p]['conditions'][condition]['outputs'].values()]
            page.wait_for_function("files=>{let fs=[...document.querySelectorAll('iframe')];return fs.length===6&&fs.every(f=>f.dataset.ready==='true'&&files.includes(f.dataset.file));}",arg=expected,timeout=120000)
            checks.append({'source':sid,'prime':p,'condition':condition,'sample':sample})
            if sid=='114' and condition=='stretched':page.screenshot(path=str(ROOT/'delivery/qwen3_direct_compare.png'),full_page=True)
        page.locator('#only').click();assert page.locator('#only').get_attribute('aria-pressed')=='true'
        page.locator('#reset').click();page.close()
        for arm in ('direct_m','direct_g','direct_bricknet'):
            file=next(r['file'] for r in new if r.get('arm')==arm and r.get('from_scratch'))
            page=browser.new_page(viewport={'width':950,'height':900});page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base+'audit_single.html?file='+urllib.parse.quote(file,safe=''),wait_until='domcontentloaded')
            page.wait_for_function('file=>window.readyFile===file',arg=file,timeout=90000)
            assert page.frame_locator('#frame').locator('canvas').count()>=1
            checks.append({'independent_gallery':arm});page.close()
        browser.close()
    server.shutdown();assert not errors,errors
    atomic(ROOT/'VIEWER_VALIDATED.json',{'new_models':876,'checks':checks,'browser_errors':errors,'cluster_browser_only':True})
    print('VIEWER VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
