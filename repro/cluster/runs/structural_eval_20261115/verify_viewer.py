import functools,json,os,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import quote
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;STAGE=ROOT/'viewer_stage'
runtime=Path('/home/sunshq114514/brickICL/runs/proposal_report_20261111');br=runtime/'browser'
entries=json.loads((STAGE/'models/manifest.json').read_text())
assert len(entries)==2596 and all((STAGE/'models'/e['file']).is_file() for e in entries)
assert not any(e.get('condition') in ('ablated','deleted') for e in entries)
for board in ('unified_compare','nzd_order_compare'):
    data=json.loads((STAGE/f'{board}.json').read_text())
    assert len(data['sources'])==18
    assert all(set(p['conditions'])=={'original','stretched'} for s in data['sources'].values() for p in s['primes'].values())
(STAGE/'audit_single.html').write_text("""<!doctype html><meta charset="utf-8">
<iframe id="frame" style="width:100vw;height:90vh;border:0"></iframe>
<script>let f=new URLSearchParams(location.search).get('file');document.getElementById('frame').src='./?embed=1&model='+encodeURIComponent(f)+'#'+encodeURIComponent(f);
addEventListener('message',e=>{if(e.data?.type==='bricknet-viewer-ready')window.readyFile=e.data.file});</script>""")
class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(STAGE)))
threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}/'
errors=[];checks=[]
try:
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(br/'chromium'),headless=True,
            env=dict(os.environ,LD_LIBRARY_PATH=f"{br}:{br/'lib'}:{runtime/'nspr-focal/usr/lib/x86_64-linux-gnu'}",
                     FONTCONFIG_FILE=str(br/'fonts.conf'),FONTCONFIG_PATH=str(br)),
            args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
        try:
            for board,count in [('unified_compare',6),('nzd_order_compare',2)]:
                page=browser.new_page(viewport={'width':1440,'height':1100})
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(base+board+'.html?source=114&prime=1',wait_until='domcontentloaded')
                page.wait_for_function("n=>document.querySelectorAll('iframe[data-ready=true]').length===n",arg=count,timeout=90000)
                assert page.locator('[data-condition="ablated"]').count()==0
                data=json.loads((STAGE/f'{board}.json').read_text())
                for sid,p,cond,sample in [('114',1,'stretched',0),('458',0,'original',2)]:
                    page.locator('#source').select_option(value=sid)
                    page.locator('#prime').select_option(index=p)
                    page.locator(f'[data-condition="{cond}"]').click()
                    page.locator('#sample').select_option(index=sample)
                    expected=[c[sample]['file'] for c in data['sources'][sid]['primes'][str(p)]['conditions'][cond]['outputs'].values()]
                    page.wait_for_function("""files=>{
                        const frames=[...document.querySelectorAll('iframe')];
                        return frames.length===files.length&&frames.every(f=>f.dataset.ready==='true'&&files.includes(f.dataset.file));
                    }""",arg=expected,timeout=90000)
                    checks.append([board,sid,p,cond,sample])
                    if sid=='114':page.screenshot(path=str(ROOT/f'{board}.png'),full_page=True)
                page.close()
            page=browser.new_page(viewport={'width':1100,'height':1000})
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.add_init_script("addEventListener('message',e=>{if(e.data?.type==='bricknet-viewer-ready')window.readyFile=e.data.file})")
            rollback=json.loads((STAGE/'rollback_conditions.json').read_text())
            for condition in ('original','stretched'):
                r=rollback['sources']['114']['conditions'][condition]['results']['post_pt'][0]
                page.goto(base+f'rollback_compare.html?source=114&condition={condition}&arm=post_pt&sample=0',wait_until='domcontentloaded')
                page.wait_for_function('f=>window.readyFile===f',arg=r['file'],timeout=90000)
                assert '截断' in page.locator('#metrics').inner_text()
                checks.append(['rollback','114',condition])
            page.close()
            for arm,file in [('post_pt','gallery/unconditional_post_pt_s1.mpd'),('fullg_pt','gallery/unconditional_fullg_pt_s1.mpd'),
                             ('post_sft','gallery/fox_post_sft_s1.mpd'),('fullg_sft','gallery/fox_fullg_sft_s1.mpd')]:
                page=browser.new_page(viewport={'width':850,'height':850})
                page.on('pageerror',lambda e:errors.append(str(e)))
                target='structural_eval_20261115/'+file
                page.goto(base+'audit_single.html?file='+quote(target),wait_until='domcontentloaded')
                page.wait_for_function('f=>window.readyFile===f',arg=target,timeout=90000)
                checks.append(['gallery',arm]);page.close()
            assert not errors,errors
        finally:browser.close()
finally:server.shutdown();server.server_close()
report={'files_verified':len(entries),'interactive_checks':checks,'browser_errors':errors,'status':'passed'}
(ROOT/'VIEWER_VALIDATED.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
