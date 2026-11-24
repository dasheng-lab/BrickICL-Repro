import functools,json,os,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parent/'caption_audit'
runtime=Path('/home/sunshq114514/brickICL/runs/proposal_report_20261111')
br=runtime/'browser'
class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(root/'site')))
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(br/'chromium'),headless=True,
            env=dict(os.environ,LD_LIBRARY_PATH=f"{br}:{br/'lib'}:{runtime/'nspr-focal/usr/lib/x86_64-linux-gnu'}",
                     FONTCONFIG_FILE=str(br/'fonts.conf'),FONTCONFIG_PATH=str(br)),
            args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
        try:
            page=browser.new_page(viewport={'width':640,'height':686})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            for case in json.loads((root/'selected.json').read_text()):
                page.goto(f'http://127.0.0.1:{server.server_port}/?embed=1#{case["file"]}',wait_until='domcontentloaded')
                page.wait_for_function("f=>window.auditState?.entry?.file===f && window.auditState?.model",arg=case['file'],timeout=60000)
                page.locator('canvas').first.screenshot(path=str(root/f'case-{case["number"]}-a.png'))
                page.evaluate('window.auditRotate()')
                page.evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))')
                page.locator('canvas').first.screenshot(path=str(root/f'case-{case["number"]}-b.png'))
                a=Image.open(root/f'case-{case["number"]}-a.png');b=Image.open(root/f'case-{case["number"]}-b.png')
                pair=Image.new('RGB',(a.width+b.width,max(a.height,b.height)),'white')
                pair.paste(a,(0,0));pair.paste(b,(a.width,0));pair.save(root/f'case-{case["number"]}.png')
                print('RENDERED',case['number'],case['source'],flush=True)
            assert not errors,errors
        finally:browser.close()
finally:
    server.shutdown();server.server_close()
