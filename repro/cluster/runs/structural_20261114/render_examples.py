import functools,json,os,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from PIL import Image
root=Path(__file__).resolve().parent/'examples'
runtime=Path('/home/sunshq114514/brickICL/runs/proposal_report_20261111');br=runtime/'browser'
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
            page=browser.new_page(viewport={'width':680,'height':726})
            for r in json.loads((root/'examples.json').read_text()):
                for state in ('before','after'):
                    file=f'{r["number"]}-{state}.mpd'
                    page.goto(f'http://127.0.0.1:{server.server_port}/?embed=1#{file}',wait_until='domcontentloaded')
                    page.wait_for_function('f=>window.auditState?.entry?.file===f && window.auditState?.model',arg=file,timeout=60000)
                    page.locator('canvas').first.screenshot(path=str(root/f'{r["number"]}-{state}.png'))
                a=Image.open(root/f'{r["number"]}-before.png');b=Image.open(root/f'{r["number"]}-after.png')
                joined=Image.new('RGB',(a.width+b.width,max(a.height,b.height)),'white')
                joined.paste(a,(0,0));joined.paste(b,(a.width,0));joined.save(root/f'example-{r["number"]}.png')
                print('Rendered actual before/after',r['number'],flush=True)
        finally:browser.close()
finally:server.shutdown();server.server_close()
