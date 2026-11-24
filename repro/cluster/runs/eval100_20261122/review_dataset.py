import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];RUNTIME=REPO/'runs/proposal_report_20261111';BR=RUNTIME/'browser';OUT=ROOT/'visual_review';OUT.mkdir(exist_ok=True)
def main():
 data=json.loads(Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer/eval100_review.json').read_text());checked=[];errors=[]
 with sync_playwright() as pw:
  b=pw.chromium.launch(executable_path=str(BR/'chromium'),headless=True,env=dict(os.environ,LD_LIBRARY_PATH=f"{BR}:{BR/'lib'}:{RUNTIME/'nspr-focal/usr/lib/x86_64-linux-gnu'}",FONTCONFIG_FILE=str(BR/'fonts.conf'),FONTCONFIG_PATH=str(BR)),args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
  p=b.new_page(viewport={'width':1100,'height':850});p.on('pageerror',lambda e:errors.append(str(e)));p.goto('http://127.0.0.1:17860/viewer/eval100_review.html?source=114',wait_until='domcontentloaded');p.locator('#source option').first.wait_for(state='attached')
  for i,sid in enumerate(data['source_order']):
   p.locator('#source').select_option(sid);p.wait_for_function("document.querySelectorAll('iframe[data-ready=true]').length===2",timeout=60000)
   p.locator('main').screenshot(path=str(OUT/f'{sid}.png'));checked.append(sid)
   if len(checked)%10==0:print('Rendered',len(checked),'/100',flush=True)
  b.close()
 assert len(checked)==100 and not errors,errors
 for start in range(0,100,10):
  sheet=Image.new('RGB',(1800,5*540),'white');draw=ImageDraw.Draw(sheet)
  for k,sid in enumerate(checked[start:start+10]):
   im=Image.open(OUT/f'{sid}.png');im.thumbnail((900,505));x=(k%2)*900;y=(k//2)*540;sheet.paste(im,(x,y+30));draw.text((x+10,y+8),f'{start+k+1:03d} | source {sid} | normal / stretched',fill='#14263d')
  sheet.save(OUT/f'contact_{start//10+1:02d}.jpg',quality=90)
 (ROOT/'RENDER_VERIFIED.json').write_text(json.dumps({'sources':checked,'reference_renders':200,'page_errors':errors,'visual_inspection_pending':True},indent=2));print('All 200 references rendered; contact sheets ready',flush=True)
if __name__=='__main__':main()
