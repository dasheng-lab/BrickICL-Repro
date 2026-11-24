import json,os,urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];BR=REPO/'runs/proposal_report_20261111/browser';RUNTIME=BR.parent;OUT=ROOT/'report/assets/renders'
def main():
 scenes=json.loads((ROOT/'report/assets/render_provenance.json').read_text());errors=[]
 with sync_playwright() as pw:
  b=pw.chromium.launch(executable_path=str(BR/'chromium'),headless=True,env=dict(os.environ,LD_LIBRARY_PATH=f"{BR}:{BR/'lib'}:{RUNTIME/'nspr-focal/usr/lib/x86_64-linux-gnu'}",FONTCONFIG_FILE=str(BR/'fonts.conf'),FONTCONFIG_PATH=str(BR)),args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
  p=b.new_page(viewport={'width':900,'height':800},device_scale_factor=1.5);p.on('pageerror',lambda e:errors.append(str(e)))
  for s in scenes:
   p.goto('http://127.0.0.1:17860/viewer/report_figures_20261123/?embed=1#'+urllib.parse.quote(s['file'],safe=''),wait_until='domcontentloaded');p.wait_for_function('file=>window.reportReadyFile===file',arg=s['file'],timeout=90000);p.locator('#stage canvas').screenshot(path=str(OUT/(s['scene']+'.png')));print('Rendered',s['scene'],flush=True)
  b.close()
 assert not errors,errors
 (ROOT/'report/assets/RENDER_VERIFIED.json').write_text(json.dumps({'scenes':len(scenes),'actual_viewer':True,'browser_errors':errors},indent=2))
if __name__=='__main__':main()
