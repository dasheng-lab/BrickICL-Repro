import json,os,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];BR=REPO/'runs/proposal_report_20261111/browser';RUNTIME=BR.parent;APP='http://127.0.0.1:17860'
def get(path):
 with urllib.request.urlopen(APP+path,timeout=30) as f:return json.load(f)
def main():
 health=get('/health');assert health['samples']==200
 manifest=get('/viewer/models/manifest.json');cat=get('/viewer/models/catalog.json');metrics=json.loads((ROOT/'delivery/metrics.json').read_text());assert len(metrics)==15 and len(manifest)==13112
 for arm in metrics:
  rs=[r for r in manifest if r.get('arm')==arm and r['task']=='completion_given_head'];assert len(rs)==800 and len({r['source'] for r in rs})==100 and rs[0]['source']==114
  assert len([r for r in manifest if r.get('arm')==arm and r['task']=='from_scratch'])==8
 assert len([g for g in cat['groups'] if g['task']=='posttrain_icl'])==5
 assert sum(r['task']=='two_copy' for r in manifest)==16 and sum(r['task']=='rollback_pt' for r in manifest)==96
 cfg=get('/config');text='\n'.join(c['props'].get('value','') for c in cfg['components'] if c['type']=='markdown')
 assert '100 个固定案例' in text and '82' in text and '新WJY' in text and '| Qwen (0.6B) |' not in text and 'AdamW 0.08' not in text
 for m in metrics.values():assert f"{m['completion']['1']['f1']['stretched']:.4f}" in text
 out=ROOT/'output/playwright';out.mkdir(parents=True,exist_ok=True);errors=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path=str(BR/'chromium'),headless=True,env=dict(os.environ,LD_LIBRARY_PATH=f"{BR}:{BR/'lib'}:{RUNTIME/'nspr-focal/usr/lib/x86_64-linux-gnu'}",FONTCONFIG_FILE=str(BR/'fonts.conf'),FONTCONFIG_PATH=str(BR)),args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
  page=browser.new_page(viewport={'width':1500,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)));page.goto(APP+'/viewer/',wait_until='domcontentloaded');page.wait_for_function("document.querySelectorAll('#groupSelect option').length===18")
  group=next(g for g in cat['groups'] if g['task']=='completion_given_head' and g['group']=='官方 PT');page.locator('#groupSelect').select_option(group['id']);page.wait_for_function("document.querySelectorAll('#list .item').length===800");assert '狐狸' in page.locator('#list .item').first.inner_text()
  page.goto(APP+'/viewer/latest_compare.html?source=1003&prime=1',wait_until='domcontentloaded');page.wait_for_function("document.querySelectorAll('iframe[data-ready=true]').length===4",timeout=120000);assert page.locator('#source option').count()==100;page.screenshot(path=str(out/'new_case_comparison.png'),full_page=True)
  page.goto(APP+'/',wait_until='domcontentloaded');page.get_by_text('PT 统一评测：模型 + 训练数据 + 训练目标',exact=True).click();page.locator('a[href="/viewer/eval100_review.html"]').first.wait_for();page.screenshot(path=str(out/'gradio_100case.png'),full_page=True);browser.close()
 assert not errors,errors
 (ROOT/'LIVE_VERIFIED.json').write_text(json.dumps({'state':'verified','models':15,'sources':100,'gradio_samples':200,'viewer_entries':len(manifest),'browser_errors':errors},indent=2));print('Published 100-case Gradio and Viewer verified',flush=True)
if __name__=='__main__':main()
