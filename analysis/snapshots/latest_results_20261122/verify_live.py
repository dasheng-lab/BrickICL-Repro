import json,os,urllib.request,urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];APP='http://127.0.0.1:17860';RUNTIME=REPO/'runs/proposal_report_20261111';BR=RUNTIME/'browser';OUT=ROOT/'output/playwright';OUT.mkdir(parents=True,exist_ok=True)
def get(path):
 with urllib.request.urlopen(APP+path,timeout=20) as f:return json.load(f)
def main():
 cat=get('/viewer/models/catalog.json');manifest=get('/viewer/models/manifest.json');cfg=get('/config');health=get('/health')
 assert health['samples']==36
 for task,n,count in [('completion_given_head',18,144),('rollback_pt',3,32),('from_scratch',18,None),('posttrain_icl',5,None)]:
  groups=[g for g in cat['groups'] if g['task']==task];assert len(groups)==n,(task,len(groups))
  for g in groups:
   if count:assert g['count']==count
   if task in ('completion_given_head','rollback_pt'):assert next(r for r in manifest if r['task']==task and r['group']==g['group'])['source']==114
 assert len(manifest)==2944 and len(cat['aliases'])==72
 texts=[c['props'].get('value','') for c in cfg['components'] if c['type']=='markdown'];table=next(t for t in texts if '## PT 结果' in t)
 assert '0.7093' in table and '0.4948' in table and '0.4366' in table and '600' in table
 assert 'AdamW 0.08' not in table and '| Qwen (0.6B) |' not in table
 drop=next(c for c in cfg['components'] if c['type']=='dropdown' and c['props'].get('label')=='01 / 选择模型')
 assert len(drop['props']['choices'])==15
 new=json.loads((ROOT/'service_stage/latest_models.json').read_text());errors=[];checks=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path=str(BR/'chromium'),headless=True,env=dict(os.environ,LD_LIBRARY_PATH=f"{BR}:{BR/'lib'}:{RUNTIME/'nspr-focal/usr/lib/x86_64-linux-gnu'}",FONTCONFIG_FILE=str(BR/'fonts.conf'),FONTCONFIG_PATH=str(BR)),args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox','--disable-dev-shm-usage'])
  page=browser.new_page(viewport={'width':1500,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(APP+'/viewer/',wait_until='domcontentloaded');page.wait_for_function("document.querySelectorAll('#groupSelect option').length===18")
  for spec in new:
   r=next(r for r in manifest if r.get('arm')==spec['arm'] and r['task']=='completion_given_head');g=next(g for g in cat['groups'] if g['task']==r['task'] and g['group']==r['group'])
   page.locator('#groupSelect').select_option(g['id']);page.wait_for_function("document.querySelectorAll('#list .item').length===144")
   assert '狐狸' in page.locator('#list .item').first.inner_text()
   page.wait_for_function("file=>location.hash.includes(encodeURIComponent(file)) || decodeURIComponent(location.hash.slice(1))===file",arg=r['file'])
   checks.append('model-first navigation '+spec['arm'])
  page.screenshot(path=str(OUT/'viewer_latest.png'),full_page=True)
  page.locator('#taskNav button[data-task="from_scratch"]').click();page.wait_for_function("document.querySelectorAll('#groupSelect option').length===18")
  for spec in new:
   r=next(r for r in manifest if r.get('arm')==spec['arm'] and r['task']=='from_scratch');g=next(g for g in cat['groups'] if g['task']==r['task'] and g['group']==r['group'])
   page.locator('#groupSelect').select_option(g['id']);page.wait_for_function("document.querySelectorAll('#list .item').length===8")
   assert '禁止 EOS' in page.locator('#list').inner_text();checks.append('natural/no-EOS '+spec['arm'])
  page.goto(APP+'/viewer/latest_compare.html?source=114&prime=1',wait_until='domcontentloaded')
  page.wait_for_function("document.querySelectorAll('iframe[data-ready=true]').length===4",timeout=120000)
  assert page.locator('#prime option').count()==1
  page.screenshot(path=str(OUT/'latest_comparison.png'),full_page=True);checks.append('four-model board renders through single port')
  page.goto(APP+'/',wait_until='domcontentloaded');page.get_by_text('PT 统一评测：模型 + 训练数据 + 训练目标',exact=True).click();page.locator('a[href^="/viewer/latest_compare.html"]').first.wait_for(timeout=60000);checks.append('Gradio updated result table and relative viewer link')
  browser.close()
 assert not errors,errors
 (ROOT/'LIVE_VERIFIED.json').write_text(json.dumps({'health':health,'checks':checks,'page_errors':errors,'gradio_choices':15,'viewer_models':18,'aliases':72},ensure_ascii=False,indent=2))
 print('Verified live Gradio and Viewer',len(checks),flush=True)
if __name__=='__main__':main()
