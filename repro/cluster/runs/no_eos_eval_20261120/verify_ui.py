import json,os,zipfile,shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from common import ROOT,REPO,atomic

RUNTIME=REPO/'runs/proposal_report_20261111';BR=RUNTIME/'browser'
def main():
    assert (ROOT/'PUBLISHED.json').exists()
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(BR/'chromium'),headless=True,
            env=dict(os.environ,LD_LIBRARY_PATH=f"{BR}:{BR/'lib'}:{RUNTIME/'nspr-focal/usr/lib/x86_64-linux-gnu'}",FONTCONFIG_FILE=str(BR/'fonts.conf'),FONTCONFIG_PATH=str(BR)),
            args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1800,'height':1200});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:17860/',wait_until='domcontentloaded')
        page.get_by_text('PT 统一评测：给第一块 · 按后训练起点分组',exact=True).click()
        page.get_by_text('禁 EOS 有效长度',exact=True).first.wait_for()
        tables=page.locator('table');assert tables.count()==4
        count=0
        for i in range(4):
            t=tables.nth(i);assert t.locator('thead th').count()==10
            assert '禁 EOS 有效长度' in t.inner_text() and '禁 EOS 无碰撞长度' in t.inner_text()
            count+=t.locator('tbody tr').count()
        assert count==12 and not errors
        box=page.locator('.prose').filter(has=page.locator('table')).first
        box.screenshot(path=str(ROOT/'delivery/no_eos_table.png'))
        browser.close()
    atomic(ROOT/'UI_VERIFIED.json',{'rows':12,'groups':4,'columns_per_table':10,'page_errors':errors})
    for name in ('plan.json','VERIFIED.json','PUBLISHED.json','COMPLETED.json','UI_VERIFIED.json','README.zh-CN.md'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,ROOT/'delivery'/name)
    with zipfile.ZipFile(ROOT/'delivery/no_eos_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted((ROOT/'delivery').iterdir()):
            if file.suffix!='.zip':z.write(file,file.name)
    print('Verified live 12-row, 10-column PT table',flush=True)
if __name__=='__main__':main()
