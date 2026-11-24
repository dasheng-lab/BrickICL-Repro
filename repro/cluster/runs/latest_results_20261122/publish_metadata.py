import json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer');STAGE=ROOT/'viewer_stage'
assert json.loads((LIVE/'models/manifest.json').read_text())==json.loads((STAGE/'models/manifest.json').read_text())
files=list((STAGE/'models/meta').glob('*.json'));assert len(files)>=2944,len(files)
for src in files:
 dst=LIVE/'models/meta'/src.name;dst.parent.mkdir(exist_ok=True);temp=dst.with_suffix('.latest.tmp');shutil.copy2(src,temp);temp.chmod(0o775);temp.replace(dst)
print('Published per-model metadata',len(files),flush=True)
