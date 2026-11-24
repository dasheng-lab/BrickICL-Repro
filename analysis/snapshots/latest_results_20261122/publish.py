import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer');PLAY=Path('/home/sunshq114514/brickicl-gradio-service/brickicl/playground')
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy_atomic(src,dest):
 dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.latest.tmp');shutil.copy2(src,temp);temp.chmod(0o775);temp.replace(dest)
def main():
 v=j(ROOT/'VIEWER_STAGED.json');s=j(ROOT/'SERVICE_STAGED.json');assert sha(LIVE/'models/manifest.json')==v['source_manifest_sha256'] and sha(LIVE/'index.html')==v['source_index_sha256']
 for p,digest in s['source_sha256'].items():assert sha(Path(p))==digest,p
 # Stage registry in an isolated package, without starting a second server or loading a model.
 pkg=ROOT/'stage_import/brickicl/playground';pkg.mkdir(parents=True,exist_ok=True)
 for p in [pkg/'__init__.py',pkg.parent/'__init__.py']:p.write_text('')
 for name in ['backend.py','evaluated_models.py','latest_models.json']:shutil.copy2(ROOT/'service_stage'/name,pkg/name)
 env=dict(os.environ,PYTHONPATH=str(ROOT/'stage_import'),BRICKICL_POSTTRAIN_ROOT=str(REPO))
 result=subprocess.run(['python','-c',"from brickicl.playground.evaluated_models import evaluated_registry; r=evaluated_registry(); assert len(r)==15,len(r); assert sum(k.startswith('latest-') for k in r)==3; print('15 PT model choices verified')"],cwd=ROOT/'stage_import',env=env,capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr;print(result.stdout,flush=True)
 backup=ROOT/'backup';backup.mkdir(exist_ok=True)
 targets=[]
 for p in (ROOT/'service_stage').iterdir():targets.append((p,REPO/p.name if p.name.startswith('README_') else PLAY/p.name))
 for name in ['index.html','latest_compare.html','latest_compare.json','models/manifest.json','models/catalog.json']:
  targets.append((ROOT/'viewer_stage'/name,LIVE/name))
 for src,dst in targets:
  if dst.exists():
   b=backup/dst.relative_to(Path('/home/sunshq114514'));b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,b)
 # Immutable MPDs and lazy groups precede the catalog/manifest pointer switch.
 for file in v['new_files']:copy_atomic(ROOT/'viewer_stage/models'/file,LIVE/'models'/file)
 for src in (ROOT/'viewer_stage/models/meta').glob('*.json'):copy_atomic(src,LIVE/'models/meta'/src.name)
 cat=j(ROOT/'viewer_stage/models/catalog.json')
 for g in cat['groups']:
  # Catalog schema stores per-group URLs relative to models/.
  for value in g.values():
   if isinstance(value,str) and value.endswith('.json'):
    src=ROOT/'viewer_stage/models'/value
    if src.is_file():copy_atomic(src,LIVE/'models'/value)
 for src,dst in targets:copy_atomic(src,dst)
 report={'state':'published','new_models':3,'new_viewer_entries':456,'active_grpo':'r1/checkpoint-300 (latest v2.1 blocked by weight permission)','unix_time':time.time()}
 (ROOT/'PUBLISHED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
