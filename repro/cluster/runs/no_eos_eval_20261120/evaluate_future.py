"""Apply identical no-EOS protocol to a newly evaluated dev-best checkpoint."""
import json,shutil,subprocess,sys
from pathlib import Path
from common import ROOT,sha,atomic

def main(natural_eval):
    source=Path(natural_eval).resolve();spec=json.loads((source/'plan.json').read_text())['checkpoints'][0]
    out=source/'no_eos';out.mkdir(exist_ok=True)
    if (out/'COMPLETED.json').exists():return
    for name in ('common.py','generate.py','score.py','reference_scoring.py'):
        text=(ROOT/name).read_text().replace("'models':12,'samples':768","'models':1,'samples':64")
        if name=='generate.py':text=text.replace("'models':12","'models':1")
        compile(text,str(out/name),'exec');(out/name).write_text(text)
    new={'key':spec['key'],'label':spec['label'],'base':spec['base'],'adapters':spec['adapters'],
         'local_weight_sha256':{a:sha(Path(a)/'adapter_model.safetensors') for a in spec['adapters'] if a.startswith('/')}}
    plan=json.loads((ROOT/'plan.json').read_text());plan['models']=[new];plan['scoring_sha256']=sha(out/'reference_scoring.py')
    (out/'results'/new['key']).mkdir(parents=True,exist_ok=True);atomic(out/'plan.json',plan)
    for script in ('generate.py','score.py'):subprocess.run([sys.executable,'-u',str(out/script)],cwd=source.parents[1],check=True)
    metrics=json.loads((out/'results'/new['key']/'metrics.json').read_text());assert metrics['eos_suppression_verified'] and metrics['samples']==64
    atomic(out/'COMPLETED.json',{'state':'completed','samples':64,'eos_suppression_verified':True})
    print('Future model no-EOS complete',new['key'],flush=True)
if __name__=='__main__':main(sys.argv[1])
