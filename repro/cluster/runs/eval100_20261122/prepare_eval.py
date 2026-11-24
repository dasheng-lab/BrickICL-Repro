import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];SERVICE=Path('/home/sunshq114514/brickicl-gradio-service');sys.path.insert(0,str(SERVICE))
from brickicl.playground.evaluated_models import evaluated_registry
from transformers import AutoTokenizer
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[json.loads(s) for s in (ROOT/'p1_inputs.jsonl').open()];assert len(rows)==200
 models=j(REPO/'runs/viewer_parse_fox_fix_20261121/FOX_PROTOCOL.json')['models']
 for m in j(REPO/'runs/latest_results_20261122/delivery/metrics.json'):models[m['model']['arm']]=m['model']
 specs=[]
 for key,reg in evaluated_registry().items():
  arm=next(a for a,m in models.items() if tuple(m['adapters'])==reg.adapters);s=dict(models[arm]);s.update(key=arm,arm=arm,registry_key=key,recipe=reg.label.split(' · ')[0],base=str(REPO/'runs/qwen3_direct_pt_20261117/verified_base'))
  s['local_weight_sha256']={a:sha(Path(a)/'adapter_model.safetensors') for a in s['adapters'] if a.startswith('/')};specs.append(s)
 assert len(specs)==15 and all('sft' not in s['arm'] for s in specs)
 base=Path(specs[0]['base']);tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
 # Preserve old 18-case generation grouping/caps, with the separately corrected fox pair.
 legacy=j(REPO/'runs/structural_eval_20261115/protocol.json')['batch_plan'];groups=[]
 for group in legacy:
  ids=[k for k in group['ids'] if not k.startswith('val:114:')]
  if ids:groups.append(dict(group,ids=ids))
 groups.append({'ids':['val:114:original','val:114:stretched'],'seed':291246,'max_new_tokens':736})
 new=[r for r in rows if r['source_key'].startswith('synthetic:')]
 new.sort(key=lambda r:(len(tok.encode(r['prefix'],add_special_tokens=False)),r['id']))
 for start in range(0,len(new),8):
  rs=new[start:start+8];cap=max(512,64*max(r['k_long'] for r in rs)+256)
  assert all(len(tok.encode(r['prefix'],add_special_tokens=False))+cap<=6400 for r in rs)
  groups.append({'ids':[r['id'] for r in rs],'seed':20261122+start*1009,'max_new_tokens':cap})
 assert len({k for g in groups for k in g['ids']})==200
 source=(REPO/'runs/structural_eval_20261115/score_results.py').read_text()
 source=source.replace('assert len(sources)==18 and all(',"assert len(sources)==100 and all(")
 source=source.replace('integers(18,size=(2000,18))','integers(len(sources),size=(2000,len(sources)))')
 compile(source,str(ROOT/'reference_scoring.py'),'exec');(ROOT/'reference_scoring.py').write_text(source)
 protocol={'models':specs,'sources':100,'legacy':18,'new_constructed':82,'conditions':['original','stretched'],'prime':1,'samples_per_condition':4,'temperature':1.,'top_k':20,'top_p':.95,'batch_plan':groups,'input_sha256':sha(ROOT/'p1_inputs.jsonl'),'scoring_sha256':sha(ROOT/'reference_scoring.py'),'independent_generation':{'samples_per_condition':64,'conditions':['natural','no_eos'],'prompt':'a','batch_size':32,'batch_seeds':[0,32],'max_new_tokens':4096,'max_parts':100},'checkpoint_selection':'unchanged development-best; no selection on new test','new_GRPO':'v2.1/400 unreadable; retain explicitly labeled old r1/300'}
 (ROOT/'plan.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));print('Prepared',len(specs),'PT models;',len(groups),'completion batches per model; 800 completions + 128 independent samples each',flush=True)
if __name__=='__main__':main()
