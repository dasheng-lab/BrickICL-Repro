import ast,hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];PLAY=Path('/home/sunshq114514/brickicl-gradio-service/brickicl/playground');STAGE=ROOT/'service_stage';STAGE.mkdir(exist_ok=True)
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
RECIPES={'nzd_v2_plain':'Qwen + Bricknet_traindata + Full_G','nzd_v2_masked':'Qwen + Bricknet_traindata_aug + Full_G','regularized_g_wd010_do015':'Qwen + ourdata + Full_G + AdamW 0.10'}
def main():
 rows=j(ROOT/'delivery/metrics.json');records=[];by_recipe={}
 for m in rows:
  s=dict(m['model']);s['registry_key']='latest-'+s['key']+'-'+Path(s['checkpoint']).name;s['evaluation_root']=m['evaluation_root'];s['recipe']=RECIPES[s['arm']];s['objective']='Full-G';records.append(s);by_recipe[s['recipe']]=m
 (STAGE/'latest_models.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 p=PLAY/'backend.py';text=p.read_text();needle='    return specs\n';assert text.count(needle)==1
 block='''    latest_file = Path(__file__).with_name('latest_models.json')
    if latest_file.exists():
        for record in json.loads(latest_file.read_text()):
            cp = Path(record['checkpoint'])
            provenance = json.loads((cp/'provenance.json').read_text())
            assert provenance['initial_adapters'] == [] and provenance['base_provenance']['verified']
            specs[record['registry_key']] = ModelSpec(record['registry_key'], record['recipe'] + f" · {record['step']}",
                record['base'], tuple(record['adapters']), 'PT', record['objective'])
'''
 text=text.replace(needle,block+needle)
 needle='    config = json.loads((path / "adapter_config.json").read_text())'
 text=text.replace(needle,"    for registered in registry().values():\n        if registered.adapters and Path(registered.adapters[-1]) == path:\n            return registered\n"+needle)
 compile(text,str(p),'exec');(STAGE/p.name).write_text(text)
 p=PLAY/'evaluated_models.py';text=p.read_text();start=text.index('    display_order = ');end=text.index('\n    selected = {r["key"]',start)
 order=ast.literal_eval(text[start:end].split('=',1)[1].strip());insertion=next(i for i,r in enumerate(order) if r['key'].startswith('queue-regularized_g'))
 order[insertion:insertion]=[{'key':s['registry_key'],'label':s['recipe']+f' · {s["step"]} · 已完成评测'} for s in records if s['arm'].startswith('nzd_v2')][::-1]
 insertion=next(i for i,r in enumerate(order) if r['key'].startswith('grpo-'))
 s=records[-1];order.insert(insertion,{'key':s['registry_key'],'label':s['recipe']+f' · {s["step"]} · dropout 0.15 · 已完成评测'})
 block='''    latest_file = Path(__file__).with_name('latest_models.json')
    if latest_file.exists():
        for record in json.loads(latest_file.read_text()):
            ev = Path(record['evaluation_root']); cp = Path(record['checkpoint'])
            assert (ev/'FOX_PATCH_VERIFIED.json').exists() and (ev/'no_eos/COMPLETED.json').exists()
            assert Path(json.loads((cp.parent/'best.json').read_text())['path']) == cp
            spec = all_models[record['registry_key']]
            assert spec.adapters == tuple(record['adapters']) and spec.base == record['base']
            results = [json.loads(line) for line in (ev/'results'/record['key']/'p1.jsonl').open()]
            assert len(results) == 36 and all(len(r['samples']) == 4 and tuple(r['adapters']) == spec.adapters for r in results)
            selected[record['registry_key']] = spec
'''
 text=text[:start]+block+'    display_order = '+repr(order)+text[end:];compile(text,str(p),'exec');(STAGE/p.name).write_text(text)
 notes='新GRPO更新待权重读取权限：已完成的最新模型为 WJY `v2.1/checkpoint-400`，由其开发集奖励选定；当前权重为600、sunshq114514不可读。表中GRPO仍为旧r1/step300，未将WJY内部指标混入本统一表；取得读取权限并完成同口径评测后替换。v2.2/v2.6在本次检查时尚无完成记录。'
 for p in (PLAY/'results_summary.md',REPO/'README_MODELS_DATA_RESULTS.zh-CN.md'):
  text=p.read_text();out=[]
  for line in text.splitlines():
   if line.startswith('| '):
    fields=[s.strip() for s in line.strip('|').split('|')]
    m=by_recipe.get(fields[0]);s=m['model'] if m else None
    if m and len(fields)==9:
     c=m['completion']['1'];ph=m['physical'];ne=m['no_eos'];step=str(s['step'])+(' · dropout 0.15' if s['arm'].startswith('regularized') else '')
     line='| '+' | '.join([fields[0],step,f'{c["delta"]:+.3f}',f'{c["f1"]["original"]:.4f}',f'{c["f1"]["stretched"]:.4f}',f'{ph["valid_length"]:.2f}',f'{ph["collision_free_length"]:.2f}',f'{ne["no_eos_valid_length"]:.2f}',f'{ne["no_eos_collision_free_length"]:.2f}'])+' |'
    elif m and len(fields)==3:line=f'| {fields[0]} | 已完成训练及允许/禁EOS评测；dev-best={s["step"]} | `{s["checkpoint"]}` |'
   out.append(line)
  text='\n'.join(out)+'\n';text=re.sub(r'2026-11-21 00:39 PST','2026-11-22 PST',text)
  text=text.replace('当前下一组是weight_decay=0.10＋dropout=0.15。','weight_decay=0.10＋dropout=0.15已完成，使用开发集最佳step1000。')
  text=text.replace('Qwen + nzd v2 plain/masked（进行中）','Qwen + nzd v2 plain/masked（已完成）').replace('Qwen + ourdata + Full_G + AdamW .10（排队）','Qwen + ourdata + Full_G + AdamW .10（已完成）')
  text=text.replace('新v2结果出来前保留原始标签。','历史结果保留原始标签；当前v2结果已单列。')
  text=text.replace('未完成条目留空，不用历史v4结果代替v2。','v2两组和更强正则化均已完成；不以历史v4替代v2。')
  text=text.replace('队列顺序：nzd v2 plain → nzd v2 masked → 新Full-G wd.10/dropout.15，每组训练后自动做允许/禁止EOS评测。先前禁止EOS优先任务已经完成，v2训练已恢复。实际新增的是weight_decay=.10、dropout=.15。','三个排队任务及其允许/禁止EOS评测均已完成。训练耗时分别为4.21、2.86、4.22小时，总计11.29小时（不含评测）。开发集最佳checkpoint依次为2211、2211、1000。')
  text=text.replace('下拉框现在是15个已完成评测的模型','下拉框现在是18个已完成评测的模型').replace('仍按15个模型分组。12个PT','按18个模型分组。15个PT')
  text=text.replace('真正失败会显示“部分可解析”或“解析失败”','真正失败统一显示“解析失败”')
  text+='\n### 2026-11-22 新结果与GRPO更新状态\n\n'+notes+'\n\n新三组已加入“给第一块”和“从零生成”的模型下拉框。选“Qwen3 直接 PT · 对照 · 2211”“Qwen3 直接 PT · 增强 · 2211”或“Full-G 正则化 · WD0.10 / dropout0.15 · 1000”。每模型144个补全结果，独立生成按允许/禁止EOS各展示4次；狐狸置顶，输入已去重。\n\n[最新三组与官方PT渲染对比](/viewer/latest_compare.html?source=114&prime=1)。训练曲线仅保存在集群 `runs/latest_results_20261122/delivery/`，不嵌入网页。\n'
  (STAGE/p.name).write_text(text)
 expected={str(p):sha(p) for p in [PLAY/'backend.py',PLAY/'evaluated_models.py',PLAY/'results_summary.md',REPO/'README_MODELS_DATA_RESULTS.zh-CN.md']}
 (ROOT/'SERVICE_STAGED.json').write_text(json.dumps({'source_sha256':expected,'records':records},ensure_ascii=False,indent=2));print('Service and README staged',flush=True)
if __name__=='__main__':main()
