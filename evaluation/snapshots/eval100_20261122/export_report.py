import csv,json,shutil,os,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=ROOT/'delivery';OUT.mkdir(exist_ok=True)
def main():
 assert (ROOT/'LIVE_VERIFIED.json').exists();metrics=json.loads((OUT/'metrics.json').read_text());rows=[]
 for arm,m in metrics.items():
  c=m['completion']['1'];p=m['physical'];n=m['no_eos'];s=m['model']
  rows.append({'model':s['recipe'],'arm':arm,'step':s.get('step'),'sources':100,'delta_log_odds':c['delta'],'normal_pose_f1':c['f1']['original'],'stretched_pose_f1':c['f1']['stretched'],'valid_length':p['valid_length'],'collision_free_length':p['collision_free_length'],'no_eos_valid_length':n['valid_length'],'no_eos_collision_free_length':n['collision_free_length'],'natural_coverage':p['collision_coverage'],'no_eos_coverage':n['collision_coverage']})
 with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for name in ['DATASET_VALIDATED.json','RENDER_VERIFIED.json','VISUAL_APPROVED.json','SCORING_SELF_CHECK.json','BODY_INVARIANCE.json','TRAINING_OVERLAP_AUDIT.json','LIVE_VERIFIED.json','plan.json','synthetic_sources.json']:
  shutil.copy2(ROOT/name,OUT/name)
 for src in (ROOT/'output/playwright').glob('*.png'):shutil.copy2(src,OUT/src.name)
 shutil.copy2(REPO/'README_MODELS_DATA_RESULTS.zh-CN.md',OUT/'README_MODELS_DATA_RESULTS.zh-CN.md')
 (OUT/'README.zh-CN.md').write_text('本次评测：100个不同几何作品，18原案例＋82新构造作品。12种主体搭建规则、8种重复部件组合，含向上/向下连接。没有用采样顺序、颜色或同一作品的不同mask扩充作品数；正常/加长属于每作品内对照。\n\n15个PT模型各800次补全；允许/禁止EOS各64次重新生成。结果见metrics.csv和results_summary.md。原checkpoint均按开发集选取；未根据本测试集选点。\n\n构造与推理完整代码在 '+str(ROOT)+'。100组输入/标准答案均有Viewer渲染记录。新增作品与原自然作品存在分布差异，12种程序化设计规则不是82种独立语义类别。\n\n新GRPO v2.1/400不可读，本表GRPO仍为旧r1/300。\n')
 shutil.make_archive(str(ROOT/'eval100_results'),'zip',OUT)
 for current,dirs,files in os.walk(ROOT,followlinks=False):
  for p in [Path(current)]+[Path(current)/x for x in files]:
   s=p.lstat()
   if s.st_uid==os.getuid() and not stat.S_ISLNK(s.st_mode):p.chmod(0o775)
 (ROOT/'DELIVERY_READY.json').write_text(json.dumps({'state':'ready','zip':str(ROOT/'eval100_results.zip'),'models':15,'sources':100}));print('Final 100-case results bundle ready',flush=True)
 live=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
 for root in [live/'models'/ROOT.name,live/'models/meta',live/'models/groups']:
  if not root.exists():continue
  for current,dirs,files in os.walk(root,followlinks=False):
   for p in [Path(current)]+[Path(current)/x for x in files]:
    s=p.lstat()
    if s.st_uid==os.getuid() and not stat.S_ISLNK(s.st_mode):p.chmod(0o775)
if __name__=='__main__':main()
