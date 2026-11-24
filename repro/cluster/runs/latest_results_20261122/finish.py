import json,os,shutil,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=ROOT/'delivery';PLAY=Path('/home/sunshq114514/brickicl-gradio-service/brickicl/playground');LIVE=Path('/home/sunshq114514/brickicl-viewer/baseline/repro/viewer')
rows=json.loads((OUT/'metrics.json').read_text())
text='# 2026-11-22：已完成训练结果与GRPO待办\n\n三个Qwen模型都已完成训练、统一补全与允许/禁止EOS独立生成评测。仅给第一块，18案例×正常/加长×4次；独立生成各64次。狐狸输入采用去重后的版本。\n\n| 模型 | best step | Δlog-odds | 正常F1 | 加长F1 | 有效长度 | 无碰撞长度 | 禁EOS有效长度 | 禁EOS无碰撞长度 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for m in rows:
 c=m['completion']['1'];p=m['physical'];n=m['no_eos']
 text+=f'| {m["model"]["label"]} | {m["model"]["step"]} | {c["delta"]:+.3f} | {c["f1"]["original"]:.4f} | {c["f1"]["stretched"]:.4f} | {p["valid_length"]:.2f} | {p["collision_free_length"]:.2f} | {n["no_eos_valid_length"]:.2f} | {n["no_eos_collision_free_length"]:.2f} |\n'
text+='\n对照/增强严格使用nzd v2配对数据，70,722训练、2,133开发，Qwen3-0.6B原始基座＋LoRA r64/alpha128/dropout.05，AdamW wd.01，lr2e-4，有效batch32，一轮2211步。更强正则化使用相同Full-G结构修改70k数据、LoRA dropout.15、AdamW wd.10，其他Qwen参数相同，共2188步，dev-best1000。\n\n训练耗时4.21+2.86+4.22=11.29小时，不含评测。新Full-G最低val CE=0.26465，末步0.26933，仍有轻微回升；其独立生成表现弱于旧wd.05/dropout.10，不能笼统认为越强正则化越好。\n\n## GRPO状态\n\n最新完成并有best.json的是 `/home/jywu2289/RL/runs/v2.1/train/grpo/checkpoint-400`，原开发集477题平均reward=0.315297，起点仍为官方PT＋nzd增强step2000。这是WJY内部开发结果，不是本表的18案例统一结果。新v2.2/v2.6检查时无完成记录。\n\n权重文件600，当前账号无法读取，未执行新GRPO推理、未伪造结果或冒充替换旧展示。请文件所有者执行：\n\n```bash\nchmod 644 /home/jywu2289/RL/runs/v2.1/train/grpo/checkpoint-400/adapter_model.safetensors\n```\n\n权限恢复后同口径评测入口（sunshq114514执行）：\n\n```bash\nbash /home/sunshq114514/brickICL/runs/grpo_v21_eval_20261122/run_eval.sh\n```\n\n该脚本生成→评分→验证→狐狸去重输入补评→禁止EOS生成/评分。完成后仍需把新GRPO接入注册表、覆盖旧GRPO渲染与表格；脚本不会自动发布未验证结果。原始旧GRPO评测文件保留供追溯。\n\n## 网页\n\nGradio与Viewer服务均在集群；本机只建立SSH隧道。`http://127.0.0.1:17860/` 与 `http://127.0.0.1:17860/viewer/`。新三组与官方PT比较页：`/viewer/latest_compare.html?source=114&prime=1`。Viewer现18个已评测模型（含3个历史SFT），Gradio只列15个已评测PT模型。曲线只存此delivery目录，不放网页。\n'
(OUT/'RESULTS.zh-CN.md').write_text(text)
access=REPO/'README_LOCALHOST_ACCESS.zh-CN.md'
if access.exists():
 s=access.read_text().replace('15个已完成评测的模型','18个已完成评测的模型').replace('仍按15个模型分组。12个PT','按18个模型分组。15个PT')
 s+='\n更新（2026-11-22）：已加入Qwen对照2211、增强2211及Full-G正则化WD.10/dropout.15 step1000。主Viewer共18模型，Gradio默认15个PT模型。新GRPO v2.1/400尚待权重读取权限；旧GRPO未冒充为新结果。\n';access.write_text(s)
for src in (REPO/'README_MODELS_DATA_RESULTS.zh-CN.md',access):
 if src.exists():shutil.copy2(src,OUT/src.name)
targets=[ROOT,REPO/'runs/grpo_v21_eval_20261122',LIVE/'models'/ROOT.name]
for root in targets:
 for current,dirs,files in os.walk(root,followlinks=False):
  for p in [Path(current)]+[Path(current)/x for x in files]:
   st=p.lstat()
   if st.st_uid==os.getuid() and not stat.S_ISLNK(st.st_mode):p.chmod(0o775)
for p in [access,REPO/'README_MODELS_DATA_RESULTS.zh-CN.md',PLAY/'latest_models.json']:
 if p.exists():p.chmod(0o775)
shutil.make_archive(str(ROOT/'latest_results_20261122'),'zip',OUT)
print('Report, access README, 775 permissions, delivery ZIP ready',flush=True)
