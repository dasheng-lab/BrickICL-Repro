# 0.6B 后训练与统一首块评测

本目录包含已经运行过的 M-only / Full-G PT、SFT 后训练代码，以及目前使用的统一评测。训练和推理在 GPU 集群运行；本机只需 Git 和 SSH。碰撞检查目前用于数据验证、评分和展示，**生成时的碰撞撤回／重试尚未实现**。

## 已完成的实验

| 实验 | 初始化 | 训练输入 | 监督范围 | 已运行实验的 dev-best |
|---|---|---|---|---:|
| M-only PT | Qwen3 + 官方 BrickNet-PT | 已搭主体及可见参照 | 缺失 M + EOS | 1750 |
| M-only SFT | Qwen3 + 官方 PT + 官方 SFT | 同上，不加 caption | 缺失 M + EOS | 2188 |
| Full-G PT | Qwen3 + 官方 BrickNet-PT | 节点编号 a，无给定积木 | 整个 G 的后续 tokens + EOS | 500 |
| Full-G SFT | Qwen3 + 官方 PT + 官方 SFT | 作者 caption 和必要编辑说明 | 整个 G + EOS，prompt 不计 loss | 250 |

共同配方为 LoRA r=16、alpha=32、dropout=0.05，AdamW 学习率 1e-4，微批4、累积8，有效批32；每组70,000次抽样、v2/v3各50%、一轮2188步。每例先对监督 tokens（含 EOS）平均，再对样本平均。Full-G SFT使用有caption的子集，所以不是只改变loss范围的单变量实验。

权重按开发集选择，测试仅报告结果；没有把所有checkpoint按test分数重新排名。历史运行记录中的代码哈希属于当时的执行版本；整理后的入口改变时，请重新生成新运行的配置，不要绕过哈希检查恢复旧任务。

## 环境与产物

需要已配置 CUDA 的 `torch`、`transformers`、`peft`、`numpy`、`networkx` 和本仓库的 `bricknet`。数据准备遵循 [数据目录说明](../../baseline/bricknet/data/README.md)。模型只从已有 Hugging Face 缓存加载。

```bash
# 在集群、仓库根目录执行；按实际环境设置。
export BRICKNET_DATA=/path/to/BrickNet/data
export BRICKICL_POSTTRAIN_ROOT=/path/to/experiment-repository
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

`BRICKICL_POSTTRAIN_ROOT` 指向已有训练run和checkpoint的仓库；默认是当前代码仓库。它用于读取模型与缓存，写入位置由各命令的 `--out` / `--run-dir` 指定。

不提交数据集、captions、token缓存、checkpoint、原始推理JSONL、生成MPD/LDraw、日志或运行目录链接。`runs` 即使是符号链接也被忽略。代码和聚合指标摘要可以版本化。

## 数据与训练入口

| 文件 | 职责 |
|---|---|
| `build_dataset.py`、`build_v2.py`、`build_v3.py` | 共享结构指纹及重复结构数据构造 |
| `finalize_splits.py`、`screen_colliders.py`、`screen_v3_body.py`、`finalize_v3_profile.py` | 划分与几何/复杂度筛查 |
| `reference_last.py`、`materialize_reference_last.py` | 按主体→完整参照→待补部分重排，核对位姿，生成训练缓存 |
| `data.py`、`bricknet_runtime.py` | prompt mask、EOS、collator、等权样本loss和新LoRA初始化 |
| `submit_06_plan.py`、`submit_fullg_plan.py` | 固定采样计划、配置及哈希；名字中的submit只生成文件，不提交任务 |
| `train_bricknet06.py`、`train_fullg06.py` | 训练、开发集选优、保存与显式恢复 |

v2强调较大的重复子装配；v3强调复杂主体内6–10件的局部补全。来源与无颜色结构指纹在train/dev/test之间分组隔离，同源增强不跨集合；官方val来源不加入这些后训练数据。原图仍可能被官方预训练见过。

所有入口都支持 `--help`，import不会启动训练或推理。数据构造耗时较长，应在新版本目录运行。以已有reference-last缓存为例：

```bash
python -m brickicl.posttrain.submit_06_plan --data runs/reference_last_06 --out runs/bricknet_06_reference_last
python -m brickicl.posttrain.train_bricknet06 --config runs/bricknet_06_reference_last/PT.json
python -m brickicl.posttrain.train_bricknet06 --config runs/bricknet_06_reference_last/SFT.json

python -m brickicl.posttrain.prepare_fullg --data runs/reference_last_06 --out runs/fullg_06
python -m brickicl.posttrain.submit_fullg_plan --data runs/fullg_06 --out runs/bricknet_06_fullg
python -m brickicl.posttrain.train_fullg06 --config runs/bricknet_06_fullg/PT.json
python -m brickicl.posttrain.train_fullg06 --config runs/bricknet_06_fullg/SFT.json
```

上述命令会训练，只有需要新开实验时才执行。恢复需显式添加 `--resume`，配置/数据/代码哈希必须匹配；不加载test挑checkpoint。

## 当前主评测：同一18来源，只改变首块提示

六个模型为官方、M-only、Full-G各自的PT/SFT。所有模型输入只有装配路径，不额外提供caption或完整示范。正常、加长、删除参照三条件固定，r=4；每个输入采样4次。

- 不给第一块：模型自己生成目标子装配的首块与后续部分。
- 给第一块：相同首块移入输入，只评价剩余生成部分。

两种设置的完整参考文本逐字相同，共用批次、随机种子和绝对token上限。每模型每部分216次，共2592次。18个来源是已探索过的固定案例，不能称为全新泛化测试。删除参照的短答案只是诊断参考。

```bash
# 首次准备无需任何旧推理输出；输入来自原有stretch采样器。
python -m brickicl.posttrain.prepare_first_brick_eval \
  --input /path/to/stretch.contiguous.r4.jsonl \
  --out runs/history_first_brick06

python -m brickicl.posttrain.evaluate_first_brick --run-dir runs/history_first_brick06
python -m brickicl.posttrain.report_first_brick --run-dir runs/history_first_brick06
```

若希望复用旧p1结果，在准备命令中加 `--reuse-existing`，必要时指定 `--legacy-eval` 和 `--fullg-eval`。只有输入文本、checkpoint、seed、token上限、ID和4次采样全部匹配才复用，否则重新评测。重跑评测命令会按完整批次恢复，未完成的批次整体重跑，保持随机计划。

`python -m brickicl.posttrain.evaluate_first_brick --run-dir ... --validate-only` 只检查完整性，不加载模型。旧的 `evaluate_posttrain.py` 保留共享评分函数和历史复现入口；新的主流程使用上述命令。

Δlog-odds用长/短参考答案的整段对数概率（含EOS）相减，再取加长−正常的变化。共有首块的概率抵消，所以两种首块设置的Δ通常相同。首块提示的效果主要看自由生成F1、解析率及停止行为。

## Viewer 与 Gradio

```bash
python -m brickicl.posttrain.pack_first_brick_viewer --run-dir runs/history_first_brick06
python -m brickicl.posttrain.publish_first_brick_viewer --run-dir runs/history_first_brick06
python -m http.server 18765 --bind 127.0.0.1 --directory baseline/repro/viewer
```

`unified_compare.html` 通过案例、是否给首块、三条件和采样编号选择结果，每次显示六个模型。图中保留失败与过度生成样本，不按效果筛选。需要发布到另一目录时，用 `--viewer-dir`；覆盖索引前会备份，原MPD不删除。

可选双副本展示使用已有历史缓存：

```bash
python -m brickicl.posttrain.compact_double_copy \
  --legacy-eval /path/to/posttrain_eval06 \
  --run-dir runs/history_first_brick06
```

它只展示狐狸、出租车、桌子/拱门、电线杆四个来源，加长条件各前4次，合计16个输出。额外诊断不进入展示目录。

[Gradio说明](../playground/README.md) 覆盖默认六个已评测模型、108个统一输入、LDraw上传与SSH转发。两个Viewer采用按需绘制；主对照页切换时复用六个iframe。测试不需要本机浏览器。

聚合结果见 [RESULTS.zh-CN.md](RESULTS.zh-CN.md)。从零生成的旧复现工具仍单独保留，未混入当前统一补全主表。碰撞撤回实验仅列后续工作，尚未执行。
