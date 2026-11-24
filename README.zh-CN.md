# BrickICL：LEGO 装配中的上下文学习

论文各章节、图表与仓库文件的对应关系见 [PAPER_MAP.md](PAPER_MAP.md)。

CS182 项目，作者：Sixian Shen、Zidong Ning、Shengqing Sun、Junyang Wu。[English](README.md)

我们研究 0.6B 装配模型能否利用已出现的重复子装配补全另一份，以及这种能力与无参照生成的关系。主评测使用 BrickNet 官方验证集的 100 个来源，比较原始与加长的可见参照；历史 18 来源结果单独保留。

## 复现论文数值图表

`repro/report/` 提供去标识化的逐案例数值观测和分析代码，不含 BrickNet 原始图、caption、搭建路径、LDraw 场景或验证集渲染模型。

```bash
cd repro/report
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

该命令无需 GPU、模型权重或数据集下载，即可重建定量表格、图和 bootstrap 区间。定性渲染图需要获准访问 BrickNet 数据后另行生成。输入、输出及校验见 `repro/report/README.md`。

## 数据申请与最小测试样例

BrickNet 的完整训练集和验证集并非公开下载资源。请按 [BrickNet 数据说明](baseline/bricknet/BrickNet/DATA.md)中的[申请表](https://forms.gle/dm4eYSa5gh4DqzRT6)向作者申请。获得授权后，在本机设置 `BRICKNET_SHARE_URL` 并运行 `baseline/repro/fetch_data.sh`。不要把授权分享地址、原始数据或派生几何提交到仓库。100 来源基准的原始输入也因尚无再分发许可而不包含在公开包中。

`repro/report/data/smoke_example.jsonl` 是独立编写的小型装配样例，不属于 BrickNet 验证集，也不参与论文分数。安装 BrickNet 和公开碰撞网格后，运行 `python repro/report/scripts/smoke_example.py` 可检查解析、连接与碰撞评分；加 `--tokenizer MODEL_OR_PATH` 可检查补全损失掩码。

## 代码与证据范围

`brickicl/posttrain/` 包含重复模块数据构造、M-only/Full-G 监督、LoRA 训练和相关评测工具。`brickicl/sampling/`、`brickicl/eval/` 保存采样器与早期受控探针。`repro/report/scripts/` 是报告统计与绘图代码，`repro/report/repro/evaluation/` 是 New100 数据构造和评分源码快照；后者需要获准取得原始数据和模型才能完整运行。

报告的 B-RL 来自 GRPO v2.1 运行。仓库 `RL/step9_grpo/grpo.py` 和 `RL/common/rewards.py` 的 SHA-256 均与训练记录一致。重新训练仍需获准取得 BrickNet 派生输入、模型权重和兼容的 GPU 环境；仅重建论文定量图表无需这些资源。

集群源码扫描清单见 [repro/cluster/CLUSTER_AUDIT.md](repro/cluster/CLUSTER_AUDIT.md)：补入了 100 来源评测、GRPO 评测、无 EOS 生成及各训练对照的代码快照；运行计划、原始输入、生成装配和模型权重不在公开仓库中。

主评测为 100 来源严格 DFS 序列；历史 18 来源使用不同来源集合。检查点根据开发集选择。本仓库不公开受限数据、原始生成文本或私人分享地址，也不声称完成多随机种子训练或物理稳定性评测。
