# Proposal DLO · 100 件配对作品

公开版说明：这里只保存评分与汇总代码。下文提到的 inputs、weights、runtime、logs 和原始模型输出未公开；匿名逐案例 log-odds 数值位于 ../../../reproduction/data/proposal_dlo_100/。

主结果在 results/summary.csv。每行一个 checkpoint，包含 100 件作品的平均 DLO、逐作品 bootstrap 95% 区间和两个条件的平均 log-odds。results/contrasts.csv 每行比较两个 checkpoint，方向为 model 减 reference。

## Proposal DLO 的数学定义

对作品 $i$，令 $c=0$ 表示原始参考，$c=1$ 表示拉伸参考。$C_{ic}$ 是对应的输入前缀。$A_{ic}=(a_{ic,1},\ldots,a_{ic,T_{ic}})$ 是该前缀对应的完整长目标续写（输入行中的 `target_long` 加一个 EOS）。两种条件都为同一个长结构目标打分，各自使用与该前缀相容的序列表示。

固定模型参数 $\theta$，用 teacher forcing 计算整段续写的条件概率：

$$
p_{ic}=P_\theta(A_{ic}\mid C_{ic})
=\prod_{t=1}^{T_{ic}}P_\theta(a_{ic,t}\mid C_{ic},a_{ic,<t}).
$$

前缀只提供条件。求和范围覆盖全部目标 token 和末尾的 EOS，保留完整序列长度效应。`scores/*.csv` 中的 `logp` 对应 $\ln p_{ic}$，`log_odds` 对应 $L_{ic}$：

$$
L_{ic}=\ln\frac{p_{ic}}{1-p_{ic}},\qquad
d_i=L_{i1}-L_{i0},\qquad
\mathrm{DLO}=\frac{1}{100}\sum_{i=1}^{100}d_i.
$$

$d_i>0$ 表示对作品 $i$，拉伸参考提高了模型赋给完整长目标的条件概率。`summary.csv` 的 `mean_dlo` 为上述等权平均；`contrasts.csv` 对同一作品先求两个模型的 $d_i$ 之差，再跨作品平均。区间按作品成对重采样。

## 目录

- inputs/ — 指向原评测计划 plan.json 和配对输入 p1_inputs.jsonl 的符号链接。计划记录 checkpoint 路径、适配器顺序与哈希。
- scripts/evaluate.py — 从两种 context 对应的 target_long 计算完整序列概率（含 EOS）；每个模型写一个 CSV，原子提交，重复执行时跳过已有文件。
- scripts/summarize.py — 对同一作品求拉伸减原始的 log-odds 差，再跨作品平均；按作品重采样 100,000 次，种子 20261123。
- results/scores/ — 12 份逐条件打分表，每份 200 行。包括作品 ID、条件、目标 token 数、整段 log 概率、EOS log 概率和 log-odds。
- weights/official/ — 官方 BrickNet 适配器。其余 checkpoint 按 plan.json 中的共享路径读取。
- runtime/deps/ — 此次运行使用的 PEFT 0.17.1；PyTorch、Transformers 和 NumPy 来自账户下已有的 cs183-hw5 环境。
- logs/ — 原始执行记录和首次下载留下的空 .part 文件。

## 在 brick-h20 上运行

从本目录执行。汇总只使用 CPU。评测已有完整输出，再次执行时会跳过已完成的模型。

```bash
cd ~/brickicl_eval/proposal_dlo_100
~/miniconda3/envs/cs183-hw5/bin/python scripts/summarize.py
PYTHONPATH="$PWD/runtime/deps" TOKENIZERS_PARALLELISM=false ~/miniconda3/envs/cs183-hw5/bin/python scripts/evaluate.py
```
