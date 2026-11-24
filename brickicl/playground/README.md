# BrickICL Gradio Playground

在GPU集群提供交互式补全与生成，本机通过SSH转发访问。模型、数据和生成文件保留在服务器；本机不运行模型推理。

## 配置与启动

激活已安装 `gradio`、`torch`、`transformers`、`peft`、`bricknet` 的环境，在仓库根目录执行：

```bash
export BRICKNET_DATA=/path/to/BrickNet/data
export BRICKICL_POSTTRAIN_ROOT=/path/to/experiment-repository
bash brickicl/playground/cluster_service.sh
```

默认监听 `127.0.0.1:17860`。本机只建立转发：

```powershell
.\brickicl\playground\connect.ps1 -SshHost your-cluster -NoBrowser
```

或者执行 `ssh -N -L 17860:127.0.0.1:17860 your-cluster` 后手动访问 <http://127.0.0.1:17860/>。不要自动启动多批本地3D浏览器验证。

| 环境变量 | 含义 |
|---|---|
| `BRICKICL_POSTTRAIN_ROOT` | 含训练run、最佳checkpoint及评测结果的仓库；默认当前仓库 |
| `BRICKNET_DATA` | 包含dataset、ldraw、cvx的数据根目录 |
| `BRICKICL_UI_PYTHON` | 服务使用的Python，默认当前环境的python |
| `BRICKICL_UI_HOST` / `BRICKICL_UI_PORT` | 默认127.0.0.1 / 17860 |
| `BRICKICL_PLAYGROUND_OUTPUT` | 导出目录，默认当前仓库runs/playground |
| `BRICKICL_SERVICE_ID` | 健康检查标识，默认brickicl-playground；连接脚本可传`-ServiceId` |
| `BRICKICL_GPU_MEMORY_FRACTION` | 默认0.25，限制本进程GPU显存比例 |
| `BRICKICL_AUTO_UNLOAD` | 默认1，请求结束卸载模型；设0可保留缓存 |

服务串行处理GPU请求，不自动申请集群资源，不自动下载权重。`/health` 返回实例标识和可用模型/样例数。使用其他端口时同步修改SSH转发。

## 已完成评测和默认模型

主表仅分“不给第一块”和“给第一块”，来自相同18个案例、三条件、固定种子与token预算。报告读取自 `$BRICKICL_POSTTRAIN_ROOT/runs/history_first_brick06/reports/GRADIO_SUMMARY.zh-CN.md`；缺少报告时提示尚不可用，不显示旧的静态数字。

默认模型列表保留官方0.6B PT/SFT，以及四个 **dev-best且完成统一test eval** 的后训练权重。注册时检查best.json、训练完成标记、provenance中的adapter路径及两套评测的完整记录。没有对所有checkpoint按test排名。“显示全部checkpoint／其他尺寸”可打开完整缓存列表。

默认输入库只有统一测试的108个输入，默认选狐狸加长、不给首块。文字caption为空，参考答案只供预览，不会自动输入模型。现场生成采用用户当前采样参数，不保证逐字复现离线批次中的样本。

## 输入、输出与3D

- 可选择统一输入，直接编辑path，或上传TXT、JSON/JSONL、LDraw `.ldr` / 自包含`.mpd`。
- LDraw导入保留全部零件；断开结构、循环引用或无法映射的零件会明确报错，不静默截断。
- 自然语言生成按官方约定分别编码caption和换行，不套聊天模板。无条件PT只输入节点编号`a`。
- 下载包含原始path、LDraw、内嵌零件的MPD和生成参数/解析状态JSON。解析失败也保留原文，不把部分3D当完整成功。
- 3D仅在视角、步骤、样例或播放状态改变时绘制；静止、隐藏或屏幕外暂停。它不等于物理稳定性验证。

旧额外诊断不进入主表。原始Qwen或交叉输入方式仅是手动交互能力，不作为默认评测结论。

## 检查

在集群、已配置数据和模型缓存的环境运行：

```bash
python -m unittest brickicl.playground.test_backend -v
```

浏览器之外的渲染调度与iframe复用检查见 [TESTING.md](../../TESTING.md)。源代码整理不会自动重启任何既有服务；部署与启动需显式执行。
