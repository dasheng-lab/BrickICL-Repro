# BrickNet 数据目录（`BRICKNET_DATA`）

`BRICKNET_DATA` 指向本目录（`baseline/bricknet/data`），`baseline/repro/env.sh` 会根据仓库位置自动设置。`bricknet` 和 `bricknet-render` 都从这里找网格。
完整数据集须通过 BrickNet 作者提供的[申请表](https://forms.gle/dm4eYSa5gh4DqzRT6)申请。得到授权后，在本机设置 `BRICKNET_SHARE_URL`，再运行 `../../repro/fetch_data.sh`（只下载数据集）或加上 `--meshes`（同时下载 ldraw/inset/watertight）。不要提交授权分享地址或下载的数据。cvx/ 和 glb/ 可用上游命令 `python -m bricknet fetch-meshes`、`python -m bricknet_render fetch-glbs` 获取。

| 路径 | 来源（共享里的文件） | 大小 | 用途 |
|---|---|---|---|
| `cvx/` | `collider.tar.xz`（`bricknet fetch-meshes`） | 1.0 GB，14,462 个 `.cvx.obj` | 碰撞检测 v0.1.1：凸分解 + Bullet 穿透深度（当前代码默认使用） |
| `inset/` | `inset.tar.xz` | 1.5 GB，21,084 个 `.ply` | 碰撞检测 v0.1.0：内缩 0.25 LDU 的三角网格，**论文中的碰撞数字基于它** |
| `watertight/` | `watertight.tar.xz` | 450 MB，6,145 个 `.ply` | 未做凸分解的封闭网格，附带 `README.md`、`manifest.tsv`、`SHA256SUMS` |
| `glb/` | `glb.tar.xz`（`bricknet_render fetch-glbs`） | 1.9 GB | 渲染网格（BrickNet-Render 使用） |
| `ldraw/` | `ldraw.tar.xz` | 593 MB | 作者的 LDraw 零件库快照，3D 查看器从这里读零件 |
| `dataset/` | 见下表 | 13 GB（全部为 xz 压缩包） | 训练和评测数据集 |

## `dataset/`

文件保持原始的 `.xz` 压缩格式：`scripts/train.py` 能直接读取 `.jsonl.xz`。
`.npz.xz` 需要先解压到别处再用 `bricknet.load_graphs` 读取，例如：
`xz -dkc dataset/val.npz.xz > /tmp/val.npz`。不要把解压后的文件放回这个目录，`pt.npz` 解压后有 6.6 GB。
字段的具体含义见 `../BrickNet/DATA.md`。

| 文件 | 压缩后大小 | 内容 | 已校验 |
|---|---|---|---|
| `paths_pt.jsonl.xz` | 11.2 GB | 预训练用的搭建序列（path text），32,572,697 行 | 行数与 DATA.md 一致 |
| `paths_sft.jsonl.xz` | 744 MB | SFT 用的搭建序列，5,912,934 行 | 行数与 DATA.md 一致 |
| `paths_val.jsonl.xz` | 5.4 MB | val 搭建序列，45,516 行 | 行数与 DATA.md 一致 |
| `captions_sft.jsonl.xz` | 20.8 MB | 每个模型 5 条描述，共 335,925 行 | 行数与 DATA.md 一致 |
| `captions_val.jsonl.xz` | 24 KB | 512 条评测描述（Table 3 的 prompt） | 行数与 DATA.md 一致 |
| `pt.npz.xz` | 625 MB | 预训练用的连接图（CSR）：253,623 个图 / 38,775,582 个零件 / 125,816,010 条边，解压后 6.6 GB | 与 DATA.md 一致 |
| `sft.npz.xz` | 18.3 MB | SFT 用的连接图：67,185 个图 / 1,774,387 个零件 / 4,124,165 条边 | 与 DATA.md 一致 |
| `val.npz.xz` | 155 KB | val 连接图：512 个图 / 13,397 个零件 / 32,547 条边 | 与 DATA.md 一致，`load_graphs` 可读 |

每个文件的字节数都和共享列表核对过，并且全部通过了 `xz -t`（CRC64）校验。
