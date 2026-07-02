# Offline R-STAMP Experiment Scaffold

> 这个目录是给无网服务器准备的离线实验包。  
> 当前目标：在已经下载好数据的前提下，整理数据、放入 STAMP 原始代码和 R-STAMP 新 idea 代码骨架，然后在 2×80GB GPU 上逐步跑实验。

## 目录角色

建议最终服务器目录长这样：

```text
/data/MLLM-SEG/                       # 已下载数据根目录
├── shared/
├── annotations/
├── datasets/
└── download_status.json

/work/MLLM-SEG-exp/                   # 实验工作区
├── code/
│   ├── STAMP/                         # 手动放入官方 STAMP 代码
│   └── R-STAMP/                       # 本目录提供的新 idea 代码骨架
├── data/                              # 指向 /data/MLLM-SEG 的软链接
├── models/                            # 手动放入模型权重
├── outputs/
└── logs/
```

本仓库里的 `offline_rstamp/` 负责提供脚本和配置；真正训练时建议把 STAMP 官方代码放进工作区的 `code/STAMP/`。

## 为什么数据还要处理

是的，数据下载完通常还不能直接训练。当前 `MLLM-SEG-data` 下载的是 raw dataset / annotation mirror，后续还需要：

1. 检查目录是否完整；
2. 按 STAMP 代码要求建立路径；
3. 运行 STAMP 自带的数据 JSON 生成脚本，例如：

```bash
python STAMP/data/create_refcoco_new.py
```

4. 如果做 R-STAMP，还要在 STAMP 的训练样本上附加结构化 reasoning prior，例如 target、attribute、relation、bbox、positive point、negative point 等字段。

本目录提供的是离线辅助脚本；具体 STAMP 内部数据格式仍以你放入的 STAMP 官方代码为准。

## 一次完整离线流程

如果你使用当前服务器布局，请先读：

```bash
cat offline_rstamp/CURRENT_SERVER_LAYOUT.md
```

并检查：

```bash
bash offline_rstamp/run/20_check_current_layout.sh
```

### 0. 编辑路径

```bash
cd MLLM-SEG-data
cp offline_rstamp/paths.example.sh offline_rstamp/paths.local.sh
vim offline_rstamp/paths.local.sh
```

至少修改：

- `MLLM_SEG_DATA_ROOT`
- `MLLM_SEG_WORKSPACE`
- `STAMP_CODE_DIR`
- `RSTAMP_CODE_DIR`
- `MODEL_ROOT`
- `SAM_CKPT`

### 1. 检查数据并整理工作区

```bash
bash offline_rstamp/run/00_prepare_workspace.sh
```

它会：

- 检查 COCO、RefCOCO、gRefCOCO、ReasonSeg 等目录；
- 生成 `data_layout_report.json`；
- 建立 `workspace/data` 软链接；
- 建立 `workspace/code`、`workspace/models`、`workspace/outputs`、`workspace/logs`。

### 2. 放入两个代码目录

因为当前方案基于 STAMP 修改，建议放两个目录：

```text
code/STAMP/      # 官方 STAMP 原始代码
code/R-STAMP/    # 我们的新 idea scaffold
```

如果服务器没网，请在有网机器下载或 clone 后压缩，再拷贝到服务器解压。

官方 STAMP：

```bash
git clone https://github.com/HKUST-LongGroup/STAMP.git
```

R-STAMP scaffold 可以由本目录安装：

```bash
python offline_rstamp/scripts/install_rstamp_scaffold.py \
  --target-code-dir "$RSTAMP_CODE_DIR"
```

### 3. 检查代码仓库是否放对

```bash
python offline_rstamp/scripts/check_code_repos.py \
  --code-root "$MLLM_SEG_WORKSPACE/code"
```

它不会联网，只检查本地目录、git remote 和关键脚本是否存在。

### 4. 先跑 STAMP baseline

先不要急着改模型。第一步应该确认 STAMP 原始代码能在你的数据和模型上跑通。

```bash
bash offline_rstamp/run/01_stamp_eval.sh
```

如果 STAMP 官方脚本路径或参数和你下载的版本不同，优先修改这个 wrapper，而不是到处手敲命令。

### 5. 跑 STAMP LoRA/SFT baseline

```bash
bash offline_rstamp/run/02_stamp_train_2x80g.sh
```

这个 wrapper 默认调用 STAMP 官方训练脚本。如果官方脚本需要改数据路径、模型路径、batch size，请在 wrapper 顶部集中改。

### 6. 安装并接入 R-STAMP scaffold

```bash
bash offline_rstamp/run/10_install_rstamp_scaffold.sh
```

这一步会把 `rstamp_src/rstamp` 复制到 R-STAMP 代码目录中。真正接入 STAMP 时，需要在 STAMP 的 model/trainer/data pipeline 中调用这些模块。

### 7. R-STAMP 推荐实验顺序

先做最稳的：

```text
STAMP baseline
→ STAMP + structured reasoning prior + LoRA SFT
→ STAMP + prior + uncertainty refinement
→ STAMP + prior + lightweight mask-aware RL
```

2×80GB 下不要一开始就上完整 RL。先把 SFT 的增益做出来。

## 推荐报告结果

至少报告：

- RefCOCO / RefCOCO+ / RefCOCOg；
- ReasonSeg；
- gRefCOCO 或 GRES；
- inference speed；
- GPU memory；
- complex query / OOD / small object / ambiguity split。

论文故事不要写成“靠更多数据/更多卡刷榜”，而是写成：

> Resource-efficient reasoning enhancement for parallel MLLM segmentation.
