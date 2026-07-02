# Current Server Layout

更新时间：2026-07-02

当前服务器约定目录：

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/
├── code/
│   ├── STAMP/          # 官方 STAMP baseline 代码
│   └── R-STAMP/        # 我们的新方法 scaffold，需运行安装脚本生成
├── data/               # 唯一保留的数据目录，canonical data root
│   ├── annotations/
│   │   ├── grefcoco/
│   │   └── refcoco_family/
│   ├── datasets/
│   │   ├── reasonseg/
│   │   └── refclef_referit/
│   └── shared/
│       └── coco/
│           ├── annotations/
│           └── train2014/
├── MLLM-SEG-data/      # 工具/脚本 Git 仓库；不应该长期保存大数据副本
├── models/
├── outputs/
└── logs/
```

以后所有训练/转换脚本统一使用：

```bash
export MLLM_SEG_ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
export DATA_ROOT="${MLLM_SEG_ROOT}/data"
export STAMP_CODE_DIR="${MLLM_SEG_ROOT}/code/STAMP"
export RSTAMP_CODE_DIR="${MLLM_SEG_ROOT}/code/R-STAMP"
export MODEL_ROOT="${MLLM_SEG_ROOT}/models"
export OUTPUT_ROOT="${MLLM_SEG_ROOT}/outputs"
export LOG_ROOT="${MLLM_SEG_ROOT}/logs"
```

## 关于 `MLLM-SEG-data/data`

如果 `MLLM-SEG-data/data` 还存在，通常是因为之前使用了 `rsync` 复制数据到 `MLLM-SEG/data`，旧副本没有删除。

可以删除：

```bash
rm -rf "${MLLM_SEG_ROOT}/MLLM-SEG-data/data"
```

但不要删除整个：

```text
MLLM-SEG-data/
```

因为它是工具仓库，里面包含下载器、扫描脚本、R-STAMP scaffold、运行说明等。

## 当前数据是否可以直接训练

不能直接训练官方 STAMP。

原因：

- 当前 raw 数据已经齐全：COCO 图像、RefCOCO JSONL、gRefCOCO、ReasonSeg；
- 但 STAMP 训练脚本目前硬编码读取 `playground/data/json_files/*.json`；
- `data/create_refcoco_new.py` 又依赖 LISA/REFER pickle 格式，例如 `refs(unc).p`；
- 当前数据是 JSONL/mirror 格式，不是 STAMP 脚本直接需要的中间格式。

因此下一步不是下载新 raw data，而是生成 derived training data：

```text
raw data
→ STAMP-compatible training JSON
→ mask png cache
→ baseline train/eval
→ R-STAMP structured-prior train/eval
```

为了避免 baseline 和 R-STAMP 数据互相覆盖，当前脚本约定分开输出：

```text
code/STAMP/playground/data/json_files_debug/
code/STAMP/playground/data/masks_debug/

code/STAMP/playground/data/json_files_baseline/
code/STAMP/playground/data/masks_baseline/

code/STAMP/playground/data/json_files_rstamp/
code/STAMP/playground/data/masks_rstamp/
```

baseline 训练时使用 `json_files_baseline` 和 `masks_baseline`；R-STAMP 训练时使用 `json_files_rstamp` 和 `masks_rstamp`。不要让两个实验读取同一个 JSON 目录。

## Baseline 与改进方法关系

必须先跑官方 STAMP baseline，然后在同一份 derived data 上跑 R-STAMP 改进。

推荐顺序：

```text
E0: 环境/模型 import 检查
E1: 数据转换：raw RefCOCO JSONL → STAMP-compatible JSON + masks
E2: STAMP baseline：不加 structured prior
E3: R-STAMP SFT：同数据 + structured prior
E4: R-STAMP refinement/RL：后续再加
```

## 新方法代码在哪里

新方法 scaffold 在工具仓库：

```text
MLLM-SEG-data/offline_rstamp/rstamp_src/
```

运行以下命令后，会复制到：

```text
MLLM-SEG/code/R-STAMP/
```

```bash
cd "${MLLM_SEG_ROOT}/MLLM-SEG-data"
bash offline_rstamp/run/10_install_rstamp_scaffold.sh
```

如果不想使用 `paths.local.sh`，可以直接运行：

```bash
python offline_rstamp/scripts/install_rstamp_scaffold.py \
  --target-code-dir "${MLLM_SEG_ROOT}/code/R-STAMP" \
  --overwrite
```
