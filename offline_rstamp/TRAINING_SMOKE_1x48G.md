# 1×48GB Smoke Training Guide

目标：先用一张 48GB GPU 做快速可行性验证，不追求正式 SOTA。

## 重要原则

不要运行官方：

```bash
scripts/launch_all_7B.sh
scripts/launch_all_2B.sh
```

这些是多机模板，而且会重新 `pip install -r requirements.txt`，容易破坏当前 torch 环境。

## 运行顺序

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
git pull origin main
find offline_rstamp -name "*.sh" -exec sed -i 's/\r$//' {} \;

bash offline_rstamp/run/43_check_smoke_ready.sh
bash offline_rstamp/run/40_patch_stamp_local_training.sh
bash offline_rstamp/run/44_fix_stamp_imports.sh
bash offline_rstamp/run/45_force_disable_flash_attn.sh
```

先跑 baseline smoke：

```bash
STAMP_MAX_SAMPLES=1000 \
MODEL_NAME=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/models/STAMP-2B-uni \
bash offline_rstamp/run/41_train_baseline_1x48g_smoke.sh
```

再跑 R-STAMP smoke：

```bash
STAMP_MAX_SAMPLES=1000 \
MODEL_NAME=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/models/STAMP-2B-uni \
bash offline_rstamp/run/42_train_rstamp_1x48g_smoke.sh
```

如果没有 `STAMP-2B-uni`，但有 `STAMP-7B-lora`，可以把 `MODEL_NAME` 改成：

```bash
MODEL_NAME=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/models/STAMP-7B-lora
```

但 7B 在 48GB 上更容易 OOM，建议先用 2B。

## 显存不够时

逐步降低：

```bash
STAMP_MAX_SAMPLES=200
STAMP_GRAD_ACCUM=16
STAMP_LORA_R=8
STAMP_MAX_LENGTH=1024
STAMP_ATTN_IMPL=eager
STAMP_DISABLE_CUDNN=1
STAMP_MAX_PIXELS=100352
```

例如：

```bash
STAMP_MAX_SAMPLES=200 STAMP_LORA_R=8 STAMP_MAX_LENGTH=1024 STAMP_ATTN_IMPL=eager \
bash offline_rstamp/run/41_train_baseline_1x48g_smoke.sh
```

## 当前 R-STAMP smoke 的含义

这个版本只是把 `structured_prior_text` 注入 prompt：

```text
Structured reasoning prior: <TARGET> ... </TARGET> <BOX> ... </BOX> <POS> ... </POS>
Please segment ...
```

它用于快速验证“显式 structured prior 是否可能有帮助”。它不是最终的 prior-fusion 架构。
