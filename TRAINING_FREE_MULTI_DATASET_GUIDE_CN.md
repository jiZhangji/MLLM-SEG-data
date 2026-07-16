# Training-Free 多数据集评测说明

## 1. 当前覆盖与新增目标

已经完成的主要实验是 RefCOCOg val/test。下一步优先补齐标准 RefCOCO 系列：

```text
RefCOCO:  val, testA, testB
RefCOCO+: val, testA, testB
```

`testA` 主要包含人物目标，`testB` 主要包含非人物目标。这六个划分可以检验方法是否只
对 RefCOCOg 的长表达有效，以及人物、物体和缺少位置词的表达上能否保持稳定提升。

所有新实验继续使用已经冻结的参数：`n_segments=1024`、`graph_lambda=1.0`，不能根据
这些 test split 的结果重新调参。

## 2. 先运行小规模 smoke test

每个 split 只运行 20 个样本，验证 JSON、模型和输出路径：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && git pull --rebase --autostash https://github.com/jiZhangji/MLLM-SEG-data.git main && STAMP7B_OTHER_EVAL_LIMIT=20 CUDA_DEVICE=0 nohup bash run_training_free_stamp7b_refcoco_family_eval.sh > ../outputs/training_free_stamp7b_refcoco_family_smoke.log 2>&1 < /dev/null &
```

Smoke 输出使用 `limit20` 目录，不会与全量 dumps 混合。

## 3. STAMP-7B 六划分全量评测

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && git pull --rebase --autostash https://github.com/jiZhangji/MLLM-SEG-data.git main && mkdir -p ../outputs && (CUDA_DEVICE=0 nohup bash run_training_free_stamp7b_refcoco_family_eval.sh > ../outputs/training_free_stamp7b_refcoco_family_full.log 2>&1 < /dev/null & echo "Multi-dataset PID: $!")
```

脚本自动完成：

1. 从已下载的 RefCOCO-family 标注生成六个 baseline evaluation JSON 和 GT masks。
2. 使用 STAMP-7B 逐 split 导出或续跑 patch logits dumps。
3. 使用与 RefCOCOg 完全相同的冻结 Training-Free 参数评测。
4. 生成六个 split 的统一 Markdown/JSON 表格。

## 4. 查看日志和状态

```bash
tail -n 50 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/training_free_stamp7b_refcoco_family_full.log
```

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && bash check_training_free_multidataset_status.sh
```

最终汇总：

```text
outputs/training_free_refine_stamp7b_refcoco_family_full_comparison/combined_summary.md
outputs/training_free_refine_stamp7b_refcoco_family_full_comparison/combined_summary.json
```

任务中断后重新执行同一条全量命令即可续跑已有 dumps。

## 5. 只跑指定 split

例如只跑两个 val：

```bash
STAMP7B_OTHER_SPLITS='refcoco_val refcoco+_val' CUDA_DEVICE=0 \
  bash run_training_free_stamp7b_refcoco_family_eval.sh
```

## 6. Text4Seg 跨数据集验证

建议先完成 STAMP-7B 六划分，确认趋势后再运行较慢的 Text4Seg。以下命令运行
Text4Seg 的 RefCOCO val：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && TEXT4SEG_EVAL_JSON=../code/STAMP/playground/data/json_eval_baseline/refcoco_val.json TEXT4SEG_RESULTS_ROOT=../outputs/text4seg_official_refcoco_val TEXT4SEG_REFINE_OUTPUT=../outputs/text4seg_training_free_refcoco_val TEXT4SEG_SAM_PATH=../models/SAM/sam_vit_h_4b8939.pth CUDA_DEVICE=0 nohup bash run_text4seg_training_free_eval.sh > ../outputs/text4seg_training_free_refcoco_val.log 2>&1 < /dev/null &
```

RefCOCO+ val 只需对应替换为：

```text
JSON:    json_eval_baseline/refcoco+_val.json
results: text4seg_official_refcocoplus_val
refined: text4seg_training_free_refcocoplus_val
```

Text4Seg 是自回归 7B 推理，耗时远高于 STAMP dump 后处理，建议一次只运行一个 split。

## 7. 暂不混入同一脚本的数据集

- ReasonSeg：需要 reasoning segmentation 专用 JSON、prompt 和指标协议。
- gRefCOCO：包含 no-target 与多目标语义，需要显式报告 generalized referring 指标。
- RefCLEF/ReferIt：图像与标注布局不同，需要单独的数据转换检查。

这些数据集可以继续接入同一个 Training-Free 核心，但必须先实现各自公平的基础模型
评测协议，不能仅替换 JSON 文件后与 RefCOCO 系列混表。
