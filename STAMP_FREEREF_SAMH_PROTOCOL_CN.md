# STAMP + FreeRef + Official SAM-H

## 修正目标

本评测比较同一批 STAMP dump 上的四条配对路径：

1. `STAMP`
2. `STAMP + FreeRef`
3. `STAMP + SAM-H`
4. `STAMP + FreeRef + SAM-H`

目标顺序是 `STAMP < STAMP + FreeRef < STAMP + FreeRef + SAM-H`。代码只报告真实结果，
不会根据 GT 选择输出或强行制造该顺序。

## 本次修正

- 直接导入指定 STAMP 仓库中的 `eval.utils.compute_logits_from_mask`。
- 直接导入 `eval.utils.masks_sample_points`，保持官方每类 10 个点的采样算法。
- 保持官方的一次初始预测和两次低分辨率 logit 级联。
- STAMP 与 FreeRef 分支对同一样本使用相同随机种子，减少随机点差异造成的比较噪声。
- 运行时验证 `eval.utils` 的真实文件路径，避免导入同名第三方模块。
- 新结果写入 `stamp_official_samh_*`，不会复用旧版错误协议的 `frozen_samh_*` 文件。

## 64 样本快速验证

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data

SAMH_OUTPUT_ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/stamp_official_samh_smoke \
SAMH_COMBINED_DIR=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/stamp_official_samh_smoke/combined \
SAMH_ONLY_JOBS="STAMP-2B:refcoco_testA" \
SAMH_LIMIT=64 \
CUDA_DEVICE=0 \
SAM_PATH=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/models/SAM/sam_vit_h_4b8939.pth \
bash run_frozen_samh_full_eval.sh
```

结果文件：

```text
../outputs/stamp_official_samh_smoke/stamp_official_samh_stamp-2b_refcoco_testA/eval_summary.json
```

重点检查：

```text
sam_failures_with_input_fallback == 0
target_ordering_mean_iou == true
target_ordering_cIoU == true
```

## 全量评测

快速验证通过后再运行。单张 H100 80GB 推荐两个 split 并行；每个进程各加载一份
SAM-H，同一 split 内部仍保持官方逐样本评测逻辑：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE=0 \
SAMH_PARALLEL_JOBS=2 SAMH_MIN_FREE_MB=24000 \
SAM_PATH=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/models/SAM/sam_vit_h_4b8939.pth \
nohup bash run_frozen_samh_full_eval.sh \
  > ../outputs/stamp_official_samh_full_eval.log 2>&1 < /dev/null &
```

如果同时运行其他显存任务，去掉 `SAMH_PARALLEL_JOBS=2` 即恢复默认串行。

这里对 SAM-H 后处理采用官方协议，但前三条 STAMP 路径仍来自同一批已保存 dump。
因此它是严格配对的后处理实验，不等同于从模型生成阶段开始重跑 STAMP 论文评测。
