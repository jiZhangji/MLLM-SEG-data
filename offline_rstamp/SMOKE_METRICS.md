# Smoke Metrics

当前已有两个 1×48GB smoke training：

- `outputs/smoke_baseline_1x48g`
- `outputs/smoke_rstamp_1x48g`

运行以下脚本生成训练指标对比：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
bash offline_rstamp/run/46_compare_smoke_training_metrics.sh
```

输出：

```text
outputs/smoke_training_metrics_comparison.json
outputs/smoke_training_metrics_comparison.csv
outputs/smoke_training_metrics_comparison.md
```

注意：

这些是训练过程指标，包括：

- `loss`
- `loss_lm`
- `loss_seg`
- `mean_token_accuracy`

它们只能证明 smoke training 是否正常，以及 early optimization signal。它们不能替代最终分割评估。

真正证明方法有效还需要下一步做验证集推理并计算：

- IoU；
- cIoU；
- gIoU；
- hard subset 上的 improvement。

