# gRefCOCO、ReasonSeg 与 RefCLEF 串行评测

## 1. 串行任务内容

`run_training_free_special_datasets_serial.sh` 按以下顺序运行：

```text
1. gRefCOCO: val, testA, testB       STAMP-7B
2. ReasonSeg: val, test              ReasonSeg checkpoint 或 STAMP-2B zero-shot
3. RefCLEF/ReferIt: val, test        STAMP-7B；仅在官方数据完整时运行
```

每个 split 启动模型前，脚本每 10 秒通过 `nvidia-smi` 检查一次指定 GPU 的空闲显存。
默认空闲显存达到 `24000 MiB` 才启动；前一个 split 完成并释放模型后才会检查并运行
下一个 split，因此不会并行占用 GPU。

## 2. 数据协议

### gRefCOCO

- 从 `grefs(unc).json` 和 `instances.json` 生成 val/testA/testB JSON。
- 同一表达对应多个实例时合并为一个二值 GT mask。
- no-target 样本保留空 GT，而不是丢弃。
- STAMP 没有生成 `<SEG>` 时保存为空预测并计入 gIoU/cIoU。
- `mean_iou` 在该数据集上应解释为 per-sample gIoU。

### ReasonSeg

- 从每张图对应的 polygon JSON 生成 target mask 和 ignore mask。
- ignore 区域从 IoU 计算中排除。
- 论文使用专门进行 reasoning segmentation 微调的 STAMP-2B。
- 若 `${ROOT}/models/STAMP-2B-reasonseg` 不存在，脚本使用
  `${ROOT}/models/STAMP-2B-uni`，并将结果明确标记为 `base_zero_shot`；该结果不能
  与论文 Table 4 的微调模型作严格复现比较。
- 可通过 `REASONSEG_MODEL_NAME=/path/to/checkpoint` 指定正确权重。

### RefCLEF/ReferIt

- 需要官方 `refs(unc).p`、`instances.json` 和 `saiapr_tc-12` 图像。
- 当前 `yiqun/referit` Hugging Face 仓库主要包含加载代码，并不保证包含授权图像。
- 如果服务器数据不完整，脚本会写入 `skipped_missing_authorized_data` 后结束，不会生成
  虚假的 RefCLEF 指标，也不会影响之前完成的 gRefCOCO/ReasonSeg 结果。

## 3. 一条命令全量运行

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && git pull --rebase --autostash https://github.com/jiZhangji/MLLM-SEG-data.git main && mkdir -p ../outputs && (CUDA_DEVICE=0 SPECIAL_MIN_FREE_GPU_MB=24000 SPECIAL_GPU_POLL_SECONDS=10 nohup bash run_training_free_special_datasets_serial.sh > ../outputs/training_free_special_datasets_serial.log 2>&1 < /dev/null & echo "Special datasets PID: $!")
```

这条命令默认运行全量数据，不设置 sample limit。

## 4. 日志和状态

```bash
tail -n 50 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/training_free_special_datasets_serial.log
```

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && CUDA_DEVICE=0 bash check_training_free_special_status.sh
```

等待显存时日志每 10 秒出现：

```text
[date time] grefcoco_val: GPU 0 free N MiB; required 24000 MiB.
```

## 5. 输出

每个 split：

```text
outputs/refine_stamp_dumps/{dataset}_{split}_full/
outputs/training_free_refine_{dataset}_{split}_full/eval_summary.json
outputs/training_free_refine_{dataset}_{split}_full/eval_rows.csv
```

统一汇总：

```text
outputs/training_free_special_datasets_comparison/combined_summary.md
outputs/training_free_special_datasets_comparison/combined_summary.json
outputs/training_free_special_datasets_comparison/serial_status.tsv
```

任务中断后执行同一条命令会复用已经保存的 dumps。

## 6. 可调设置

```text
CUDA_DEVICE=1                       改用 GPU 1
SPECIAL_MIN_FREE_GPU_MB=30000       提高启动显存阈值
SPECIAL_GPU_POLL_SECONDS=10         显存轮询间隔
REASONSEG_MODEL_NAME=/path/model    指定 ReasonSeg 专用 checkpoint
SPECIAL_EVAL_LIMIT=N                仅调试使用；默认 0 为全量
```
