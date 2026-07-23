# FreeRef 免训练后处理基线：500 样本双 H100 流程

该流程直接读取保存的 STAMP-7B soft-logit dumps 和 Text4Seg-p24 hard-mask manifest，
不会重新运行基础 MLLM。比较方法包括：

- None / Base
- DenseCRF
- Guided Filter
- SLIC Region Averaging
- Frozen SAM-H
- FreeRef

固定数据为 `RefCOCO | unc | testA` 的前 500 个配对样本。精度阶段在两张 H100
上按模型并行；测速阶段只使用第一张 H100，并严格串行执行所有任务。

## 1. 准备环境

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data"
bash prepare_freeref_baseline_env.sh
```

准备脚本只补充 `pydensecrf` 并检查现有 CuPy/cuCIM 环境。运行脚本本身不会自动安装依赖。

## 2. 后台运行

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data"

nohup env \
FREEREF_BASELINE_GPUS="0 1" \
FREEREF_BASELINE_SAMPLES=500 \
FREEREF_BASELINE_MIN_FREE_MB=60000 \
bash run_freeref_postprocess_baselines_n500.sh \
> "$ROOT/outputs/freeref_postprocess_baselines_n500.log" 2>&1 < /dev/null &

echo "Post-process baseline PID: $!"
```

## 3. 查看状态和实时进度

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data"
bash check_freeref_postprocess_baselines.sh
```

```bash
tail -n 30 -F "$ROOT"/outputs/freeref_postprocess_baselines_n500/logs/*.log | \
stdbuf -oL tr '\r' '\n' | \
grep --line-buffered -E '[0-9]+%|DONE|ERROR|Traceback'
```

最终表格：

```text
outputs/freeref_postprocess_baselines_n500/combined/postprocess_comparison.md
```

## 4. 生成论文图

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data"

conda run --no-capture-output -n text4seg-tf \
python -m training_free_refine.plot_postprocess_baselines \
  --input "$ROOT/outputs/freeref_postprocess_baselines_n500/combined/postprocess_comparison.csv" \
  --output-dir "$ROOT/outputs/freeref_postprocess_baselines_n500/plots" \
  --dpi 300
```

输出包括：

```text
postprocess_accuracy_gains.png
postprocess_accuracy_gains.pdf
postprocess_accuracy_latency.png
postprocess_accuracy_latency.pdf
```

## 协议说明

- Text4Seg 硬掩码仅在 DenseCRF、Guided Filter 和 SLIC Averaging 中固定映射为
  前景/背景概率 `0.95/0.05`；Base 和 FreeRef 保留硬掩码原始接口。
- 精度阶段的 FreeRef 使用参考 CPU 实现，以匹配论文主结果；测速阶段使用指标等价的
  CuPy/cuCIM GPU 实现。
- SAM-H 使用现有 STAMP/Text4Seg 配对提示协议。测速值为一次图像编码加一个提示分支，
  不包含基础 MLLM 推理。
- 当前 500 样本实验用于诊断和确定代码路径。正式论文参数应在 `val` 上冻结后再运行
  完整 `testA`，不能使用 testA 调参。
