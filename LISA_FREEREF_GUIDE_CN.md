# LISA + FreeRef 配对评测

## 评测路径

```text
图像 + 指代表达式
  -> LISA-7B-v1
  -> LISA 内置且已训练的 SAM mask decoder
  -> 原始 mask logits / 原始二值 mask
  -> 固定参数 FreeRef
  -> refined mask
  -> 同一样本上的前后 mIoU、cIoU 和 Boundary IoU
```

这项实验验证 FreeRef 能否接在已有 mask decoder 的模型之后。LISA 自身包含 SAM 解码器，因此该实验应表述为 `LISA + FreeRef` 的通用性实验，不能写成 `No SAM` 主结果。

## 实现原则

- 使用项目现有的 STAMP flat evaluation JSON，确保 LISA、STAMP 和 Text4Seg 使用相同图像、表达式与 GT。
- 使用 LISA 官方验证协议：把 `[SEG].` 作为 assistant response 输入，通过对应隐藏状态驱动 LISA 的 SAM decoder；不使用自由生成文本替代官方验证协议。
- 保存阈值化前的 SAM mask logits 为压缩 NPZ。FreeRef 直接读取 logits，因此能够利用原始置信度，而不只依赖硬边界。
- 每个样本同时保存原始二值 PNG、GT PNG 和 logits；重复运行会跳过完整样本并继续未完成部分。
- FreeRef 参数与 STAMP/Text4Seg 实验保持一致，不针对 LISA 调参。

## 默认运行

默认在 `RefCOCO testA` 完整测试集上运行。该划分适合作为当前 `LISA-7B-v1` 的第一项实验，因为官方说明 v1 权重使用了 train+val 数据，不能把其 validation 结果当作严格未见验证集结果。

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
CUDA_DEVICE=0 nohup bash run_lisa_freeref_eval.sh \
  > ../outputs/lisa_freeref_refcoco_testA.log 2>&1 < /dev/null &
```

完整 RefCOCO 系列串行评测：

```bash
LISA_SPLITS="refcoco_testA refcoco_testB refcoco+_testA refcoco+_testB refcocog_test" \
CUDA_DEVICE=0 nohup bash run_lisa_freeref_eval.sh \
  > ../outputs/lisa_freeref_test_splits.log 2>&1 < /dev/null &
```

先用 16 个样本做集成检查：

```bash
LISA_EVAL_LIMIT=16 CUDA_DEVICE=0 \
LISA_RESULTS_ROOT=../outputs/lisa_official_smoke \
LISA_FREEREF_ROOT=../outputs/universal_freeref_lisa_smoke \
bash run_lisa_freeref_eval.sh
```

已有环境且服务器不能联网时，可设置 `LISA_SETUP_ENV=0` 跳过安装。推理本身强制读取本地 LISA 和 CLIP 权重，不访问 Hugging Face。

## 状态与日志

```bash
bash check_lisa_freeref_status.sh
```

```bash
tail -n 100 -F ../outputs/lisa_freeref_refcoco_testA.log
```

最终结果：

```text
../outputs/universal_freeref_lisa/combined/comparison.md
```

每个 split 的详细指标与逐样本结果分别位于：

```text
../outputs/universal_freeref_lisa/<split>/eval_summary.json
../outputs/universal_freeref_lisa/<split>/eval_rows.csv
```
