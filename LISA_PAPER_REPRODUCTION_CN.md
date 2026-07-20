# LISA 论文结果严格复现

## 实验原则

所有外部方法都按以下顺序评测：

1. 使用作者代码、作者权重、作者原始数据划分和作者指标复现论文基线。
2. 基线样本数与论文协议不一致时停止。
3. 基线 cIoU 未达到论文复现容差时停止，不报告 FreeRef 对比。
4. 通过复现门槛后，读取同一次运行导出的原始 logits，再加入 FreeRef。
5. 前后结果使用完全相同的样本、GT、阈值和指标。

`universal_freeref/export_lisa_masks.py` 使用 STAMP flat JSON，适合跨方法配对分析，但不属于 LISA 论文复现协议。论文复现必须使用本目录新增的 `eval_lisa_paper_protocol.py`。

## 所需文件

LISA 官方数据根目录必须具有以下布局：

```text
<LISA_PAPER_DATA_ROOT>/
├── refcoco/refs(unc).p
├── refcoco/instances.json
├── refcoco+/refs(unc).p
├── refcoco+/instances.json
├── refcocog/refs(umd).p
├── refcocog/instances.json
└── images/mscoco/images/train2014/*.jpg
```

服务器还需要：

```text
models/freeref_missing_methods/lisa/LISA-7B-v1
models/freeref_missing_methods/shared/clip-vit-large-patch14
models/SAM/sam_vit_h_4b8939.pth
code/third_party/lisa
```

这里使用的是 LISA 已训练好的权重，不需要重新微调。

## 第一项全量复现

先复现论文截图中 `LISA-7B (fine-tuned on ReferSeg)` 的 RefCOCO testA `cIoU=79.1`：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
LISA_PAPER_DATA_ROOT=/path/to/official/lisa/dataset \
CUDA_DEVICE=0 \
nohup bash run_lisa_paper_reproduction.sh \
  > ../outputs/lisa_paper_refcoco_testA.log 2>&1 < /dev/null &
```

脚本默认全量运行 `refcoco|unc|testA`，要求 5657 个表达式，并要求最终 cIoU 与 79.1 的差距不超过 0.5 个百分点。失败时仍保存结果，但返回非零状态，防止误把未复现的输出接入 FreeRef。

查看日志和状态：

```bash
tail -n 100 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/lisa_paper_refcoco_testA.log
```

```bash
bash check_lisa_paper_reproduction_status.sh
```

## 完整论文表格

RefCOCO testA 通过后，再串行运行八个 split：

```bash
LISA_PAPER_SPLITS='refcoco|unc|val refcoco|unc|testA refcoco|unc|testB refcoco+|unc|val refcoco+|unc|testA refcoco+|unc|testB refcocog|umd|val refcocog|umd|test' \
CUDA_DEVICE=0 bash run_lisa_paper_reproduction.sh
```

注意：LISA 官方 README 说明 v1 使用了 train+val；严格报告 validation 结果时需要核对并使用作者对应的 v0 权重/旧版代码。testA、testB 和 RefCOCOg test 应优先作为当前 v1 权重的复现检查。

## 输出

每个 split 会生成：

```text
outputs/lisa_paper_reproduction/<split>/paper_reproduction_summary.json
outputs/lisa_paper_reproduction/<split>/manifest.jsonl
outputs/lisa_paper_reproduction/<split>/pred_logits/*.npz
outputs/lisa_paper_reproduction/<split>/pred_masks/*.png
outputs/lisa_paper_reproduction/<split>/gt_masks/*.png
```

`manifest.jsonl` 是第二阶段 FreeRef 的唯一输入。只有 `paper_match=true` 后才继续第二阶段。
