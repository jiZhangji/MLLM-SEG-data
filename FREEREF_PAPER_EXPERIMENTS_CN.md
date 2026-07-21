# FreeRef 论文实验代码与运行说明

## 1. 论文主线

FreeRef 的主张是：对基础模型已经输出的二维概率图或掩码进行免训练细化，而不修改基础模型参数。论文主实验只使用可逐样本配对、可追溯的本地预测，不把论文表中抄录的数字当作实验结果。

| 基础模型 | FreeRef 接入位置 | 输入类型 | 当前用途 |
|---|---|---|---|
| STAMP-2B/7B | patch logits 之后 | 软 logits | 已完成的主结果 |
| Text4Seg public p24 | token mask 之后 | 硬掩码 | 已完成的跨模型主结果 |
| PixelLM-7B public | 轻量 pixel decoder 之后 | 软 logits | 新增跨架构主结果 |
| SegAgent-SimpleClick | 最终 SimpleClick 掩码之后 | 硬掩码 | 补充实验；资源完整时运行 |
| HIPIE/ReLA/PolyFormer/UNINEXT/SegLLM | 官方最终预测之后 | 软图或硬掩码 | 通过通用导入器接入 |

LISA、GSVA、READ 等方法的 SAM 解码器属于训练系统的一部分。FreeRef 只能接在完整模型最终掩码之后；LISA 的“FreeRef 插到原生 SAM 前”小规模实验已经下降，因此不进入论文主表。

## 2. 新增代码

- `universal_freeref/export_pixellm_masks.py`：按 PixelLM 官方公开推理配置导出连续 logits、硬掩码、GT 和 Manifest。
- `universal_freeref/import_segagent_outputs.py`：把 SegAgent 官方轨迹 JSON 的最终 SimpleClick 掩码转换为 Manifest。
- `universal_freeref/run_segagent_official.py`：复用官方模型与推理类，同时解除官方代码中 `self.coco[:5000]` 的截断。
- `universal_freeref/build_eval_json_manifest.py`：按样本索引把任意外部方法预测和 STAMP 平铺评测 JSON 对齐。
- `universal_freeref/summarize_paper_suite.py`：汇总 STAMP、Text4Seg、PixelLM 和 SegAgent 的 paired 前后结果并核验样本数。
- `run_freeref_paper_suite.sh`：跳过已有 STAMP/Text4Seg，运行剩余模型并生成统一表格。
- `check_freeref_paper_suite_status.sh`：显示进程、GPU、逐 split 进度和当前论文表格。

## 3. 协议说明

PixelLM 使用公开 `PixelLM-7B/hf_model`，参数固定为 3 个 segmentation codebook token、2 个图像特征尺度、448 CLIP 输入和 bf16。多个预测掩码通过概率并集合并。该实验使用项目现有的平铺 JSON 做逐样本 paired 对比，因此会标记为 `stamp_flat_json_not_paper_reproduction`；只有基础 PixelLM 指标复现到合理范围后，结果才进入论文主表。

SegAgent 使用公开 Qwen 模型、7 次交互和 SimpleClick ViT-L。FreeRef 接在最终 SimpleClick 掩码之后，不插入交互循环，也不改变其语言动作策略。

## 4. 下载与推理解耦

下载代码和权重时执行：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
mkdir -p ../outputs
nohup env METHODS="pixellm segagent" MIN_FREE_GB=80 bash prepare_freeref_paper_assets.sh > ../outputs/freeref_paper_assets.log 2>&1 < /dev/null &
```

下载状态：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && bash check_missing_method_weights.sh
```

下载结束后，正式推理不再访问网络。

## 5. 先做门控测试

PixelLM 64 样本：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && mkdir -p ../outputs && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIXELLM_SPLITS="refcoco_testA" PIXELLM_LIMIT=64 PIXELLM_CUDA_DEVICES="0" PIXELLM_OUTPUT_ROOT=../outputs/pixellm_public_freeref_smoke nohup bash run_pixellm_freeref_full_eval.sh > ../outputs/pixellm_public_freeref_smoke.log 2>&1 < /dev/null &
```

SegAgent 64 条轨迹：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && mkdir -p ../outputs && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SEGAGENT_SPLITS="refcoco_testA" SEGAGENT_LIMIT_ITEMS=64 SEGAGENT_CUDA_DEVICES="0" SEGAGENT_OUTPUT_ROOT=../outputs/segagent_freeref_smoke nohup bash run_segagent_freeref_full_eval.sh > ../outputs/segagent_freeref_smoke.log 2>&1 < /dev/null &
```

应先核对基础模型指标、空预测数量、样本数和定性图，再决定是否放大全量。

## 6. 两张 H100/H200 全量运行

一张卡一个模型进程，最稳妥：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && mkdir -p ../outputs && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIXELLM_CUDA_DEVICES="0 1" SEGAGENT_CUDA_DEVICES="0 1" PAPER_RUN_PIXELLM=1 PAPER_RUN_SEGAGENT=auto nohup bash run_freeref_paper_suite.sh > ../outputs/freeref_paper_suite.log 2>&1 < /dev/null & echo "Paper suite PID: $!"
```

若两张 80GB GPU 确认均为空闲，可在每张卡放两个进程：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && mkdir -p ../outputs && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIXELLM_CUDA_DEVICES="0 0 1 1" SEGAGENT_CUDA_DEVICES="0 0 1 1" PIXELLM_MIN_FREE_MB_PER_JOB=22000 SEGAGENT_MIN_FREE_MB_PER_JOB=22000 PAPER_RUN_PIXELLM=1 PAPER_RUN_SEGAGENT=auto nohup bash run_freeref_paper_suite.sh > ../outputs/freeref_paper_suite.log 2>&1 < /dev/null & echo "Paper suite PID: $!"
```

这里的并行单位是独立 split 和独立模型进程，不改变单样本推理结果。若显存不足，worker 会每 10 秒等待；已有完整 summary 会自动跳过。

## 7. 持续监控

```bash
watch -n 10 'cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && bash check_freeref_paper_suite_status.sh'
```

主日志：

```bash
tail -n 100 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/freeref_paper_suite.log
```

最终统一表格：

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/freeref_paper_suite/paper_results.md
```

## 8. 接入其他模型的已保存预测

预测文件按评测 JSON 的全局索引命名为 `00000000.png`、`00000001.png` 等时：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && METHOD=HIPIE SPLIT=refcoco_val PREDICTION_ROOT=/absolute/path/to/hipie_predictions PREDICTION_TEMPLATE='{index:08d}.png' PREDICTION_KIND=mask bash run_external_freeref_eval.sh
```

对于 `.npy/.npz/.pt` 软概率或 logits，将 `PREDICTION_KIND` 设为 `probability` 或 `logits`，必要时通过 `ARRAY_KEY` 指定数组键。该接口适用于 HIPIE、ReLA、PolyFormer、UNINEXT、SegLLM 以及作者提供的其他最终二维预测。
