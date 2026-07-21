# PolyFormer-L + FreeRef 配对验证

## 验证目的

本实验使用 PolyFormer 官方代码、官方 PolyFormer-L 数据集专用权重和官方 REFER 数据协议。PolyFormer 先生成连续多边形顶点，随后严格复用官方 `polygon2mask` 坐标约定栅格化为硬掩码；FreeRef 只处理该最终硬掩码，不进入 PolyFormer 的自回归顶点生成过程。

第一轮固定评测 RefCOCO testA 的前 64 个表达式。该实验是方法适用性门控，不用于替代论文的全量指标。脚本会验证官方导出器与 FreeRef 读取到的基线 mIoU 完全一致，再报告配对增益、95% 置信区间和 Wilcoxon 检验。

## 阶段一：准备代码、权重和独立环境

这一阶段需要网络，但不会启动模型推理：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
nohup bash prepare_polyformer_freeref_assets.sh > ../outputs/polyformer_freeref_assets.log 2>&1 < /dev/null &
```

日志：

```bash
tail -n 100 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/polyformer_freeref_assets.log
```

## 阶段二：离线运行 64 样本验证

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
nohup env CUDA_DEVICE=0 POLYFORMER_LIMIT=64 POLYFORMER_BATCH_SIZE=8 bash run_polyformer_freeref_smoke.sh > ../outputs/polyformer_freeref_smoke_n64.log 2>&1 < /dev/null &
```

该阶段设置 Hugging Face 离线模式，不会下载模型。GPU 显存低于 30GB 时每 10 秒检查一次，满足条件后自动开始。

## 进度与结果

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data
POLYFORMER_LIMIT=64 bash check_polyformer_freeref_status.sh
```

持续日志：

```bash
tail -n 100 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/polyformer_freeref_smoke_n64.log
```

最终配对表位于：

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/polyformer_freeref_smoke_n64_o0/comparison.md
```

只有在小样本的基线导出一致性通过，且增益方向合理时，才扩展至三个数据集的全量实验。
