# R-STAMP Scaffold

R-STAMP 是基于 STAMP 的资源友好型改进方向：

> 用短结构化 reasoning prior 指导 STAMP 的并行 all-mask prediction，并在后续加入 uncertainty refinement 和轻量 mask-aware RL。

这个目录不是完整 fork 版 STAMP，而是一个可复制进 STAMP 工作区的轻量 scaffold。原因是 STAMP 官方代码版本可能变化，最稳妥的做法是：

1. 先把官方 STAMP 跑通；
2. 再把这里的 `rstamp/` 模块接入 STAMP 的 data/model/trainer；
3. 每一步都保留 STAMP baseline，方便做消融。

## 推荐接入点

### Data pipeline

在 STAMP 训练样本中增加：

```json
{
  "structured_prior": {
    "target": "black dog",
    "attributes": ["black", "dog"],
    "relation": "left of the woman",
    "bbox": [x1, y1, x2, y2],
    "positive_points": [[x, y]],
    "negative_points": [[x, y]]
  }
}
```

### Model

在 all-mask prediction 前，将 structured prior 编码成 prior tokens，然后让 mask tokens cross-attend / fuse 这些 prior tokens。

### Loss

第一阶段 SFT：

- mask loss；
- bbox/point auxiliary loss；
- prior format loss；
- optional boundary loss。

第二阶段 refinement：

- coarse mask loss；
- uncertainty calibration loss；
- local refinement loss。

第三阶段轻量 RL：

- format reward；
- IoU/cIoU reward；
- boundary reward；
- conciseness penalty。

## 2×80GB 建议

- bf16；
- LoRA rank 16/32 起步；
- 每卡 micro batch 1-4；
- gradient accumulation 8-32；
- ZeRO-2/ZeRO-3；
- 先 SFT，后 RL；
- RL sampling number 2 或 4，不要一开始设 8。

