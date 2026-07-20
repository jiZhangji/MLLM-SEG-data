# Text4Seg 配置对齐说明

## 核心结论

当前已经完成的 Text4Seg RefCOCOg val/test 实验使用公开 checkpoint
`lmc22/text4seg-llava-7b-p24`。`p24` 表示输出为 `24 x 24 = 576` 个语义描述符，
不是 24 个视觉输入 token。该权重是作者用于演示的混合数据 checkpoint，不能作为
STAMP 表格中任一 Text4Seg 行的严格复现。

作者 OneDrive 中公开的 `llava-v1.5-7b-p16.zip` 对应 Text4Seg 论文 Table 4
的描述符分辨率/后处理实验。它可用于同一公开权重上的配对研究，但也不是 STAMP
截图 Table 1 所使用的训练权重。

STAMP 截图中的 Text4Seg 行来自 Text4Seg Table 1，实际混合了不同模型和后处理：

| STAMP 表格中的行 | Text4Seg 实际配置 |
|---|---|
| Vicuna-13B，decoder-free，Avg. 70.9 | LLaVA-1.5-13B，p16 + CRF |
| InternLM2.5-7B，decoder-free，Avg. 71.4 | InternVL2-8B，p32，无 refiner |
| Vicuna-7B，with decoder，Avg. 74.9 | LLaVA-1.5-7B，p16 + SAM-H |
| InternLM2.5-7B，with decoder，Avg. 75.4 | InternVL2-8B，p16 + SAM-H |
| Vicuna-13B，with decoder，Avg. 76.2 | LLaVA-1.5-13B，p16 + SAM-H |

官方 `ms-swift` 推理脚本引用的是作者机器上的本地训练目录，例如 LLaVA-1.5-7B
的 `checkpoint-33930`。截至 2026-07-20，作者仓库、Hugging Face 与公开 OneDrive
均未提供这些 Table 1 merged checkpoint/LoRA adapter。因此，目前不存在一个可直接
下载并严格复现 STAMP 截图 Text4Seg 数值的公开权重。

## 当前运行器的准确用途

`training_free_refine.export_text4seg_masks` 使用显式参数
`--descriptor-grid-size`，并根据 checkpoint 路径中的 `p16/p24/p32` 校验配置。
例如，p24 checkpoint 配置成 p16 会直接退出。旧参数 `--visual-tokens` 仅为兼容保留。

```bash
bash run_text4seg_llava7b_p16_full_eval.sh
```

该入口只接受本地 LLaVA-1.5-7B-p16 checkpoint，并在 RefCOCO、RefCOCO+、
RefCOCOg 八个 split 上计算 Text4Seg、Text4Seg + FreeRef 和独立 SAM-H 对照。
它使用本项目的配对 flat JSON。即使输入作者公开的 Table 4 p16 权重，结果也必须标为
“released p16 paired evaluation”，不能标为“STAMP Table 1 reproduction”。

状态检查：

```bash
bash check_text4seg_llava7b_p16_status.sh
```

## 如何得到与 STAMP 表格相同的配置

有且只有两条可靠路径：

1. 向 Text4Seg 作者申请 Table 1 的 merged checkpoint 或 LoRA adapter、对应
   `ms-swift` commit 和评测命令。若复现 Vicuna-7B + SAM-H 行，应申请
   LLaVA-1.5-7B p16 的 `checkpoint-33930`。
2. 使用官方训练脚本和数据重新训练。LLaVA-1.5-7B p16 的公开训练设置为 LoRA
   rank 64、global batch 128、5 epochs、学习率 `2e-4`，原论文使用 8 张 A800-40GB。
   单张 H100-80GB 可通过梯度累积保持 global batch，但训练结果不保证逐点等于论文。

在拿到或重训出 Table 1 权重之前，论文表格应将“原论文报告值”与“公开 checkpoint
配对结果”分成两个区块，不能把 p24/p16 Table 4 本地结果与 Table 1 数值写成同协议
的前后比较。
