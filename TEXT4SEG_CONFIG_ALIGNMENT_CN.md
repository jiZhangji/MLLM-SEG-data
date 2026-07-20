# Text4Seg 配置对齐说明

## 结论

当前已经完成的 Text4Seg RefCOCOg val/test 实验使用
`lmc22/text4seg-llava-7b-p24`。这是作者用于快速演示的
LLaVA-1.5/Vicuna-7B checkpoint；`p24` 表示输出为 `24 x 24 = 576` 个语义描述符，
不是 24 个视觉 token。它不能作为论文表格中任意一行的严格复现。

Text4Seg 论文默认使用 `16 x 16 = 256` 个语义描述符（`p16`），并在消融中测试
`p24` 和 `p32`。论文表格中的 Text4Seg 行并非同一设置：

| 截图中的行 | 论文中的实际设置 |
|---|---|
| Text4Seg, Vicuna-13B, 74.1/.../70.1 | LLaVA-1.5-13B, p16 + CRF |
| Text4Seg, InternLM2.5-7B, 74.7/.../71.6 | InternVL2-8B, p32, 无 mask refiner |
| Text4Seg, Vicuna-7B, 79.3/.../73.9 | LLaVA-1.5-7B, p16 + SAM-H |
| Text4Seg, InternLM2.5-7B, 79.2/.../75.3 | InternVL2-8B, p16 + SAM-H |
| Text4Seg, Vicuna-13B, 80.2/.../75.1 | LLaVA-1.5-13B, p16 + SAM-H |

因此，与当前 Vicuna-7B 代码路径最容易统一的论文配置是
**Text4Seg LLaVA-1.5-7B-p16**。论文附录给出的同一模型基准为：

| Refiner | RefCOCO val/testA/testB | RefCOCO+ val/testA/testB | RefCOCOg val/test | Avg. |
|---|---|---|---|---:|
| None | 70.5 / 72.3 / 69.3 | 64.4 / 68.7 / 60.6 | 65.1 / 66.5 | 67.2 |
| CRF | 73.2 / 75.7 / 71.4 | 67.0 / 71.9 / 62.4 | 67.3 / 68.9 | 69.7 |
| SAM-H | 79.3 / 81.9 / 76.2 | 72.1 / 77.6 / 66.1 | 72.1 / 73.9 | 74.9 |

## 本项目如何避免再混淆

`training_free_refine.export_text4seg_masks` 现在使用
`--descriptor-grid-size`，并根据 checkpoint 路径中的 `p16/p24/p32` 强制校验。
例如，p24 checkpoint 配置成 p16 会直接退出，不再静默产生不可解释的结果。
旧参数 `--visual-tokens` 仅为兼容保留。

完整 p16 配对评测入口为：

```bash
bash run_text4seg_llava7b_p16_full_eval.sh
```

它只接受本地 `llava-v1.5-7b-p16` checkpoint，并对 RefCOCO、RefCOCO+、
RefCOCOg 的八个 split 串行计算 Text4Seg、Text4Seg + FreeRef 及独立 SAM 对照。
该入口统一了模型族和描述符分辨率，但仍使用本项目的配对 flat JSON；只有先通过
作者官方 REFER loader 的基线复现门槛，结果才可写成“复现论文数值”。

## 权重来源边界

作者官方仓库把 p16 checkpoints 放在 OneDrive，而 Hugging Face 公开的
`lmc22/text4seg-llava-7b-p24` 是混合数据训练的演示 checkpoint。二者不可互换。
下载应与评测命令分开完成，并将官方 p16 权重放到：

```text
models/Text4Seg/llava-v1.5-7b-p16/
```

然后用以下命令检查八个 split 的状态：

```bash
bash check_text4seg_llava7b_p16_status.sh
```
