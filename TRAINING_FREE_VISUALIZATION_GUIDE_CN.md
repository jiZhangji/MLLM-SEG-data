# Training-Free 与原始 STAMP/Text4Seg 可视化说明

## 1. 目标

可视化系统用于比较：

```text
原始 STAMP      vs STAMP + Training-Free
原始 Text4Seg   vs Text4Seg + Training-Free
```

它直接读取已经完成的 `eval_rows.csv`、STAMP dumps 和 Text4Seg masks，重新执行确定性
图细化并提取不确定性诊断，不需要重新运行 STAMP 或 Text4Seg 大模型，也不需要 GPU。

## 2. 单样本对比图

每张定性图为 `2 x 4` 布局：

| 位置 | 内容 | 作用 |
|---|---|---|
| 1 | Input image | 原始视觉信息 |
| 2 | Ground truth | 仅供事后分析的真实区域 |
| 3 | Original STAMP/Text4Seg | 原方法粗 mask 与 IoU |
| 4 | + Training-Free | 细化 mask 与 IoU |
| 5 | Original foreground probability | 原始语义置信度；Text4Seg 为二值概率 |
| 6 | Uncertainty | 方法允许图传播介入的位置 |
| 7 | Image-aware superpixel graph | SLIC 图像区域及边界 |
| 8 | Pixel outcome | 绿色为修正像素，红色为新增错误，黄色为持续错误 |

图标题同时给出 IoU delta、Boundary IoU delta、不确定区域比例和实际修改区域比例。

样本被自动分为四类，每类默认输出 3 张：

```text
strongest_improvements   提升最大的样本
strongest_degradations   退化最大的失败案例
highest_uncertainty      不确定区域最大的样本
representative           接近中位数增益的代表样本
```

这些类别使用 GT 指标进行事后挑选，只用于分析和论文作图；GT 从不进入细化算法。

## 3. 单模型总体分析

每个模型输出 `run_overview.png` 和矢量 `run_overview.pdf`，包含：

1. 样本级 IoU 增益分布。
2. 原始 IoU 与增益的关系。
3. 不确定区域比例与增益的关系。
4. 目标面积与增益的关系。
5. Boundary IoU 增益分布。
6. improved/degraded/unchanged 样本比例。

同时生成 `visual_analysis_rows.csv`，记录每个样本的：

```text
coarse/refined IoU 与 Boundary IoU
object_fraction
uncertainty_mean / uncertainty_q90 / uncertain_fraction
changed_fraction
corrected_fraction / regressed_fraction / persistent_error_fraction
uncertainty_on_changed / uncertainty_on_corrected
```

`visual_analysis_summary.json` 和 `.md` 会汇总 mIoU、cIoU、边界增益、提升比例以及
原始质量、不确定性、目标尺寸与最终增益之间的相关性。

## 4. 跨模型对比

`combined/cross_model_comparison.png` 和 PDF 同时展示：

- STAMP 与 Text4Seg 的原始/细化 mIoU。
- STAMP 与 Text4Seg 的原始/细化 cIoU。
- 两个模型的样本级 IoU 增益分布。
- 平均修正像素比例与新增错误比例。

这组图用于回答两个问题：方法是否在不同基础模型上都有效，以及增益究竟来自更多
正确修正还是更激进的全图改写。

## 5. 本地代码测试

```bash
cd /path/to/MLLM-SEG-data
bash run_training_free_visualization_tests.sh
```

该脚本执行 Python 编译、核心测试、可视化生成测试和 Bash 语法检查。需要用服务器
真实结果做 2 样本 smoke test 时：

```bash
VISUALIZATION_DATA_SMOKE=1 bash run_training_free_visualization_tests.sh
```

## 6. 服务器快速预览

先用每个模型 8 个样本验证路径和布局：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && TRAINING_FREE_VIS_LIMIT=8 TRAINING_FREE_VIS_PANELS_PER_GROUP=1 TRAINING_FREE_VIS_OUTPUT=../outputs/training_free_visualizations_smoke bash run_training_free_visualizations.sh
```

输出：

```text
../outputs/training_free_visualizations_smoke/stamp7b_val
../outputs/training_free_visualizations_smoke/text4seg_val
../outputs/training_free_visualizations_smoke/combined
```

## 7. 全量可视化

全量命令：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && mkdir -p ../outputs && nohup bash run_training_free_visualizations.sh > ../outputs/training_free_visualizations.log 2>&1 < /dev/null & echo "Visualization PID: $!"
```

查看日志：

```bash
tail -n 50 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/training_free_visualizations.log
```

默认读取：

```text
STAMP-7B:  outputs/training_free_refine_stamp7b_refcocog_val_full/eval_rows.csv
Text4Seg: outputs/text4seg_training_free_refcocog_val/eval_rows.csv
```

默认写入：

```text
outputs/training_free_visualizations/stamp7b_val
outputs/training_free_visualizations/text4seg_val
outputs/training_free_visualizations/combined
```

全量过程会重新执行 CPU 图细化以计算每个样本的不确定性和像素变化，但不会重新运行
7B 模型。可用 `TRAINING_FREE_VIS_PANELS_PER_GROUP` 调整每类定性图数量。

## 8. 自定义结果路径

例如比较 STAMP-2B 与 Text4Seg：

```bash
STAMP_VIS_ROWS=/path/to/stamp2b/eval_rows.csv \
TEXT4SEG_VIS_ROWS=/path/to/text4seg/eval_rows.csv \
TRAINING_FREE_VIS_OUTPUT=/path/to/output \
bash run_training_free_visualizations.sh
```

也可以单独运行一个模型：

```bash
python -m training_free_refine.visualize_comparison run \
  --kind stamp \
  --rows /path/to/eval_rows.csv \
  --output-dir /path/to/visuals \
  --label STAMP-7B \
  --panels-per-group 3
```

## 9. 论文使用建议

- 正文放跨模型总体图及每个模型 1 个代表提升案例。
- 附录放 strongest improvement、degradation 和 high uncertainty 三类案例。
- 正文明确说明不确定性图来自基础输出或二值边界带，不是训练出的 uncertainty head。
- 将红色新增错误案例用于解释方法限制：颜色相近背景、弱边界、目标极小或语义目标错误。
- PDF 为矢量统计图；单样本图包含原始图像，因此使用高分辨率 PNG。
