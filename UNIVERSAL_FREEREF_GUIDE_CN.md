# Universal FreeRef：跨模型无训练细化与配对评测

## 1. 为什么采用统一框架

HIPIE、ReLA、PolyFormer、UNINEXT、PixelLM、LISA、GSVA、READ、Text4Seg、STAMP、Seg-Zero、SegLLM 和 SegAgent 来自不同代码库，依赖、数据接口和解码方式均不相同。直接在每个仓库内部复制一份 FreeRef 会造成实现漂移，也很难保证比较公平。

因此，本实现把 FreeRef 定义为**输出级黑盒插件**：上游模型保持官方推理流程，只导出最终预测；同一个 FreeRef 实现读取预测并完成细化。这样真正检验的是“一个固定、无训练的数学模块能否迁移到不同指代分割模型”，而不是为每个模型重新调网络。

主路径如下：

```text
无 Mask Decoder：模型原生 mask / probability -> FreeRef -> refined mask
已有 SAM/Decoder：模型 -> 官方 Decoder/SAM -> final mask -> FreeRef -> refined mask
```

论文的核心无 SAM 结论应使用第一条路径。第二条路径用于证明通用性，不应与“No SAM / No Training”的主表行混淆。

## 2. 已覆盖的方法

方法注册表位于 `universal_freeref/methods.json`，包含 13 个代码基座及其官方仓库、输出类型和推荐插入位置。除 SegLLM 尚未确认可执行的官方仓库外，其余方法均登记了官方代码地址。SegLLM 仍可导入作者发布或自行复现得到的预测 mask 进行同协议评测。

框架不自动下载权重，也不把不同上游仓库安装到同一个 Python 环境。可选的源码下载命令为：

```bash
cd MLLM-SEG-data
bash prepare_universal_freeref_repos.sh
```

只下载部分仓库：

```bash
METHODS="hipie rela polyformer" bash prepare_universal_freeref_repos.sh
```

只检查将要下载的仓库而不实际克隆：

```bash
DRY_RUN=1 bash prepare_universal_freeref_repos.sh
```

## 3. 标准清单格式

每个样本占 JSONL 一行：

```json
{"name":"000001","method":"HIPIE","split":"refcoco_val","image":"images/000001.jpg","gt_mask":"gt/000001.png","prediction":"pred/000001.png","prediction_kind":"mask"}
```

必需字段：

- `name`：样本唯一名称。
- `method`：上游方法名。
- `split`：数据划分。
- `image`：原图路径。
- `gt_mask`：GT 二值 mask。
- `prediction`：原始预测路径。
- `prediction_kind`：`mask`、`probability` 或 `logits`。

可选字段：`ignore_mask`、`uncertainty`、`array_key`、`uncertainty_key`、`foreground_channel`、`threshold`、`no_target`、`instance_id`。

支持 PNG/JPEG/TIFF、NumPy `npy/npz` 和 PyTorch `pt/pth`。对于多通道 logits，可用 `foreground_channel` 指定前景通道；对于字典或 NPZ，可用 `array_key` 指定数组键。

已有 `pred/image/gt` 同名目录时，可自动构建清单：

```bash
python -m universal_freeref.build_manifest \
  --method HIPIE \
  --split refcoco_val \
  --prediction-root /path/to/hipie_predictions \
  --image-root /path/to/images \
  --gt-root /path/to/gt_masks \
  --prediction-glob '**/*_pred.png' \
  --strip-suffix _pred \
  --image-template '{relative_stem}.jpg' \
  --gt-template '{relative_stem}.png' \
  --output manifests/hipie_refcoco_val.jsonl
```

## 4. 统一评测

单个方法：

```bash
python -m universal_freeref.evaluate \
  --manifest manifests/hipie_refcoco_val.jsonl \
  --output-dir ../outputs/universal_freeref/hipie_refcoco_val
```

多个方法或划分串行评测并自动生成总表：

```bash
bash run_universal_freeref_eval.sh \
  manifests/hipie_refcoco_val.jsonl \
  manifests/rela_refcoco_val.jsonl \
  manifests/polyformer_refcoco_val.jsonl
```

服务器使用既有 STAMP 环境时，可显式指定 Python：

```bash
PYTHON_BIN=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP/bin/python \
bash run_universal_freeref_eval.sh manifests/*.jsonl
```

FreeRef 本身是 CPU 数学后处理，不需要占用 GPU；GPU 只用于各上游模型生成原始预测。

状态检查：

```bash
bash check_universal_freeref_status.sh
```

## 5. 公平比较协议

1. 前后结果必须来自完全相同的样本集合和同一批上游预测。
2. 原始指标由清单中的 `prediction` 重新计算，FreeRef 指标由同一样本细化后计算。
3. 所有方法首先使用固定的默认 FreeRef 参数，不针对测试集调参。
4. 优先导出概率或 logits；只有硬 mask 时使用边界距离构造不确定性。
5. 多目标任务每个实例写一行，并通过 `instance_id` 标识；空目标使用 `no_target=true`，空预测会被保留。
6. 有 SAM 的方法应报告 `Base+SAM` 与 `Base+SAM+FreeRef`；无 SAM 主路径应报告 `Base` 与 `Base+FreeRef`。
7. 论文表格中的作者报告值只能作背景参考。真正的 FreeRef 增益必须使用同一清单得到的 paired before/after 结果。

## 6. 输出内容

每个评测目录包含：

- `eval_summary.json`：mIoU、cIoU、Boundary IoU、增益、提升/退化样本数、95% bootstrap 区间和 Wilcoxon 配对检验。
- `eval_rows.csv`：逐样本前后指标。
- `coarse_masks/` 与 `refined_masks/`：前后 mask。
- `visualizations/`：Image、GT、Original、Uncertainty、FreeRef、修正/退化图。
- `row_cache/`：逐样本断点续跑缓存。
- `run_config.json`：清单哈希与固定参数，防止误把不同配置写入同一目录。

绿色变化像素表示 FreeRef 修正的错误，红色表示新引入的错误，黄色表示仍未修正的错误。批量比较时，除了平均增益，还应检查退化样本、目标面积、不确定区域比例与边界增益。

## 7. 上游仓库需要改什么

原则上只需要在官方评测脚本保存最终输出，不需要修改网络：

```python
save_mask(prediction, output_path)       # 至少保存硬 mask
np.save(probability_path, probability)   # 能取得概率时优先保存
```

然后把保存路径与原图、GT 写入标准 JSONL。该接口使 13 个方法共享完全相同的 FreeRef 代码；上游仓库更新时也不需要重新合并 FreeRef 分支。
