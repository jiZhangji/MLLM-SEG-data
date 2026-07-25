# FreeRef 剩余六个方法：下载、环境与四卡运行

目标方法：ReLA、PolyFormer-L、UNINEXT-L、LISA、GSVA、READ。脚本只在各方法的最终预测掩码之后执行 FreeRef，不修改第三方仓库。除 ReLA 的经典 RefCOCO 三套权重需要本地复训外，其余可运行方法使用作者发布的权重。

## 当前可执行状态

| 方法 | 代码/权重 | 八划分执行方式 | 论文状态 |
|---|---|---|---|
| READ | 官方完整公开权重 | H200 单卡，八划分续跑 | 基线门控后可作为候选 |
| GSVA | 官方 `ft-res` 权重；另需授权 LLaMA-7B 基座合并 LLaVA delta | H200 单卡、默认 fp32，八划分续跑 | 基线门控后可作为候选 |
| PolyFormer-L | 三套官方数据集专用权重 | H100 单卡，八划分续跑 | 基线门控后可作为候选 |
| ReLA | 作者只发布 gRefCOCO 权重；经典三数据集本地复训 | 三个训练作业占满四卡，然后四卡并行评测 | 基线门控后可作为候选 |
| LISA | 公开 v1 权重 | H100 单卡，平铺 JSON 八划分 | 仅诊断；已知基线门控失败 |
| UNINEXT-L | 官方 Stage-2 ConvNeXt-L 权重链接仍为 403 | 暂不启动 | 阻塞 |

所有 `paper_candidate` 都会与论文中的八个基线 cIoU 逐项比较。任一划分的绝对误差超过 2.0 个百分点时，汇总器自动把结果降为 `diagnostic_only`。

## 1. 下载全部可自动获取的代码和权重

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
REPO="$ROOT/MLLM-SEG-data"
cd "$REPO" || exit 1
mkdir -p "$ROOT/outputs"

nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  MIN_FREE_GB=160 \
  ALLOW_INCOMPLETE=1 \
  bash prepare_remaining_six_assets.sh \
  > "$ROOT/outputs/freeref_remaining_six_assets.log" 2>&1 < /dev/null &
echo "asset PID=$!"
```

查看下载状态：

```bash
cd "$REPO"
bash check_remaining_six_assets.sh
tail -n 100 -F "$ROOT/outputs/freeref_remaining_six_assets.log"
```

`ALLOW_INCOMPLETE=1` 只允许脚本在记录人工阻塞项后退出；不会把 GSVA 授权基座、ReLA 未公开经典权重或 UNINEXT-L 权重标成完成。

## 2. 串行准备隔离环境

下载完成后再执行。环境安装必须串行，避免 Conda/Pip 锁冲突：

```bash
cd "$REPO"
nohup env MLLM_SEG_ROOT="$ROOT" \
  bash prepare_remaining_six_runtimes.sh \
  > "$ROOT/outputs/freeref_remaining_six_runtimes.log" 2>&1 < /dev/null &
echo "runtime PID=$!"
```

```bash
tail -n 100 -F "$ROOT/outputs/freeref_remaining_six_runtimes.log"
```

该步骤准备 READ、PolyFormer-L、LISA、ReLA、GSVA 的独立环境和标准 RefCOCO 数据链接。READ 会把公开 checkpoint 中记录的在线 CLIP ID 强制重定向到本地 CLIP 快照。GSVA 官方 `main.py` 即使评测标准 RefCOCO 也会构造一个辅助 gRefCOCO dataset，因此脚本还会复用服务器已有的 `$ROOT/data/annotations/grefcoco`，但不会对它做额外评测。

## 3. GSVA 授权基座合并

GSVA 的公开 delta 不能替代受许可约束的原始 LLaMA-7B。若服务器已有你合法取得的 Hugging Face 格式基座：

```bash
cd "$REPO"
export GSVA_LLAMA7B_BASE=/absolute/path/to/authorized/llama-7b-hf
MLLM_SEG_ROOT="$ROOT" bash prepare_gsva_llava_legacy.sh
```

如果暂时没有该基座，后续 `ready` 阶段会跳过 GSVA，并释放对应 H200；READ、PolyFormer-L 和 LISA 不受影响。

## 4. 第一阶段：四个可推理队列并行

脚本自动按 GPU 名称识别两张 H100 和两张 H200：

- READ → 第一张 H200
- GSVA → 第二张 H200（合并基座存在时）
- PolyFormer-L → 第一张 H100
- LISA 诊断 → 第二张 H100

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=ready \
  bash run_remaining_six_experiments_4gpu.sh \
  > "$ROOT/outputs/remaining_six_ready.nohup.log" 2>&1 < /dev/null &
echo "ready PID=$!"
```

每个方法内部按八个标准划分续跑；已有 `eval_summary.json` 的划分自动跳过。不要同时启动 ReLA 训练阶段，因为两阶段使用同一组 GPU。

## 5. 第二阶段：ReLA 三模型并行复训

确认 `ready` 阶段结束后运行：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=rela-train \
  bash run_remaining_six_experiments_4gpu.sh \
  > "$ROOT/outputs/remaining_six_rela_train.nohup.log" 2>&1 < /dev/null &
echo "ReLA train PID=$!"
```

GPU 分配为：RefCOCO 使用一张 H200，RefCOCO+ 使用两张 H100，RefCOCOg 使用另一张 H200。默认训练设置为 Swin-B、BERT-base、480×480、全局 batch 24、AdamW、学习率 `1e-5`、150k iterations；存在 `last_checkpoint` 时自动续训。

训练完成后进行八划分评测：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=rela-eval \
  bash run_remaining_six_experiments_4gpu.sh \
  > "$ROOT/outputs/remaining_six_rela_eval.nohup.log" 2>&1 < /dev/null &
echo "ReLA eval PID=$!"
```

## 6. 统一监控

```bash
watch -n 20 "cd '$REPO' && MLLM_SEG_ROOT='$ROOT' bash check_remaining_six_experiments_4gpu.sh"
```

也可单独查看：

```bash
bash check_read_freeref_status.sh
bash check_gsva_freeref_status.sh
bash check_rela_freeref_status.sh
```

## 7. Table 1(b) 输出

每个完整方法都会生成：

```text
$ROOT/outputs/<method>_freeref_full/combined/table1b_row.tsv
$ROOT/outputs/<method>_freeref_full/combined/table1b_row.json
$ROOT/outputs/<method>_freeref_full/combined/table1b_row.md
```

实际目录：

- `read_freeref_full`
- `gsva_freeref_full`
- `polyformer_freeref_full`
- `rela_freeref_full`
- LISA 诊断：`universal_freeref_lisa_all8`

`table1b_row.json` 中必须检查 `baseline_gate_passed` 和 `eligibility`。只有 `baseline_gate_passed: true` 且 `eligibility: paper_candidate` 的 paired 行才可进入论文主表。LISA 当前固定为诊断协议；UNINEXT-L 等官方权重可获得后再接入，不能用其他 checkpoint 冒充。
