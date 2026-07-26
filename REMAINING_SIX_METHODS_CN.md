# FreeRef 剩余六个方法：双实例四卡运行

硬件拓扑：

- H100 实例：2×H100
- H200 实例：2×H200
- 两个实例必须能访问同一个 `MLLM_SEG_ROOT`、模型目录和输出目录

入口脚本会检查当前实例的 GPU 型号并自动选择 `h100` 或 `h200` 角色，不使用跨节点 NCCL。两个实例各自启动本地任务，通过共享文件系统汇合 checkpoint 和八划分结果。

## 当前状态和分工

| 方法 | 实例/GPU | 权重状态 | 论文状态 |
|---|---|---|---|
| PolyFormer-L | H100:0 | 三套官方数据集权重 | 基线门控后可作为候选 |
| LISA | H100:1 | 公开 v1 权重 | 仅诊断；已知基线门控失败 |
| READ | H200:0 | 官方完整公开权重 | 基线门控后可作为候选 |
| GSVA | H200:1 | 官方 `ft-res`；另需授权 LLaMA-7B 合并基座 | 基线门控后可作为候选 |
| ReLA RefCOCO+ | 两张 H100 | 作者未公开，需本地复训 | 基线门控后可作为候选 |
| ReLA RefCOCO / RefCOCOg | H200:0 / H200:1 | 作者未公开，需本地复训 | 基线门控后可作为候选 |
| UNINEXT-L | 暂不启动 | 官方 Stage-2 ConvNeXt-L 链接仍为 403 | 阻塞 |

所有 `paper_candidate` 都会与论文中的八个基线 cIoU 逐项比较。任一划分的绝对误差超过 2.0 个百分点时，结果自动降为 `diagnostic_only`。

## 1. 两个实例都先更新代码

分别在 H100、H200 实例执行：

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
REPO="$ROOT/MLLM-SEG-data"
cd "$REPO" || exit 1
git pull --rebase --autostash https://github.com/jiZhangji/MLLM-SEG-data.git main
git rev-parse --short HEAD
mkdir -p "$ROOT/outputs"
```

## 2. 下载代码和权重：只需启动一次

在任意一个实例执行；不要在两个实例同时启动下载器：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  MIN_FREE_GB=160 \
  ALLOW_INCOMPLETE=1 \
  bash prepare_remaining_six_assets.sh \
  > "$ROOT/outputs/freeref_remaining_six_assets.log" 2>&1 < /dev/null &
echo "asset PID=$!"
```

```bash
tail -n 100 -F "$ROOT/outputs/freeref_remaining_six_assets.log"
bash check_remaining_six_assets.sh
```

`ALLOW_INCOMPLETE=1` 只允许记录人工阻塞项后退出，不会把 GSVA 授权基座、ReLA 未公开经典权重或 UNINEXT-L 权重标为完成。

## 3. 分实例安装运行环境

若两个实例共享同一套 Conda 目录，请先完成 H100 环境，再安装 H200 环境，避免锁冲突。环境完成标记包含 hostname，不会把另一个实例误判为已经安装。

H100 实例：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_INSTANCE_ROLE=h100 \
  bash prepare_remaining_six_runtimes.sh \
  > "$ROOT/outputs/freeref_runtimes_h100.log" 2>&1 < /dev/null &
echo "H100 runtime PID=$!"
```

H200 实例：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_INSTANCE_ROLE=h200 \
  bash prepare_remaining_six_runtimes.sh \
  > "$ROOT/outputs/freeref_runtimes_h200.log" 2>&1 < /dev/null &
echo "H200 runtime PID=$!"
```

H100 只安装 PolyFormer、LISA、ReLA；H200 只安装 READ、GSVA、ReLA。GSVA 官方程序即使评测 RefCOCO 也会构造辅助 gRefCOCO dataset，因此脚本会复用 `$ROOT/data/annotations/grefcoco`。

## 4. GSVA 授权基座：仅在 H200 实例执行

```bash
cd "$REPO"
export GSVA_LLAMA7B_BASE=/absolute/path/to/authorized/llama-7b-hf
MLLM_SEG_ROOT="$ROOT" bash prepare_gsva_llava_legacy.sh
```

没有该基座时，H200 调度器会跳过 GSVA，并让 READ 同时使用两张 H200。

## 5. 第一阶段：两个实例同时启动 ready

H100 实例：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=ready \
  bash run_remaining_six_experiments_4gpu.sh \
  > "$ROOT/outputs/remaining_six_h100_ready.nohup.log" 2>&1 < /dev/null &
echo "H100 ready PID=$!"
```

H200 实例：

```bash
cd "$REPO"
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=ready \
  bash run_remaining_six_experiments_4gpu.sh \
  > "$ROOT/outputs/remaining_six_h200_ready.nohup.log" 2>&1 < /dev/null &
echo "H200 ready PID=$!"
```

H100 同时运行 PolyFormer-L 和 LISA；H200 同时运行 READ 和 GSVA。每个方法内部支持断点续跑。

## 6. 第二阶段：两个实例同时训练 ReLA

等待两个 `ready` 阶段都结束，然后分别执行相同入口。

H100 实例：

```bash
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=rela-train \
  bash "$REPO/run_remaining_six_experiments_4gpu.sh" \
  > "$ROOT/outputs/remaining_six_h100_rela_train.nohup.log" 2>&1 < /dev/null &
echo "H100 ReLA PID=$!"
```

H200 实例：

```bash
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=rela-train \
  bash "$REPO/run_remaining_six_experiments_4gpu.sh" \
  > "$ROOT/outputs/remaining_six_h200_rela_train.nohup.log" 2>&1 < /dev/null &
echo "H200 ReLA PID=$!"
```

默认训练设置：Swin-B、BERT-base、480×480、全局 batch 24、AdamW、学习率 `1e-5`、150k iterations。存在 `last_checkpoint` 时自动续训。

## 7. 第三阶段：两个实例同时评测各自的 ReLA 划分

三套 checkpoint 全部完成后：

H100 实例：

```bash
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=rela-eval \
  bash "$REPO/run_remaining_six_experiments_4gpu.sh" \
  > "$ROOT/outputs/remaining_six_h100_rela_eval.nohup.log" 2>&1 < /dev/null &
echo "H100 ReLA eval PID=$!"
```

H200 实例：

```bash
nohup env \
  MLLM_SEG_ROOT="$ROOT" \
  REMAINING_SIX_PHASE=rela-eval \
  bash "$REPO/run_remaining_six_experiments_4gpu.sh" \
  > "$ROOT/outputs/remaining_six_h200_rela_eval.nohup.log" 2>&1 < /dev/null &
echo "H200 ReLA eval PID=$!"
```

H100 负责 RefCOCO+ 三个划分；H200 负责 RefCOCO 三个和 RefCOCOg 两个划分。此阶段只生成各自结果，不会抢先汇总。

两个实例的 `rela-eval` 都结束后，在任意一个实例执行一次：

```bash
MLLM_SEG_ROOT="$ROOT" \
REMAINING_SIX_PHASE=rela-finalize \
bash "$REPO/run_remaining_six_experiments_4gpu.sh"
```

`rela-finalize` 只读取共享目录中的八份 summary，不会重新推理；缺少任何划分时会报错。

## 8. 监控和结果

任一实例：

```bash
watch -n 20 "cd '$REPO' && MLLM_SEG_ROOT='$ROOT' bash check_remaining_six_experiments_4gpu.sh"
```

Table 1(b) 输出：

```text
$ROOT/outputs/read_freeref_full/combined/table1b_row.tsv
$ROOT/outputs/gsva_freeref_full/combined/table1b_row.tsv
$ROOT/outputs/polyformer_freeref_full/combined/table1b_row.tsv
$ROOT/outputs/rela_freeref_full/combined/table1b_row.tsv
$ROOT/outputs/universal_freeref_lisa_all8/combined/table1b_row.tsv
```

必须同时检查 `table1b_row.json` 中：

```text
baseline_gate_passed: true
eligibility: paper_candidate
```

LISA 当前固定为诊断协议；UNINEXT-L 等官方权重可获得后再接入，不能用其他 checkpoint 冒充。
