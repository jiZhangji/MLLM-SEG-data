# 2×80GB Offline Runbook

> 目标：服务器无网、数据已下载，手动放入代码和模型后，按步骤跑 STAMP baseline，再做 R-STAMP 改进。

## 0. 你需要手动准备的东西

服务器上至少需要：

```text
MLLM-SEG-data/                 # 当前仓库，包含 offline_rstamp
数据根目录/                     # 你已经下载好的 35GB 数据
STAMP/                         # 官方 STAMP 代码
模型权重/                       # STAMP 权重、SAM 权重、必要的 LLM/HF cache
Python/Conda 环境               # torch/deepspeed/transformers/peft 等
```

如果服务器完全无网，依赖也要提前准备好。最稳妥的方式是在有网机器上按 STAMP 官方 README 建好环境，或者提前下载 wheelhouse 后拷贝到服务器。

## 1. 建议目录

```bash
/data/MLLM-SEG                         # 数据根目录
/work/MLLM-SEG-exp                     # 实验工作区
/work/MLLM-SEG-exp/code/STAMP          # 官方 STAMP
/work/MLLM-SEG-exp/code/R-STAMP        # 新 idea scaffold
/work/MLLM-SEG-exp/models              # 模型权重
```

当前服务器实际目录已经调整为：

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/
├── code/STAMP/
├── code/R-STAMP/
├── data/
├── models/
├── outputs/
└── logs/
```

当前 canonical data root 是：

```bash
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/data
```

可用下面命令检查：

```bash
bash offline_rstamp/run/20_check_current_layout.sh
```

## 2. 配置路径

```bash
cd /path/to/MLLM-SEG-data
cp offline_rstamp/paths.example.sh offline_rstamp/paths.local.sh
vim offline_rstamp/paths.local.sh
```

重点改：

```bash
export MLLM_SEG_DATA_ROOT="/data/MLLM-SEG"
export MLLM_SEG_WORKSPACE="/work/MLLM-SEG-exp"
export STAMP_CODE_DIR="${MLLM_SEG_WORKSPACE}/code/STAMP"
export RSTAMP_CODE_DIR="${MLLM_SEG_WORKSPACE}/code/R-STAMP"
export MODEL_ROOT="${MLLM_SEG_WORKSPACE}/models"
export STAMP_MODEL_PATH="${MODEL_ROOT}/STAMP-7B-lora"
export SAM_CKPT="${MODEL_ROOT}/sam_vit_h_4b8939.pth"
export CUDA_VISIBLE_DEVICES="0,1"
```

## 3. 检查数据 + 整理工作区

```bash
bash offline_rstamp/run/00_prepare_workspace.sh
```

成功后会生成：

```text
/work/MLLM-SEG-exp/data_layout_report.json
/work/MLLM-SEG-exp/data/
/work/MLLM-SEG-exp/code/
/work/MLLM-SEG-exp/models/
/work/MLLM-SEG-exp/outputs/
/work/MLLM-SEG-exp/logs/
```

如果这一步报缺数据，先补数据，不要继续训练。

## 4. 放入代码

把官方 STAMP 放到：

```bash
/work/MLLM-SEG-exp/code/STAMP
```

然后安装 R-STAMP scaffold：

```bash
bash offline_rstamp/run/10_install_rstamp_scaffold.sh
```

检查代码：

```bash
source offline_rstamp/paths.local.sh
python offline_rstamp/scripts/check_code_repos.py \
  --code-root "$MLLM_SEG_WORKSPACE/code" \
  --report "$MLLM_SEG_WORKSPACE/code_repo_report.json"
```

## 5. 检查 Python 环境

```bash
python offline_rstamp/scripts/check_python_env.py
```

如果缺包，在无网服务器上不要直接 pip install，需要用你提前准备好的离线 wheel 或已有 conda 环境解决。

## 6. STAMP 数据转换

数据下载后通常还要运行 STAMP 自己的数据生成脚本。官方 README 里给的是：

```bash
cd "$STAMP_CODE_DIR"
python STAMP/data/create_refcoco_new.py
```

注意：这个脚本可能有硬编码路径。建议先打开它，把数据根目录改成：

```bash
$MLLM_SEG_WORKSPACE/data
```

或者：

```bash
$MLLM_SEG_DATA_ROOT
```

这一点非常关键。raw data 不等于 STAMP training JSON。

## 7. 跑 STAMP baseline eval

```bash
bash offline_rstamp/run/01_stamp_eval.sh
```

如果官方 `scripts/eval_ref.sh` 里路径不对，优先复制一份：

```bash
cp "$STAMP_CODE_DIR/scripts/eval_ref.sh" "$STAMP_CODE_DIR/scripts/eval_ref_local_2x80g.sh"
vim "$STAMP_CODE_DIR/scripts/eval_ref_local_2x80g.sh"
```

然后把 `offline_rstamp/run/01_stamp_eval.sh` 里调用的脚本换成 local 版本。

## 8. 跑 STAMP 训练 baseline

```bash
bash offline_rstamp/run/02_stamp_train_2x80g.sh
```

2×80GB 建议：

```text
micro batch per GPU: 1-2
gradient accumulation: 16-32
bf16: on
gradient checkpointing: on
LoRA rank: 16 or 32
ZeRO: stage 2 first, 不够再 stage 3
```

如果官方脚本默认 4/8 卡，不要直接跑。先复制一个 2 卡版本：

```bash
cp "$STAMP_CODE_DIR/scripts/launch_all_7B.sh" "$STAMP_CODE_DIR/scripts/launch_all_7B_2x80g.sh"
vim "$STAMP_CODE_DIR/scripts/launch_all_7B_2x80g.sh"
```

主要改：

```bash
CUDA_VISIBLE_DEVICES=0,1
--nproc_per_node 2
per_device_train_batch_size 1
gradient_accumulation_steps 16
```

## 9. 接入 R-STAMP 的第一版实验

第一版不要直接做 RL，先做：

```text
STAMP + structured reasoning prior + LoRA SFT
```

你需要在 STAMP 的 data pipeline 中给每条样本加：

```json
"structured_prior_text": "<TARGET> ... </TARGET> <BOX> ... </BOX> <POS> ... </POS>"
```

如果 STAMP 生成的是 JSONL，可以先用通用脚本试一下：

```bash
python offline_rstamp/scripts/add_structured_prior_to_jsonl.py \
  --input-jsonl /path/to/stamp_train.jsonl \
  --output-jsonl /path/to/stamp_train_with_prior.jsonl \
  --rstamp-code-dir "$RSTAMP_CODE_DIR"
```

如果 STAMP 使用 JSON 而不是 JSONL，或者字段名不同，需要按实际格式轻微改这个脚本。

## 10. 实验顺序

建议按这个顺序做，别跳：

```text
E0: STAMP official inference
E1: STAMP official eval
E2: STAMP LoRA/SFT baseline on your data
E3: STAMP + structured prior SFT
E4: STAMP + structured prior + uncertainty refinement
E5: STAMP + structured prior + lightweight RL
```

每一步都保存：

```text
config
git commit hash
data report
training log
eval result
GPU memory
inference time
```

## 11. 最重要的提醒

不要一开始就改很多东西。你的 2×80GB 资源应该用来做一个很干净的故事：

> 在不牺牲 STAMP 并行 mask 速度的情况下，用短结构化 reasoning prior 提升复杂指令、OOD 和消歧能力。
