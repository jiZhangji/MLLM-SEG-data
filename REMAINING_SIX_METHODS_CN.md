# FreeRef 剩余六方法资源与运行说明

目标方法为 ReLA、PolyFormer-L、UNINEXT-L、LISA、GSVA 和 READ。所有任务均使用作者发布的训练后权重进行推理，再对最终输出执行 FreeRef；不重新训练基础模型。

## 1. 下载官方代码和权重

```bash
ROOT=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG
cd "$ROOT/MLLM-SEG-data" || exit 1

nohup env \
MLLM_SEG_ROOT="$ROOT" \
MIN_FREE_GB=160 \
ALLOW_INCOMPLETE=1 \
bash prepare_remaining_six_assets.sh \
> "$ROOT/outputs/freeref_remaining_six_assets.log" 2>&1 < /dev/null &
echo "Asset PID: $!"
```

`ALLOW_INCOMPLETE=1` 只允许脚本在记录人工阻塞项后正常结束，不会把阻塞项标记为完成。当前已知人工项是 GSVA 的许可基座合并和 UNINEXT 官方权重。

查看状态：

```bash
bash check_remaining_six_assets.sh
tail -n 100 -F "$ROOT/outputs/freeref_remaining_six_assets.log"
```

## 2. 推理环境

下载结束后、启动 GPU 任务前串行执行：

```bash
bash prepare_polyformer_freeref_assets.sh
bash prepare_lisa_freeref_env.sh
```

## 3. 当前可运行的四卡任务

```bash
nohup env \
MLLM_SEG_ROOT="$ROOT" \
RUN_LISA_FULL=1 \
bash run_ready_remaining_models_4gpu.sh \
> "$ROOT/outputs/freeref_ready_remaining_4gpu.log" 2>&1 < /dev/null &
echo "Evaluation PID: $!"
```

该调度器自动识别两张 H100 和两张 H200。三张卡运行 PolyFormer-L 八划分，另一张 H100 运行 LISA 八划分。任务可续跑，已有完整 `comparison.md` 或 `eval_summary.json` 的划分会被跳过。

```bash
watch -n 15 "cd '$ROOT/MLLM-SEG-data' && bash check_ready_remaining_models_4gpu.sh"
```

## 4. 尚未端到端就绪的方法

- ReLA：官方 evaluator 可保存聚合预测，但还需要逐表达式 Manifest 导入适配器。
- GSVA：需要许可的 Vicuna/LLaVA 合并基座；官方 evaluator 还需导出逐样本最终掩码。
- READ：官方 evaluator 默认只累计指标，需增加逐样本最终掩码导出。
- UNINEXT-L：官方权重链接受限，且需增加 RES 输出导入适配器。

这些方法在适配器和权重门控通过前不能写入论文 paired FreeRef 主表。
