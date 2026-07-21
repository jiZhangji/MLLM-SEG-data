# LISA + FreeRef + 冻结 SAM-H 官方协议评测

## 插入位置

LISA 的 `[SEG]` token 先变成 prompt embedding，再由其内置 SAM-H 掩码解码器生成二维掩码。
因此第一次 SAM-H 之前不存在可供 FreeRef 优化的空间掩码。本文采用可实现且严格配对的路径：

```text
[SEG] embedding -> LISA 原生 SAM-H -> 初始掩码 -> FreeRef -> 第二次冻结 SAM-H
```

一次模型前向同时报告四条路径：

1. LISA；
2. LISA + FreeRef；
3. LISA + 第二次冻结 SAM-H；
4. LISA + FreeRef + 第二次冻结 SAM-H。

两次第二阶段解码共享同一个 `[SEG]` sparse prompt、SAM 图像特征和样本。区别仅在 dense mask
prompt 来自原始 LISA 概率还是 FreeRef 概率。FreeRef 位于第二次冻结 SAM-H 之前。

## 评测协议

- 使用 LISA 官方 `ValDataset`、REFER 标注、对话模板和阈值；
- 使用公开 `LISA-7B-v1`、本地 CLIP 和 SAM-H 权重；
- 全量覆盖 RefCOCO、RefCOCO+、RefCOCOg 八个 split；
- 每个表达式原子保存四分支结果，支持断点续跑；
- 公开 checkpoint 的结果与论文 Table 3 分开报告，因为 RefCOCO testA 基线为 70.40，未复现论文行。

## 两张 H100 全量运行

```bash
nohup bash run_lisa_official_freeref_sam_h100.sh \
  > ../outputs/lisa_official_freeref_sam_h100.log 2>&1 < /dev/null &
```

默认每张 H100 同时运行两个 split。状态检查：

```bash
bash check_lisa_official_freeref_sam_status.sh
```
