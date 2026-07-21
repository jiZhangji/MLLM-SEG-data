# LISA Latent FreeRef：在原生 SAM-H 前细化

## 正确的插入位置

LISA 在原生 SAM-H 前只有投影后的 `[SEG]` sparse embedding 和 SAM 图像 embedding，
没有二维预测掩码。因此本实现不使用第一次 SAM 的输出，也不做级联重解码，而是直接
从解码前特征构造训练自由的空间先验：

```text
[SEG] embedding + SAM image embedding
                  |
                  v
       token-image cosine similarity
                  |
                  v
       calibrated latent probability H0
                  |
                  v
               FreeRef
                  |
                  v
       refined dense prompt H*
                  |
                  v
       LISA native SAM-H (one decode)
                  |
                  v
             final mask
```

对位置 `x`，初始先验为：

```text
s(x) = cosine(q_SEG, F_SAM(x))
z(x) = (s(x) - mean(s) - beta * std(s)) / (tau * std(s) + eps)
H0(x) = sigmoid(z(x))
H* = FreeRef(I, H0)
```

其中 `q_SEG` 是 LISA 投影后的查询，`F_SAM` 是 SAM 图像特征，`beta` 控制空间先验的
前景偏置，`tau` 是温度。默认值分别为 `0.5` 和 `1.0`，不使用标注或学习参数。

## 配对分支

同一模型前向保存三个独立分支：

1. `LISA`：原生 sparse prompt 和空 dense prompt；
2. `LISA + latent prompt + SAM-H`：未经 FreeRef 的 `H0` 作为 dense prompt；
3. `LISA + latent FreeRef + SAM-H`：FreeRef 输出 `H*` 作为 dense prompt。

为了得到配对指标，评测器会执行三个相互独立的 SAM 解码。第三条方法路径不读取前两条
路径的掩码，在真实部署时只需要一次 SAM-H 解码。日志分别报告配对评测总时间和单条方法
路径的估算时间。

## 先做快速验证

建议先在 RefCOCO testA 的 64 张图像上验证特征形状和方向：

```bash
LISA_OFFICIAL_SPLITS='refcoco|unc|testA' \
LISA_OFFICIAL_LIMIT_IMAGES=64 \
LISA_H100_PARALLEL_PER_GPU=1 \
LISA_LATENT_FREEREF_OUTPUT_ROOT=../outputs/lisa_latent_freeref_smoke \
bash run_lisa_official_freeref_sam_h100.sh
```

确认 `LISA + latent FreeRef + SAM-H` 不明显退化后，再去掉这些环境变量运行全部八个 split。

## 状态检查

```bash
bash check_lisa_official_freeref_sam_status.sh
```

全量结果默认位于：

```text
outputs/lisa_latent_freeref_before_sam
```

公开 `LISA-7B-v1` 的结果属于同权重配对迁移实验，不应写成论文 Table 3 中微调权重的复现结果。
