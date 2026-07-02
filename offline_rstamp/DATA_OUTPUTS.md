# Data Outputs

为避免 baseline 和 R-STAMP 数据混在一起，数据转换脚本使用三个独立输出目录。

## Debug

命令：

```bash
bash offline_rstamp/run/30_prepare_stamp_data_debug.sh
```

输出：

```text
code/STAMP/playground/data/json_files_debug/
code/STAMP/playground/data/masks_debug/
```

用途：

- 每个数据集只转换少量样本；
- 用来确认 JSON 和 mask 生成是否正常；
- 不用于正式训练。

## STAMP baseline

命令：

```bash
bash offline_rstamp/run/31_prepare_stamp_data_full.sh
```

输出：

```text
code/STAMP/playground/data/json_files_baseline/
code/STAMP/playground/data/masks_baseline/
```

用途：

- 官方 STAMP baseline；
- 不包含 `structured_prior`；
- 用于和 R-STAMP 做公平对比。

## R-STAMP

命令：

```bash
bash offline_rstamp/run/32_prepare_rstamp_data_full.sh
```

输出：

```text
code/STAMP/playground/data/json_files_rstamp/
code/STAMP/playground/data/masks_rstamp/
```

用途：

- R-STAMP 改进方法；
- JSON 中包含 `structured_prior` 和 `structured_prior_text`；
- 后续训练脚本需要显式读取这个目录。

## 重要提醒

不要让 baseline 和 R-STAMP 读取同一个 JSON 目录，否则实验会混乱。

后续训练脚本需要用参数或环境变量指定：

```bash
STAMP_JSON_DIR=.../json_files_baseline
STAMP_MASK_DIR=.../masks_baseline
```

或者：

```bash
STAMP_JSON_DIR=.../json_files_rstamp
STAMP_MASK_DIR=.../masks_rstamp
```

