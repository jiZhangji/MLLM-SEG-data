# FreeRef 缺失方法权重下载说明

## 下载范围

脚本只处理尚未完成 paired FreeRef 实验的方法：HIPIE、ReLA、PolyFormer-L、UNINEXT-L、PixelLM-7B、LISA-7B、GSVA-7B、READ-7B、Seg-Zero-7B、SegLLM 和 SegAgent。STAMP 与 Text4Seg 已有完整实验结果，不会重复下载或推理。

主权重统一保存到：

```text
<MLLM-SEG>/models/freeref_missing_methods/
```

下载状态、人工处理项和权重清单保存到：

```text
<MLLM-SEG>/outputs/freeref_weight_download/
```

## 一键后台下载

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && mkdir -p ../outputs && nohup env METHODS=all MIN_FREE_GB=80 bash download_missing_method_weights.sh > ../outputs/freeref_missing_method_weights.log 2>&1 < /dev/null & echo "Download PID: $!"
```

脚本会创建独立的轻量下载环境，不会修改 STAMP、Text4Seg 或其他方法的推理环境。Hugging Face、Google Drive 与 ModelScope 下载均支持重复执行；若进程中断，重新执行同一条命令即可继续。

只下载指定方法时：

```bash
METHODS="hipie polyformer read" bash download_missing_method_weights.sh
```

只查看下载计划，不访问网络：

```bash
DRY_RUN=1 ROOT=/tmp/freeref-download-check bash download_missing_method_weights.sh
```

## 查看状态

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data && bash check_missing_method_weights.sh
```

实时查看日志：

```bash
tail -n 100 -F /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/outputs/freeref_missing_method_weights.log
```

## 状态文件含义

- `download_status.tsv`：每个模型或公共依赖的下载、校验状态。
- `manual_downloads.tsv`：公共网盘自动导出失败或需要模型许可的项目。
- `download_plan.tsv`：所有官方来源、目标目录和模型名称。
- `weights_inventory.tsv`：已下载文件的大小与绝对路径。

下载脚本使用“完成标记 + 大文件校验”，不会把网盘返回的 HTML 页面误判为模型权重。SharePoint/OneDrive 文件夹若无法自动导出，会继续下载其他方法，并把准确链接写入人工处理清单。

## 已知限制

1. SegLLM 尚未确认可执行的官方代码和公开 checkpoint，因此固定标记为 `blocked`，不会用非官方复现冒充论文方法。
2. GSVA-7B 的作者 checkpoint 可以自动下载，但官方 Vicuna-7B 路线还依赖有许可要求的 LLaMA/Vicuna 基座与 LLaVA delta 合并。脚本下载公开 delta，并在人工清单中保留基座合并步骤。
3. `DOWNLOAD_DATASETS=0` 是默认值，脚本只下载模型权重。设置 `DOWNLOAD_DATASETS=1` 才会额外下载 Seg-Zero 与 SegAgent 发布的数据文件。
4. 下载完成只代表权重就绪；各官方仓库仍需独立环境和输出适配器，之后才能生成统一 Manifest 并进行 paired FreeRef 前后对比。
