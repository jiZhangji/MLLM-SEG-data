# MLLM-SEG datasets downloader

Resumable downloader for the datasets used by **Seg-Zero**, **Text4Seg**, and
**STAMP**. Large datasets are not committed to this repository.

## Quick start

```bash
git clone https://github.com/jiZhangji/MLLM-SEG-data.git
cd MLLM-SEG-data
chmod +x download.sh

# Smallest useful starting point: COCO + RefCOCO family
./download.sh common --root /data/MLLM-SEG

# Dataset combinations used by each paper
./download.sh segzero --root /data/MLLM-SEG
./download.sh text4seg --root /data/MLLM-SEG
./download.sh stamp --root /data/MLLM-SEG
```

Public datasets normally do not need authentication. If Hugging Face requires
authentication, create a **read-only** token and expose it only in the shell:

```bash
export HF_TOKEN='YOUR_READ_ONLY_TOKEN'
./download.sh stamp --root /data/MLLM-SEG
unset HF_TOKEN
```

Never put a token in this repository, a command-line argument, or a job script
that will be committed.

## Download groups

```bash
./download.sh --list
```

| Group | Contents |
|---|---|
| `common` | COCO train2014, COCO annotations, RefCOCO/+/g annotations |
| `segzero` | Common data plus the Seg-Zero ReasonSeg test package |
| `text4seg` | Common data plus RefCLEF/ReferIt and gRefCOCO |
| `stamp` | Text4Seg data plus full ReasonSeg |
| `reasonseg` | Full ReasonSeg only |
| `dialogue` | LLaVA-v1.5 mix665k JSON only; referenced images are separate |
| `open_vocab` | COCO-Stuff 164K community mirror |

Several groups may be supplied together. Shared resources are downloaded only
once:

```bash
./download.sh stamp dialogue open_vocab --root /data/MLLM-SEG --continue-on-error
```

## Directory layout

```text
/data/MLLM-SEG/
├── shared/
│   └── coco/
│       ├── train2014/
│       └── annotations/
├── annotations/
│   ├── refcoco_family/
│   ├── grefcoco/
│   └── llava_665k/
├── datasets/
│   ├── refclef_referit/
│   ├── reasonseg/
│   └── reasonseg_test_segzero/
└── optional/
    └── open_vocabulary/cocostuff164k/
```

The RefCOCO family and gRefCOCO reuse images under `shared/coco/train2014`.
Do not download another embedded copy unless a specific training repository
requires its own packaging.

## Useful options

```bash
# Keep original COCO ZIPs after extraction
./download.sh common --keep-archives --root /data/MLLM-SEG

# Select another Python executable
PYTHON_BIN=/opt/conda/bin/python ./download.sh stamp --root /data/MLLM-SEG

# Direct Python invocation
python3 download.py segzero --root /data/MLLM-SEG
```

Downloads use Hugging Face's cache-aware transfer and HTTP range requests.
Rerunning the same command resumes or skips completed files. A machine-readable
summary is written to `download_status.json` in the output root.

## Important reproducibility notes

- `PaDT-MLLM/RefCOCO`, `yiqun/referit`, and `fcxfcx/ReasonSeg` are community
  mirrors. Dataset licenses and original project terms still apply.
- Seg-Zero's 9K training subset is sampled/generated from RefCOCOg by the
  paper's preprocessing code; it is not a separate raw image dataset here.
- Text4Seg's 800K and gRefCOCO's 419K counts are generated instruction samples,
  not unique images. Run the paper repository's preprocessing after download.
- STAMP likewise generates its mask-token training annotations from these raw
  datasets.
- `llava_v1_5_mix665k.json` contains instructions only. Its COCO, GQA,
  OCR-VQA, TextVQA, and Visual Genome images must be prepared separately if the
  dialogue-and-segmentation experiment is required.

## Upstream projects

- Seg-Zero: <https://github.com/dvlab-research/Seg-Zero>
- Text4Seg: <https://github.com/mc-lan/Text4Seg>
- STAMP: <https://github.com/HKUST-LongGroup/STAMP>

