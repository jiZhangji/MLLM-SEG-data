#!/usr/bin/env python3
"""Resumable dataset downloader for Seg-Zero, Text4Seg and STAMP.

HF_TOKEN is read from the environment by huggingface_hub. It is never persisted
by this script. Public repositories can normally be downloaded without a token.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


RESOURCES = {
    "coco_train2014": {
        "kind": "hf_zip",
        "repo": "GAIA-URJC/COCO_2014",
        "filename": "train2014.zip",
        "dest": "shared/coco",
        "description": "MS COCO 2014 training images (HF mirror), shared by RefCOCO variants",
    },
    "coco_annotations2014": {
        "kind": "hf_zip",
        "repo": "GAIA-URJC/COCO_2014",
        "filename": "annotations_trainval2014.zip",
        "dest": "shared/coco",
        "description": "MS COCO 2014 annotations (HF mirror)",
    },
    "refcoco_family": {
        "kind": "hf_snapshot",
        "repo": "PaDT-MLLM/RefCOCO",
        "dest": "annotations/refcoco_family",
        "description": "RefCOCO, RefCOCO+ and RefCOCOg JSON annotations",
    },
    "refclef": {
        "kind": "hf_snapshot",
        "repo": "yiqun/referit",
        "dest": "datasets/refclef_referit",
        "description": "Community mirror of RefCLEF/ReferItGame",
    },
    "grefcoco": {
        "kind": "hf_snapshot",
        "repo": "FudanCVL/gRefCOCO",
        "dest": "annotations/grefcoco",
        "description": "gRefCOCO annotations; reuses COCO train2014 images",
    },
    "reasonseg": {
        "kind": "hf_snapshot",
        "repo": "fcxfcx/ReasonSeg",
        "dest": "datasets/reasonseg",
        "description": "ReasonSeg train/val/test community mirror",
    },
    "reasonseg_test_segzero": {
        "kind": "hf_snapshot",
        "repo": "Ricky06662/ReasonSeg_test",
        "dest": "datasets/reasonseg_test_segzero",
        "description": "ReasonSeg test package published for Seg-Zero",
    },
    "llava_665k": {
        "kind": "hf_file",
        "repo": "liuhaotian/LLaVA-Instruct-150K",
        "filename": "llava_v1_5_mix665k.json",
        "dest": "annotations/llava_665k",
        "description": "LLaVA-v1.5 665K instruction JSON (images are not included)",
    },
    "cocostuff164k": {
        "kind": "hf_snapshot",
        "repo": "GATE-engine/COCOStuff164K",
        "dest": "optional/open_vocabulary/cocostuff164k",
        "description": "COCO-Stuff 164K community HF mirror",
    },
}

GROUPS = {
    "common": ["coco_train2014", "coco_annotations2014", "refcoco_family"],
    "segzero": ["coco_train2014", "coco_annotations2014", "refcoco_family", "reasonseg_test_segzero"],
    "text4seg": ["coco_train2014", "coco_annotations2014", "refcoco_family", "refclef", "grefcoco"],
    "stamp": ["coco_train2014", "coco_annotations2014", "refcoco_family", "refclef", "grefcoco", "reasonseg"],
    "reasonseg": ["reasonseg"],
    "dialogue": ["llava_665k"],
    "open_vocab": ["cocostuff164k"],
}


class DownloadProgress(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  found archive: {destination}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {}
    mode = "wb"
    existing = partial.stat().st_size if partial.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", 200)
        if existing and status != 206:
            existing = 0
            mode = "wb"
        remaining = int(response.headers.get("Content-Length", 0))
        total = existing + remaining
        with open(partial, mode) as output, tqdm(
            total=total, initial=existing, unit="B", unit_scale=True,
            desc=destination.name,
        ) as bar:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                bar.update(len(chunk))
    partial.replace(destination)
    return destination


def extract_zip(archive: Path, destination: Path, keep_archive: bool) -> None:
    marker = destination / f".{archive.name}.extracted"
    if marker.exists():
        print(f"  already extracted: {archive.name}")
        return
    print(f"  extracting {archive.name} -> {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in tqdm(zf.infolist(), desc="extract", unit="file"):
            zf.extract(member, destination)
    marker.write_text("ok\n", encoding="utf-8")
    if not keep_archive:
        archive.unlink(missing_ok=True)


def download_resource(name: str, root: Path, keep_archives: bool) -> dict:
    item = RESOURCES[name]
    destination = root / item["dest"]
    print(f"\n[{name}] {item['description']}")
    if item["kind"] == "hf_snapshot":
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=item["repo"], repo_type="dataset",
            local_dir=destination,
        )
    elif item["kind"] == "hf_file":
        destination.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=item["repo"], repo_type="dataset",
            filename=item["filename"], local_dir=destination,
        )
    elif item["kind"] == "hf_zip":
        destination.mkdir(parents=True, exist_ok=True)
        archive = Path(hf_hub_download(
            repo_id=item["repo"], repo_type="dataset",
            filename=item["filename"], local_dir=destination,
        ))
        extract_zip(archive, destination, keep_archives)
    elif item["kind"] == "url_zip":
        destination.mkdir(parents=True, exist_ok=True)
        archive = download_url(item["url"], destination / Path(item["url"]).name)
        extract_zip(archive, destination, keep_archives)
    else:
        raise ValueError(f"Unsupported resource kind: {item['kind']}")
    return {"name": name, "status": "ok", "path": str(destination.resolve())}


def available_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download datasets used by Seg-Zero, Text4Seg and STAMP."
    )
    parser.add_argument(
        "groups", nargs="*", metavar="GROUP", default=None,
        help="Dataset groups to download (default: common).",
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Output root directory.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep ZIP archives after extraction.")
    parser.add_argument("--list", action="store_true", help="List groups and exit.")
    parser.add_argument("--continue-on-error", action="store_true", help="Try remaining downloads after an error.")
    args = parser.parse_args()

    if args.list:
        for group, names in GROUPS.items():
            print(f"{group:12s} " + ", ".join(names))
        return 0

    valid_groups = set(GROUPS) | {"all"}
    invalid_groups = sorted(set(args.groups or []) - valid_groups)
    if invalid_groups:
        parser.error(
            "unknown group(s): " + ", ".join(invalid_groups)
            + "; choose from: " + ", ".join(sorted(valid_groups))
        )

    selected = []
    groups = args.groups or ["common"]
    if "all" in groups:
        groups = list(GROUPS)
    for group in groups:
        for name in GROUPS[group]:
            if name not in selected:
                selected.append(name)

    root = args.root.expanduser().resolve()
    print(f"Output root: {root}")
    print(f"Free space: {available_bytes(root) / (1024**3):.1f} GiB")
    print("Resources: " + ", ".join(selected))
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN detected from environment (value is not printed or stored).")

    results = []
    failed = False
    for name in selected:
        try:
            results.append(download_resource(name, root, args.keep_archives))
        except Exception as exc:
            failed = True
            results.append({"name": name, "status": "failed", "error": str(exc)})
            print(f"ERROR [{name}]: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    root.mkdir(parents=True, exist_ok=True)
    (root / "download_status.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nStatus written to {root / 'download_status.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
