from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an official-format PolyFormer evaluation TSV from REFER annotations."
    )
    parser.add_argument("--polyformer-code-dir", type=Path, required=True)
    parser.add_argument("--refer-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("refcoco", "refcoco+", "refcocog"), required=True)
    parser.add_argument("--split-by", choices=("unc", "umd", "google"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.offset < 0:
        raise ValueError("limit and offset must be non-negative.")

    code_dir = args.polyformer_code_dir.expanduser().resolve()
    refer_root = args.refer_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    required = {
        "PolyFormer source": code_dir / "data" / "poly_utils.py",
        "REFER instances": refer_root / args.dataset / "instances.json",
        "REFER refs": refer_root / args.dataset / f"refs({args.split_by}).p",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if not image_root.is_dir():
        missing.append(f"COCO image root: {image_root}")
    if missing:
        raise FileNotFoundError("PolyFormer evaluation inputs are incomplete:\n" + "\n".join(missing))

    sys.path.insert(0, str(code_dir))
    from data.poly_utils import (  # type: ignore[import-not-found]
        approximate_polygons,
        image_to_base64,
        interpolate_polygons,
        is_clockwise,
        polygons_to_string,
        reorder_points,
        revert_direction,
    )
    from refer.refer import REFER  # type: ignore[import-not-found]

    refer = REFER(str(refer_root), args.dataset, args.split_by)
    ref_ids = refer.getRefIds(split=args.split)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    expression_index = 0
    written = 0
    image_cache: dict[str, str] = {}
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for ref_id in tqdm(ref_ids, desc=f"PolyFormer TSV {args.dataset} {args.split}"):
            image_id = refer.getImgIds(ref_id)[0]
            image_info = refer.Imgs[image_id]
            image_path = image_root / image_info["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"COCO image is missing: {image_path}")
            cache_key = str(image_path)
            if cache_key not in image_cache:
                image_cache[cache_key] = image_to_base64(Image.open(image_path).convert("RGB"), format="jpeg")

            reference = refer.loadRefs(ref_id)[0]
            ref_mask = np.asarray(refer.getMask(reference)["mask"])
            mask_image = Image.fromarray((ref_mask == 1).astype(np.uint8), mode="P")
            mask_base64 = image_to_base64(mask_image, format="png")

            polygons = []
            for polygon in refer.getPolygon(reference)["polygon"]:
                polygon = polygon if is_clockwise(polygon) else revert_direction(polygon)
                polygons.append(reorder_points(polygon))
            polygons.sort(key=lambda value: (value[0] ** 2 + value[1] ** 2, value[0], value[1]))
            interpolated = interpolate_polygons(polygons)
            approximated = approximate_polygons(polygons, 5, 400)
            polygon_text = polygons_to_string(approximated)
            interpolated_text = polygons_to_string(interpolated)

            x, y, width, height = refer.getRefBox(ref_id)
            box_text = f"{x},{y},{x + width},{y + height}"
            ref_sentences = refer.Refs[ref_id]["sentences"]
            for sentence_number, sentence in enumerate(ref_sentences):
                if expression_index < args.offset:
                    expression_index += 1
                    continue
                if args.limit and written >= args.limit:
                    break
                unique_id = f"{ref_id}_{sentence_number}"
                row = [
                    unique_id,
                    str(image_id),
                    sentence["sent"],
                    box_text,
                    polygon_text,
                    image_cache[cache_key],
                    mask_base64,
                    interpolated_text,
                ]
                handle.write("\t".join(row) + "\n")
                expression_index += 1
                written += 1
            if args.limit and written >= args.limit:
                break

    if written == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("The requested PolyFormer slice contains no expressions.")
    temporary.replace(output)
    report = {
        "source": "official_refer_to_polyformer_tsv",
        "dataset": args.dataset,
        "split_by": args.split_by,
        "split": args.split,
        "samples": written,
        "offset": args.offset,
        "limit": args.limit,
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
