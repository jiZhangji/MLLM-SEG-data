from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a universal FreeRef JSONL manifest from saved predictions.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-kind", choices=("mask", "probability", "logits"), default="mask")
    parser.add_argument("--prediction-glob", default="**/*.png")
    parser.add_argument("--strip-suffix", default="")
    parser.add_argument("--image-template", default="{relative_stem}.jpg")
    parser.add_argument("--gt-template", default="{relative_stem}.png")
    parser.add_argument("--ignore-root", type=Path)
    parser.add_argument("--ignore-template", default="{relative_stem}.png")
    parser.add_argument("--uncertainty-root", type=Path)
    parser.add_argument("--uncertainty-template", default="{relative_stem}.npy")
    parser.add_argument("--array-key")
    parser.add_argument("--uncertainty-key")
    parser.add_argument("--foreground-channel", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--relative-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def _render(template: str, relative_stem: str, name: str) -> str:
    return template.format(relative_stem=relative_stem, stem=Path(relative_stem).name, name=name)


def _manifest_path(path: Path, output: Path, relative: bool) -> str:
    path = path.resolve()
    if not relative:
        return str(path)
    return Path(os.path.relpath(path, output.parent.resolve())).as_posix()


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root.expanduser().resolve()
    predictions = sorted(path for path in prediction_root.glob(args.prediction_glob) if path.is_file())
    if not predictions:
        raise FileNotFoundError(
            f"No predictions matched {args.prediction_glob!r} below {prediction_root}."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for prediction in predictions:
        relative = prediction.relative_to(prediction_root)
        relative_stem = relative.with_suffix("").as_posix()
        if args.strip_suffix:
            if not relative_stem.endswith(args.strip_suffix):
                continue
            relative_stem = relative_stem[: -len(args.strip_suffix)]
        name = Path(relative_stem).name
        image = args.image_root / _render(args.image_template, relative_stem, name)
        target = args.gt_root / _render(args.gt_template, relative_stem, name)
        required = [image, target]
        if not all(path.is_file() for path in required):
            missing.append(
                {
                    "prediction": str(prediction),
                    "image": str(image),
                    "gt_mask": str(target),
                }
            )
            continue
        row = {
            "name": relative_stem,
            "method": args.method,
            "split": args.split,
            "image": _manifest_path(image, args.output, args.relative_paths),
            "gt_mask": _manifest_path(target, args.output, args.relative_paths),
            "prediction": _manifest_path(prediction, args.output, args.relative_paths),
            "prediction_kind": args.prediction_kind,
            "foreground_channel": args.foreground_channel,
            "threshold": args.threshold,
        }
        if args.array_key:
            row["array_key"] = args.array_key
        if args.ignore_root:
            ignore = args.ignore_root / _render(args.ignore_template, relative_stem, name)
            if ignore.is_file():
                row["ignore_mask"] = _manifest_path(ignore, args.output, args.relative_paths)
        if args.uncertainty_root:
            uncertainty = args.uncertainty_root / _render(args.uncertainty_template, relative_stem, name)
            if uncertainty.is_file():
                row["uncertainty"] = _manifest_path(uncertainty, args.output, args.relative_paths)
                if args.uncertainty_key:
                    row["uncertainty_key"] = args.uncertainty_key
        rows.append(row)
    if missing and not args.allow_missing:
        example = json.dumps(missing[:3], indent=2, ensure_ascii=True)
        raise FileNotFoundError(
            f"{len(missing)} prediction(s) have missing image/GT pairs. Examples:\n{example}\n"
            "Fix the templates or pass --allow-missing to emit only complete rows."
        )
    if not rows:
        raise ValueError("No complete prediction/image/GT rows were produced.")
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "rows": len(rows), "missing": len(missing)},
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
