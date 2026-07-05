from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def setup_stamp_imports(stamp_code_dir: Path) -> None:
    stamp_code_dir = stamp_code_dir.resolve()
    if str(stamp_code_dir) not in sys.path:
        sys.path.insert(0, str(stamp_code_dir))


def load_segmenter_class(stamp_code_dir: Path):
    setup_stamp_imports(stamp_code_dir)
    try:
        from segment_predictor import GenerativeSegmenter  # type: ignore
    except Exception:
        from segment_predictor_cache import GenerativeSegmenter  # type: ignore
    return GenerativeSegmenter


def safe_source(obj: Any, max_chars: int = 30000) -> str:
    try:
        text = inspect.getsource(obj)
    except Exception as exc:
        return f"<source unavailable: {type(exc).__name__}: {exc}>"
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n# ... truncated ...\n"
    return text


def grep_file(path: Path, needles: list[str], context: int = 4) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: dict[str, list[str]] = {}
    for needle in needles:
        snippets = []
        for idx, line in enumerate(lines):
            if needle in line:
                start = max(0, idx - context)
                end = min(len(lines), idx + context + 1)
                snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
                snippets.append(snippet)
        hits[needle] = snippets[:20]
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    stamp_code_dir = args.stamp_code_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    GenerativeSegmenter = load_segmenter_class(stamp_code_dir)
    class_file = Path(inspect.getfile(GenerativeSegmenter)).resolve()
    methods = [
        name
        for name, value in inspect.getmembers(GenerativeSegmenter)
        if callable(value) and not name.startswith("__")
    ]

    report = {
        "stamp_code_dir": str(stamp_code_dir),
        "generative_segmenter_file": str(class_file),
        "generative_segmenter_methods": methods,
        "has_generate_with_segmentation": hasattr(GenerativeSegmenter, "generate_with_segmentation"),
        "has_refinement_export": any(
            hasattr(GenerativeSegmenter, name)
            for name in [
                "generate_with_refinement_outputs",
                "generate_with_refinement",
                "export_refinement_outputs",
                "forward_for_refinement",
            ]
        ),
    }

    source_paths = [
        class_file,
        stamp_code_dir / "segment_predictor.py",
        stamp_code_dir / "segment_predictor_cache.py",
        stamp_code_dir / "model" / "modeling_qwen2_vl.py",
    ]
    needles = [
        "generate_with_segmentation",
        "segmentation",
        "seg_mask",
        "mask_logits",
        "logits",
        "hidden_states",
        "output_hidden_states",
        "image_grid_thw",
        "grid_thw",
        "<|seg|>",
        "[MASK]",
        "mask_token",
        "seg_token",
    ]
    grep_report = {str(path): grep_file(path, needles) for path in source_paths if path.exists()}

    (output_dir / "stamp_refinement_inspection.json").write_text(
        json.dumps({**report, "grep": grep_report}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "GenerativeSegmenter_source.py.txt").write_text(
        safe_source(GenerativeSegmenter),
        encoding="utf-8",
    )
    if hasattr(GenerativeSegmenter, "generate_with_segmentation"):
        (output_dir / "generate_with_segmentation_source.py.txt").write_text(
            safe_source(getattr(GenerativeSegmenter, "generate_with_segmentation")),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2))
    print("")
    print("Wrote inspection files:")
    print(output_dir / "stamp_refinement_inspection.json")
    print(output_dir / "GenerativeSegmenter_source.py.txt")
    print(output_dir / "generate_with_segmentation_source.py.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
