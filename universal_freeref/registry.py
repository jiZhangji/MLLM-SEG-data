from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("methods.json")


def load_registry() -> list[dict[str, Any]]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("Method registry must be a JSON list.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect upstream methods supported by universal FreeRef.")
    parser.add_argument("--names-only", action="store_true")
    parser.add_argument("--cloneable-only", action="store_true")
    args = parser.parse_args()
    methods = load_registry()
    if args.cloneable_only:
        methods = [method for method in methods if method.get("repo_url")]
    if args.names_only:
        for method in methods:
            print(method["id"])
    else:
        print(json.dumps(methods, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
