from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from onepass_stamp.lora import DEFAULT_TARGETS, OnePassLoRALinear


@dataclass(frozen=True)
class StampAdapterConfig:
    path: Path
    rank: int
    alpha: float
    dropout: float
    use_rslora: bool
    target_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["path"] = str(self.path)
        values["target_modules"] = list(self.target_modules)
        return values


def read_stamp_adapter_config(path: str | Path) -> StampAdapterConfig:
    path = Path(path).resolve()
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if str(values.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Expected a LoRA PEFT adapter at {path}.")
    targets = tuple(str(value) for value in values.get("target_modules", []))
    missing_targets = sorted(set(DEFAULT_TARGETS) - set(targets))
    if missing_targets:
        raise ValueError(f"STAMP adapter is missing required LoRA targets: {missing_targets}.")
    return StampAdapterConfig(
        path=path,
        rank=int(values["r"]),
        alpha=float(values["lora_alpha"]),
        dropout=float(values.get("lora_dropout", 0.0)),
        use_rslora=bool(values.get("use_rslora", False)),
        target_modules=targets,
    )


def _weight_path(adapter_dir: Path) -> Path:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        return safetensors_path
    binary_path = adapter_dir / "adapter_model.bin"
    if binary_path.exists():
        return binary_path
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {adapter_dir}.")


def _canonical_lora_key(value: str) -> str | None:
    key = value.replace(".lora_A.default.weight", ".lora_a.weight")
    key = key.replace(".lora_B.default.weight", ".lora_b.weight")
    key = key.replace(".lora_A.weight", ".lora_a.weight")
    key = key.replace(".lora_B.weight", ".lora_b.weight")
    if not (key.endswith(".lora_a.weight") or key.endswith(".lora_b.weight")):
        return None
    marker = "layers."
    if marker in key:
        return key[key.index(marker) :]
    marker = "language_model."
    if marker in key:
        return key[key.index(marker) :]
    return key


def _target_lora_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, OnePassLoRALinear):
            continue
        for branch, tensor in (("lora_a", module.lora_a.weight), ("lora_b", module.lora_b.weight)):
            canonical = _canonical_lora_key(f"{module_name}.{branch}.weight")
            if canonical is None:
                raise RuntimeError(f"Cannot canonicalize injected LoRA module {module_name}.")
            if canonical in values:
                raise RuntimeError(f"Duplicate target LoRA key {canonical}.")
            values[canonical] = tensor
    if not values:
        raise ValueError("The OnePass model has no injected LoRA layers.")
    return values


def _classifier_key(keys: list[str], parameter_name: str) -> str | None:
    suffixes = (
        f".classifier.modules_to_save.default.{parameter_name}",
        f".classifier.{parameter_name}",
    )
    for suffix in suffixes:
        matches = [key for key in keys if key.endswith(suffix)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous classifier {parameter_name} tensors: {matches}.")
        if matches:
            return matches[0]
    return None


def _copy_tensor(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if target.shape != source.shape:
        raise ValueError(
            f"STAMP adapter tensor {name} has shape {tuple(source.shape)}, "
            f"expected {tuple(target.shape)}."
        )
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def load_stamp_adapter_initialization(
    model: torch.nn.Module,
    adapter_dir: str | Path,
    *,
    initialize_classifier: bool = True,
) -> dict[str, Any]:
    """Load PEFT STAMP LoRA tensors into the custom OnePass LoRA wrappers."""
    config = read_stamp_adapter_config(adapter_dir)
    weight_path = _weight_path(config.path)
    targets = _target_lora_tensors(model)

    if weight_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            source_keys = list(handle.keys())
            source_lora: dict[str, str] = {}
            for source_key in source_keys:
                canonical = _canonical_lora_key(source_key)
                if canonical is None:
                    continue
                if canonical in source_lora:
                    raise ValueError(f"Duplicate source LoRA key {canonical}.")
                source_lora[canonical] = source_key
            missing = sorted(set(targets) - set(source_lora))
            unexpected = sorted(set(source_lora) - set(targets))
            if missing or unexpected:
                raise ValueError(
                    "STAMP/OnePass LoRA layout mismatch; "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                    f"target_tensors={len(targets)}, source_tensors={len(source_lora)}."
                )
            for canonical, target in targets.items():
                _copy_tensor(target, handle.get_tensor(source_lora[canonical]), canonical)

            classifier_loaded = False
            classifier_keys: dict[str, str] = {}
            if initialize_classifier:
                classifier = model.mask_classifier
                for name, target in classifier.named_parameters(recurse=False):
                    source_key = _classifier_key(source_keys, name)
                    if source_key is None:
                        raise ValueError(f"STAMP adapter does not contain classifier parameter {name}.")
                    _copy_tensor(target, handle.get_tensor(source_key), source_key)
                    classifier_keys[name] = source_key
                classifier_loaded = True
    else:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        source_keys = list(checkpoint)
        source_lora = {}
        for source_key in source_keys:
            canonical = _canonical_lora_key(source_key)
            if canonical is not None:
                source_lora[canonical] = source_key
        missing = sorted(set(targets) - set(source_lora))
        unexpected = sorted(set(source_lora) - set(targets))
        if missing or unexpected:
            raise ValueError(
                f"STAMP/OnePass LoRA layout mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}."
            )
        for canonical, target in targets.items():
            _copy_tensor(target, checkpoint[source_lora[canonical]], canonical)
        classifier_loaded = False
        classifier_keys = {}
        if initialize_classifier:
            for name, target in model.mask_classifier.named_parameters(recurse=False):
                source_key = _classifier_key(source_keys, name)
                if source_key is None:
                    raise ValueError(f"STAMP adapter does not contain classifier parameter {name}.")
                _copy_tensor(target, checkpoint[source_key], source_key)
                classifier_keys[name] = source_key
            classifier_loaded = True

    return {
        "adapter": str(config.path),
        "weights": str(weight_path),
        "lora_tensors_loaded": len(targets),
        "lora_layers_loaded": len(targets) // 2,
        "classifier_loaded": classifier_loaded,
        "classifier_keys": classifier_keys,
        "adapter_config": config.to_dict(),
    }


__all__ = [
    "StampAdapterConfig",
    "load_stamp_adapter_initialization",
    "read_stamp_adapter_config",
]
