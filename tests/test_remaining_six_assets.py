from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "prepare_remaining_six_assets.sh"
CHECK = ROOT / "check_remaining_six_assets.sh"
POLYFORMER_FULL = ROOT / "run_polyformer_freeref_full_eval.sh"
READY_RUNNER = ROOT / "run_ready_remaining_models_4gpu.sh"


def test_asset_entrypoint_uses_official_repo_and_weight_preparation() -> None:
    text = PREPARE.read_text(encoding="utf-8")
    assert 'METHODS="rela polyformer uninext lisa gsva read"' in text
    assert "prepare_universal_freeref_repos.sh" in text
    assert "download_missing_method_weights.sh" in text
    assert "DOWNLOAD_DATASETS=0" in text
    assert "run_" not in text.replace("run_artifact", "")
    assert "ALLOW_INCOMPLETE" in text
    assert "manual_downloads.tsv" in text


def test_asset_status_reports_all_six_methods_and_manual_blockers() -> None:
    text = CHECK.read_text(encoding="utf-8")
    for method in ("rela", "polyformer", "uninext", "lisa", "gsva", "read"):
        assert method in text
    assert "manual_downloads.tsv" in text
    assert "nvidia-smi" in text
    assert "df -h" in text


def test_polyformer_full_runner_covers_all_eight_splits_resumably() -> None:
    text = POLYFORMER_FULL.read_text(encoding="utf-8")
    expected = (
        "refcoco|unc|val",
        "refcoco|unc|testA",
        "refcoco|unc|testB",
        "refcoco+|unc|val",
        "refcoco+|unc|testA",
        "refcoco+|unc|testB",
        "refcocog|umd|val",
        "refcocog|umd|test",
    )
    for split in expected:
        assert split in text
    assert "POLYFORMER_CUDA_DEVICES" in text
    assert 'POLYFORMER_LIMIT=0' in text
    assert 'SKIP completed PolyFormer' in text
    assert "combined/comparison.md" in text


def test_ready_four_gpu_runner_is_explicit_about_supported_and_blocked_methods() -> None:
    text = READY_RUNNER.read_text(encoding="utf-8")
    assert "H100_GPUS" in text and "H200_GPUS" in text
    assert "run_polyformer_freeref_full_eval.sh" in text
    assert "run_lisa_freeref_eval.sh" in text
    assert "ReLA/GSVA/READ: blocked" in text
    assert "UNINEXT: blocked" in text
    assert "run_rela" not in text
    assert "run_gsva" not in text
    assert "run_read" not in text
