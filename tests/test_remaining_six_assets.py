from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "prepare_remaining_six_assets.sh"
CHECK = ROOT / "check_remaining_six_assets.sh"
POLYFORMER_FULL = ROOT / "run_polyformer_freeref_full_eval.sh"
READY_RUNNER = ROOT / "run_ready_remaining_models_4gpu.sh"
INSTANCE_RUNNER = ROOT / "run_remaining_six_experiments_instance.sh"


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


def test_legacy_ready_runner_delegates_to_complete_ready_phase() -> None:
    text = READY_RUNNER.read_text(encoding="utf-8")
    assert "run_remaining_six_experiments_4gpu.sh" in text
    assert "REMAINING_SIX_PHASE=ready" in text
    assert "READ, PolyFormer-L" in text
    assert "GSVA when its licensed base is ready" in text


def test_complete_four_gpu_runner_covers_ready_and_rela_phases() -> None:
    text = (ROOT / "run_remaining_six_experiments_4gpu.sh").read_text(encoding="utf-8")
    assert "H100_GPUS" in text and "H200_GPUS" in text
    for runner in (
        "run_read_freeref_full_eval.sh",
        "run_polyformer_freeref_full_eval.sh",
        "run_lisa_freeref_eval.sh",
        "run_gsva_freeref_full_eval.sh",
        "run_rela_classic_training_4gpu.sh",
        "run_rela_freeref_full_eval.sh",
    ):
        assert runner in text
    assert "UNINEXT-L remains gated" in text
    assert "rela-train" in text and "rela-eval" in text
    assert "run_remaining_six_experiments_instance.sh" in text
    assert "REMAINING_SIX_INSTANCE_ROLE=h100" in text
    assert "REMAINING_SIX_INSTANCE_ROLE=h200" in text


def test_two_instance_runner_assigns_models_and_rela_without_cross_node_nccl() -> None:
    text = INSTANCE_RUNNER.read_text(encoding="utf-8")
    assert "PolyFormer-L -> ${GPU0}; LISA diagnostic -> ${GPU1}" in text
    assert "READ -> ${GPU0}; GSVA -> ${GPU1}" in text
    assert "RefCOCO+ -> ${GPU0},${GPU1}" in text
    assert "RefCOCO -> ${GPU0}; RefCOCOg -> ${GPU1}" in text
    assert 'specs="refcoco+|val refcoco+|testA refcoco+|testB"' in text
    assert "RELA_FINALIZE_FULL=0" in text
    assert "RELA_FINALIZE_ONLY=1" in text
    assert "torchrun" not in text


def test_runtime_preparation_is_role_scoped_for_two_instances() -> None:
    text = (ROOT / "prepare_remaining_six_runtimes.sh").read_text(encoding="utf-8")
    assert "REMAINING_SIX_INSTANCE_ROLE" in text
    assert "h100)" in text and "h200)" in text
    assert "prepare_polyformer_freeref_assets.sh" in text
    assert "prepare_read_freeref_assets.sh" in text


def test_node_local_runtime_markers_include_the_hostname() -> None:
    for filename in (
        "prepare_read_freeref_env.sh",
        "prepare_gsva_freeref_env.sh",
        "prepare_rela_freeref_env.sh",
    ):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert 'HOST_TAG="${FREEREF_HOST_TAG:-$(hostname)}"' in text
        assert "${HOST_TAG}.ready" in text
