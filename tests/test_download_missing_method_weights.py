from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "download_missing_method_weights.sh"
CHECK_SCRIPT = ROOT / "check_missing_method_weights.sh"


def test_download_script_covers_every_missing_method_and_excludes_completed_ones():
    text = SCRIPT.read_text(encoding="utf-8")
    expected = {
        "hipie",
        "rela",
        "polyformer",
        "uninext",
        "pixellm",
        "lisa",
        "gsva",
        "read",
        "seg-zero",
        "segllm",
        "segagent",
    }
    for method in expected:
        assert f"selected {method}" in text or method in text.split("ALL_METHODS=(", 1)[1].split(")", 1)[0]

    all_methods = text.split("ALL_METHODS=(", 1)[1].split(")", 1)[0]
    assert "stamp" not in all_methods.lower()
    assert "text4seg" not in all_methods.lower()


def test_download_sources_are_official_and_status_is_size_validated():
    text = SCRIPT.read_text(encoding="utf-8")
    required_sources = (
        "KonstantinosKK/HIPIE",
        "maverickrzw/PixelLM-7B",
        "xinlai/LISA-7B-v1",
        "rui-qian/READ-LLaVA-v1.5-7B-for-fprefcoco",
        "Ricky06662/Seg-Zero-7B",
        "zzzmmz/SegAgent-Model",
        "15P6m5RI6HAQE2QXQXMAjw_oBsaPii7b3",
        "1Jw7GKiN-Y2tgLL6ueOKOKfikiWVOl2-n",
        "Et6GBDgKgPZDn5zp49yKwDYBd50EBTxaKs7R6Yuck_lf7g",
    )
    for source in required_sources:
        assert source in text

    assert "validate_artifact" in text
    assert ".freeref_download_complete" in text
    assert "manual_downloads.tsv" in text
    assert "grep -q -- '--remaining-ok'" in text
    assert "drive.usercontent.google.com" in text
    assert "seafile_share_files" in text
    assert "cloud.tsinghua.edu.cn" in text
    assert "cocolvis_vit_large.pth" in text
    simpleclick_block = text.split("simpleclick_url=", 1)[1].split(
        'if [[ "${DOWNLOAD_DATASETS}"', 1
    )[0]
    assert "1AjSGobd0Bq-50RFJfJnotAUWDsYKPC2B" in text
    assert "gdrive_file" in simpleclick_block
    assert "block_or_adopt_artifact" not in simpleclick_block


def test_status_script_reports_processes_disk_artifacts_and_manual_queue():
    text = CHECK_SCRIPT.read_text(encoding="utf-8")
    assert "Active download" in text
    assert "Disk" in text
    assert "Large files by method" in text
    assert "Manual or blocked items" in text
