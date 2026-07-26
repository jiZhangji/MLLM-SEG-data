from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from paper_assets.framework.upload_framework_outputs import (
    UPLOAD_PATTERNS,
    main as upload_main,
)


class FrameworkHfUploadTests(unittest.TestCase):
    def test_uploads_only_viewable_outputs(self):
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for filename in UPLOAD_PATTERNS[:6]:
                (output_dir / filename).write_bytes(b"fixture")
            components = output_dir / "freeref_framework_real_components"
            components.mkdir()
            (components / "input_scene.png").write_bytes(b"fixture")
            (output_dir / "selected_real_sample.npz").write_bytes(b"do-not-upload")

            commit = MagicMock()
            commit.commit_url = "https://huggingface.co/test/repo/commit/abc"
            api = MagicMock()
            api.upload_folder.return_value = commit
            argv = [
                "upload_framework_outputs",
                "--output-dir",
                str(output_dir),
                "--repo-id",
                "test/repo",
                "--path-in-repo",
                "paper/framework",
            ]
            with patch("sys.argv", argv), patch(
                "paper_assets.framework.upload_framework_outputs.HfApi",
                return_value=api,
            ):
                self.assertEqual(upload_main(), 0)

            kwargs = api.upload_folder.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], "test/repo")
            self.assertEqual(kwargs["path_in_repo"], "paper/framework")
            self.assertEqual(kwargs["allow_patterns"], UPLOAD_PATTERNS)
            self.assertNotIn("*.npz", kwargs["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
