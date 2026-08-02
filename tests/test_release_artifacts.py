from __future__ import annotations

import hashlib
import io
import json
import tarfile
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import release_artifacts


class ReleaseArtifactTests(unittest.TestCase):
    VERSION = "1.2.3"

    def create_artifacts(self, dist_dir: Path) -> tuple[Path, Path]:
        wheel = dist_dir / f"py_lucidum-{self.VERSION}-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name in release_artifacts.REQUIRED_WHEEL_PATHS:
                archive.writestr(name, b"fixture")
            archive.writestr(
                f"py_lucidum-{self.VERSION}.dist-info/METADATA",
                f"Name: py-lucidum\nVersion: {self.VERSION}\n",
            )

        sdist = dist_dir / f"py_lucidum-{self.VERSION}.tar.gz"
        prefix = f"py_lucidum-{self.VERSION}"
        with tarfile.open(sdist, "w:gz") as archive:
            for relative in release_artifacts.REQUIRED_SDIST_PATHS:
                data = b"fixture"
                info = tarfile.TarInfo(f"{prefix}/{relative}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return wheel, sdist

    def test_inspect_artifacts_checks_contents_and_writes_hashes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir)
            wheel, sdist = self.create_artifacts(dist_dir)

            release_artifacts.inspect_artifacts(dist_dir, self.VERSION)

            lines = (dist_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            expected = {
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
                for path in (wheel, sdist)
            }
            self.assertEqual(set(lines), expected)

    def test_inspect_artifacts_rejects_missing_wheel_asset(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir)
            wheel, _ = self.create_artifacts(dist_dir)
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"py_lucidum-{self.VERSION}.dist-info/METADATA",
                    f"Name: py-lucidum\nVersion: {self.VERSION}\n",
                )

            with self.assertRaisesRegex(ValueError, "wheel is missing required paths"):
                release_artifacts.inspect_artifacts(dist_dir, self.VERSION)

    def test_verify_index_accepts_matching_published_hashes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir)
            wheel, sdist = self.create_artifacts(dist_dir)
            payload = {
                "urls": [
                    {
                        "filename": path.name,
                        "digests": {"sha256": release_artifacts.sha256(path)},
                    }
                    for path in (wheel, sdist)
                ]
            }
            response = io.BytesIO(json.dumps(payload).encode("utf-8"))
            response.__enter__ = lambda value: value  # type: ignore[attr-defined]
            response.__exit__ = lambda *_: None  # type: ignore[attr-defined]

            with patch.object(release_artifacts, "urlopen", return_value=response):
                release_artifacts.verify_index(
                    dist_dir,
                    self.VERSION,
                    "https://packages.example.invalid/pypi",
                    timeout=0,
                    interval=0,
                )

    def test_release_workflow_keeps_checksums_out_of_index_uploads(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")
        distribution_upload = workflow.split("- name: Store distributions", 1)[1].split(
            "- name: Store checksums", 1
        )[0]
        checksum_upload = workflow.split("- name: Store checksums", 1)[1].split(
            "draft-release:", 1
        )[0]

        self.assertIn("dist/*.whl", distribution_upload)
        self.assertIn("dist/*.tar.gz", distribution_upload)
        self.assertNotIn("dist/SHA256SUMS", distribution_upload)
        self.assertIn("name: python-package-checksums", checksum_upload)
        self.assertIn("path: dist/SHA256SUMS", checksum_upload)


if __name__ == "__main__":
    unittest.main()
