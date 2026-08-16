#!/usr/bin/env python3
"""Validate Lucidum release artifacts and their package-index copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_NAME = "py-lucidum"
DIST_PREFIX = "py_lucidum"
REQUIRED_WHEEL_PATHS = (
    "py_lucidum/datasets/motor_premiums.parquet",
    "py_lucidum/example_workflows/01_external_gbm_artifacts_demo.py",
    "py_lucidum/example_workflows/01_external_glm_artifacts_demo.py",
    "py_lucidum/example_workflows/02_external_gbm_report_demo.py",
    "py_lucidum/example_workflows/02_external_glm_report_demo.py",
    "py_lucidum/example_workflows/03_external_gbm_summary_report_demo.py",
    "py_lucidum/example_workflows/03_external_glm_summary_report_demo.py",
    "py_lucidum/example_workflows/external_model_helpers.py",
    "py_lucidum/example_workflows/external_model_results.py",
    "py_lucidum/example_workflows/external_report_helpers.py",
    "py_lucidum/example_workflows/lucidum_install.py",
    "py_lucidum/specs/feature_spec.csv",
    "py_lucidum/specs/filter_spec.csv",
    "py_lucidum/specs/kpi_spec.csv",
    "py_lucidum/static/favicon.ico",
    "py_lucidum/static/index.html",
    "py_lucidum/tools/uk_map/static/geodata/areas_MappaR.geojson",
    "py_lucidum/tools/uk_map/static/geodata/sectors_MappaR.geojson",
)
REQUIRED_SDIST_PATHS = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "datasets/motor_premiums.parquet",
    "examples/01_external_gbm_artifacts_demo.py",
    "examples/01_external_glm_artifacts_demo.py",
    "examples/02_external_gbm_report_demo.py",
    "examples/02_external_glm_report_demo.py",
    "examples/03_external_gbm_summary_report_demo.py",
    "examples/03_external_glm_summary_report_demo.py",
    "examples/external_model_helpers.py",
    "examples/external_model_results.py",
    "examples/external_report_helpers.py",
    "examples/lucidum_install.py",
    "src/py_lucidum/static/index.html",
)


def release_files(dist_dir: Path, version: str) -> tuple[Path, Path]:
    wheel = dist_dir / f"{DIST_PREFIX}-{version}-py3-none-any.whl"
    sdist = dist_dir / f"{DIST_PREFIX}-{version}.tar.gz"
    expected = {wheel.name, sdist.name}
    actual = {
        path.name
        for path in dist_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    }
    if actual != expected:
        raise ValueError(
            f"expected release files {sorted(expected)}, found {sorted(actual)} in {dist_dir}"
        )
    return wheel, sdist


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifacts(dist_dir: Path, version: str) -> tuple[Path, Path]:
    wheel, sdist = release_files(dist_dir, version)

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = [name for name in REQUIRED_WHEEL_PATHS if name not in names]
        if missing:
            raise ValueError(f"wheel is missing required paths: {', '.join(missing)}")
        metadata_paths = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_paths) != 1:
            raise ValueError(f"wheel must contain one METADATA file, found {metadata_paths}")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if metadata.get("Name") != PROJECT_NAME:
            raise ValueError(f"wheel project name is {metadata.get('Name')!r}, expected {PROJECT_NAME!r}")
        if metadata.get("Version") != version:
            raise ValueError(f"wheel version is {metadata.get('Version')!r}, expected {version!r}")

    prefix = f"{DIST_PREFIX}-{version}"
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        missing = [name for name in REQUIRED_SDIST_PATHS if f"{prefix}/{name}" not in names]
        if missing:
            raise ValueError(f"sdist is missing required paths: {', '.join(missing)}")

    checksum_path = dist_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted((wheel, sdist))),
        encoding="utf-8",
    )
    return wheel, sdist


def index_digests(repository_url: str, version: str) -> dict[str, str]:
    url = f"{repository_url.rstrip('/')}/{PROJECT_NAME}/{version}/json"
    request = Request(url, headers={"User-Agent": "py-lucidum-release-verifier/1"})
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    return {
        str(item["filename"]): str(item.get("digests", {}).get("sha256", ""))
        for item in payload.get("urls", [])
    }


def verify_index(
    dist_dir: Path,
    version: str,
    repository_url: str,
    *,
    timeout: int,
    interval: int,
) -> None:
    wheel, sdist = release_files(dist_dir, version)
    expected = {path.name: sha256(path) for path in (wheel, sdist)}
    deadline = time.monotonic() + timeout
    last_error = "release was not returned by the index"

    while True:
        try:
            published = index_digests(repository_url, version)
            if published == expected:
                return
            last_error = f"expected hashes {expected}, found {published}"
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"could not verify {PROJECT_NAME} {version} at {repository_url}: {last_error}"
            )
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect local artifacts and write checksums")
    inspect_parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    inspect_parser.add_argument("--version", required=True)

    verify_parser = subparsers.add_parser("verify-index", help="compare local artifacts with an index")
    verify_parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    verify_parser.add_argument("--version", required=True)
    verify_parser.add_argument("--repository-url", required=True)
    verify_parser.add_argument("--timeout", type=int, default=180)
    verify_parser.add_argument("--interval", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        inspect_artifacts(args.dist_dir, args.version)
    else:
        verify_index(
            args.dist_dir,
            args.version,
            args.repository_url,
            timeout=args.timeout,
            interval=args.interval,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
