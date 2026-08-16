from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tomllib
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from py_lucidum.example_sync import EXAMPLE_SCRIPT_NAMES


RUN_PIPX_INSTALL_TESTS = os.environ.get("PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS") == "1"
PIPX_SPEC_ENV = "PY_LUCIDUM_PIPX_SPEC"
EXPECTED_VERSION_ENV = "PY_LUCIDUM_EXPECTED_VERSION"


class PipxInstallTests(unittest.TestCase):
    def test_gbm_extra_includes_lightgbm_arrow_runtime(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = pyproject["project"]["optional-dependencies"]["gbm"]

        self.assertTrue(
            any(requirement.startswith("cffi>=") for requirement in requirements),
            "The GBM training path imports cffi directly, so the gbm extra must install it.",
        )

    def test_wheel_force_include_does_not_duplicate_package_files(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        force_include = (
            pyproject.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("force-include", {})
        )

        duplicated_package_sources = [
            source
            for source in force_include
            if Path(source).parts[:2] == ("src", "py_lucidum")
        ]
        self.assertEqual(
            duplicated_package_sources,
            [],
            "Files under src/py_lucidum are already included by the wheel package mapping; "
            "force-include them only from outside the package tree.",
        )

    def test_wheel_force_includes_maintained_example_scripts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        expected = {
            f"examples/{name}": f"py_lucidum/example_workflows/{name}"
            for name in EXAMPLE_SCRIPT_NAMES
        }

        self.assertEqual(
            {source: target for source, target in force_include.items() if source.startswith("examples/")},
            expected,
        )

    @unittest.skipUnless(RUN_PIPX_INSTALL_TESTS, "set PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS=1 to run pipx install tests")
    def test_pipx_installed_lucidum_launches_project_csv(self) -> None:
        pipx = shutil.which("pipx")
        if pipx is None:
            self.skipTest("pipx is not installed")

        repo_root = Path(__file__).resolve().parents[1]
        install_python = os.environ.get("PY_LUCIDUM_PIPX_PYTHON", sys.executable)
        install_spec = os.environ.get(PIPX_SPEC_ENV, str(repo_root))
        expected_version = os.environ.get(EXPECTED_VERSION_ENV)

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pipx_home = root / "pipx-home"
            pipx_bin = root / "pipx-bin"
            pipx_man = root / "pipx-man"
            project = root / "project"
            project.mkdir()
            data_path = project / "dummy.csv"
            data_path.write_text("x,Actual\n1,10\n2,20\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "PIPX_HOME": str(pipx_home),
                    "PIPX_BIN_DIR": str(pipx_bin),
                    "PIPX_MAN_DIR": str(pipx_man),
                    "PIPX_DEFAULT_PYTHON": install_python,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                }
            )

            install = subprocess.run(
                [pipx, "install", "--force", "--python", install_python, install_spec],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                install.returncode,
                0,
                f"pipx install failed\nSTDOUT:\n{install.stdout}\nSTDERR:\n{install.stderr}",
            )

            lucidum = pipx_bin / ("lucidum.exe" if os.name == "nt" else "lucidum")
            self.assertTrue(lucidum.exists(), f"Expected pipx to expose {lucidum}")

            version = subprocess.run(
                [str(lucidum), "--version"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                version.returncode,
                0,
                f"lucidum --version failed\nSTDOUT:\n{version.stdout}\nSTDERR:\n{version.stderr}",
            )
            if expected_version:
                self.assertEqual(version.stdout.strip(), f"lucidum {expected_version}")

            synced_examples = root / "synced-examples"
            sync = subprocess.run(
                [str(lucidum), "--sync-examples", str(synced_examples)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                sync.returncode,
                0,
                f"lucidum --sync-examples failed\nSTDOUT:\n{sync.stdout}\nSTDERR:\n{sync.stderr}",
            )
            self.assertEqual(
                sorted(path.name for path in synced_examples.iterdir()),
                sorted(EXAMPLE_SCRIPT_NAMES),
            )
            self.assertTrue(all(synced_examples.joinpath(name).stat().st_size > 0 for name in EXAMPLE_SCRIPT_NAMES))

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])

            process = subprocess.Popen(
                [str(lucidum), str(data_path.name), "--no-token", "--port", str(port)],
                cwd=project,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                self._wait_for_health(port, process)
            finally:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=10)

            self.assertIn(f"lucidum serving {data_path.resolve()}", stdout)
            self.assertNotIn("Traceback", stdout + stderr)

    def _wait_for_health(self, port: int, process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + 30
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                self.fail(
                    f"lucidum exited before health check succeeded with code {process.returncode}\n"
                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
                    if response.status == 200 and response.read() == b'{"status":"ok"}':
                        return
            except BaseException as error:  # pragma: no cover - only reported on timeout.
                last_error = error
            time.sleep(0.2)
        self.fail(f"lucidum health check did not succeed within 30 seconds: {last_error!r}")


if __name__ == "__main__":
    unittest.main()
