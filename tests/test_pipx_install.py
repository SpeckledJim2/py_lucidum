from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


RUN_PIPX_INSTALL_TESTS = os.environ.get("PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS") == "1"


class PipxInstallTests(unittest.TestCase):
    @unittest.skipUnless(RUN_PIPX_INSTALL_TESTS, "set PY_LUCIDUM_RUN_PIPX_INSTALL_TESTS=1 to run pipx install tests")
    def test_pipx_installed_lucidum_launches_project_csv(self) -> None:
        pipx = shutil.which("pipx")
        if pipx is None:
            self.skipTest("pipx is not installed")

        repo_root = Path(__file__).resolve().parents[1]
        install_python = os.environ.get("PY_LUCIDUM_PIPX_PYTHON", sys.executable)

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
                [pipx, "install", "--force", "--python", install_python, str(repo_root)],
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

            self.assertIn(f"py_lucidum serving {data_path.resolve()}", stdout)
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
