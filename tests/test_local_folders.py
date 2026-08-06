from __future__ import annotations

import asyncio
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from py_lucidum.app import create_app
from py_lucidum.app.local_folders import (
    LocalFolderOpenError,
    LocalFolderPathError,
    client_is_loopback,
    confined_existing_directory,
    folder_open_command,
    local_folder_opening_available,
    open_local_folder,
)


def asgi_request(
    app: Any,
    method: str,
    path: str,
    *,
    client_host: str | None = "127.0.0.1",
    token: str = "",
) -> tuple[int, dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    body = b"{}" if method == "POST" else b""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    headers = [(b"content-type", b"application/json")]
    if token:
        headers.append((b"x-lucidum-token", token.encode("utf-8")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": (client_host, 12345) if client_host is not None else None,
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    status = next(message for message in messages if message["type"] == "http.response.start")["status"]
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body or b"{}")


class LocalFolderHelperTests(unittest.TestCase):
    def test_loopback_detection_supports_ipv4_ipv6_mapped_and_localhost(self) -> None:
        for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"):
            with self.subTest(host=host):
                self.assertTrue(client_is_loopback(host))
        for host in (None, "", "192.168.1.20", "example.test"):
            with self.subTest(host=host):
                self.assertFalse(client_is_loopback(host))

    def test_platform_capability_requires_a_supported_desktop_opener(self) -> None:
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(folder_open_command(platform="darwin"), "/usr/bin/open")
            self.assertTrue(local_folder_opening_available(platform="darwin"))
        self.assertEqual(
            folder_open_command(
                platform="linux",
                environ={"DISPLAY": ":99"},
                which=lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None,
            ),
            "/usr/bin/xdg-open",
        )
        self.assertFalse(
            local_folder_opening_available(
                platform="linux",
                environ={},
                which=lambda _name: "/usr/bin/xdg-open",
            )
        )
        self.assertFalse(
            local_folder_opening_available(
                platform="linux",
                environ={"WAYLAND_DISPLAY": "wayland-0"},
                which=lambda _name: None,
            )
        )
        self.assertFalse(
            local_folder_opening_available(
                platform="linux",
                environ={"XDG_CURRENT_DESKTOP": "GNOME"},
                which=lambda _name: "/usr/bin/xdg-open",
            )
        )
        self.assertFalse(local_folder_opening_available(platform="freebsd"))

    def test_open_local_folder_dispatches_without_a_shell(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "model with spaces"
            directory.mkdir()
            with (
                patch.object(Path, "is_file", return_value=True),
                patch("py_lucidum.app.local_folders.subprocess.Popen") as popen,
            ):
                open_local_folder(directory, platform="darwin")
            popen.assert_called_once_with(
                ["/usr/bin/open", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            with patch("py_lucidum.app.local_folders.os.startfile", create=True) as startfile:
                open_local_folder(directory, platform="win32")
            startfile.assert_called_once_with(os.fspath(directory), "explore")

            with patch("py_lucidum.app.local_folders.subprocess.Popen") as popen:
                open_local_folder(
                    directory,
                    platform="linux",
                    environ={"DISPLAY": ":99"},
                    which=lambda _name: "/usr/bin/xdg-open",
                )
            self.assertEqual(popen.call_args.args[0], ["/usr/bin/xdg-open", str(directory)])
            self.assertNotIn("shell", popen.call_args.kwargs)

    def test_open_local_folder_reports_missing_openers_paths_and_launch_failures(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "model"
            directory.mkdir()
            with self.assertRaisesRegex(LocalFolderOpenError, "unavailable"):
                open_local_folder(directory, platform="linux", environ={}, which=lambda _name: None)
            with (
                patch.object(Path, "is_file", return_value=True),
                patch("py_lucidum.app.local_folders.subprocess.Popen", side_effect=OSError("failed")),
                self.assertRaisesRegex(LocalFolderOpenError, "Could not open"),
            ):
                open_local_folder(directory, platform="darwin")
            with self.assertRaises(LocalFolderPathError):
                open_local_folder(Path(temp_dir) / "missing", platform="darwin")

    def test_confined_directory_rejects_workspace_escape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "workspace"
            inside = root / "model"
            outside = Path(temp_dir) / "outside"
            inside.mkdir(parents=True)
            outside.mkdir()
            self.assertEqual(confined_existing_directory(root, inside), inside.resolve())
            with self.assertRaises(LocalFolderPathError):
                confined_existing_directory(root, outside)


class ModelFolderRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_path = Path(self.temp_dir.name) / "data.csv"
        self.data_path.write_text("x,y\n1,2\n", encoding="utf-8")

    def make_app(self, kind: str, *, token: str = "") -> tuple[Any, Any, str]:
        app = create_app(
            self.data_path,
            token=token,
            tools=[kind, "line_bar"],
            use_saved_filters=False,
            use_kpis=False,
            use_features=False,
        )
        store = getattr(app.state, f"{kind}_store")
        model_id = f"{kind}-model"
        model_dir = store.create_model_dir(model_id)
        store.write_json(model_dir / "manifest.json", {"model_id": model_id, "label": model_id})
        return app, store, model_id

    def test_schema_capability_is_request_specific(self) -> None:
        app, _store, _model_id = self.make_app("glm")
        app.state.local_folder_opener_available = lambda: True
        local_status, local_payload = asgi_request(app, "GET", "/api/schema")
        remote_status, remote_payload = asgi_request(
            app,
            "GET",
            "/api/schema",
            client_host="192.168.1.50",
        )
        unknown_status, unknown_payload = asgi_request(
            app,
            "GET",
            "/api/schema",
            client_host=None,
        )
        app.state.local_folder_opener_available = lambda: False
        unsupported_status, unsupported_payload = asgi_request(app, "GET", "/api/schema")
        self.assertEqual((local_status, remote_status, unknown_status, unsupported_status), (200, 200, 200, 200))
        self.assertTrue(local_payload["capabilities"]["open_model_folders"])
        self.assertFalse(remote_payload["capabilities"]["open_model_folders"])
        self.assertFalse(unknown_payload["capabilities"]["open_model_folders"])
        self.assertFalse(unsupported_payload["capabilities"]["open_model_folders"])

    def test_glm_and_gbm_routes_open_only_the_valid_local_model_folder(self) -> None:
        for kind in ("glm", "gbm"):
            with self.subTest(kind=kind):
                app, store, model_id = self.make_app(kind)
                opened: list[Path] = []
                app.state.local_folder_opener_available = lambda: True
                app.state.local_folder_opener = opened.append
                path = f"/api/{kind}/models/{model_id}/open-folder"
                status, payload = asgi_request(app, "POST", path)
                self.assertEqual(status, 200)
                self.assertEqual(payload, {"opened": True, "model_id": model_id})
                self.assertEqual(opened, [store.model_dir(model_id).resolve()])
                self.assertNotIn(str(store.root), json.dumps(payload))

                remote_status, _ = asgi_request(app, "POST", path, client_host="192.168.1.50")
                missing_status, _ = asgi_request(app, "POST", f"/api/{kind}/models/missing/open-folder")
                app.state.local_folder_opener_available = lambda: False
                unsupported_status, _ = asgi_request(app, "POST", path)
                self.assertEqual(remote_status, 403)
                self.assertEqual(missing_status, 404)
                self.assertEqual(unsupported_status, 503)
                self.assertEqual(len(opened), 1)

    def test_routes_enforce_token_before_opening(self) -> None:
        for kind in ("glm", "gbm"):
            with self.subTest(kind=kind):
                app, _store, model_id = self.make_app(kind, token="secret")
                opened: list[Path] = []
                app.state.local_folder_opener_available = lambda: True
                app.state.local_folder_opener = opened.append
                path = f"/api/{kind}/models/{model_id}/open-folder"
                missing_status, _ = asgi_request(app, "POST", path)
                accepted_status, _ = asgi_request(app, "POST", path, token="secret")
                self.assertEqual(missing_status, 401)
                self.assertEqual(accepted_status, 200)
                self.assertEqual(len(opened), 1)

    def test_route_rejects_symlink_escape(self) -> None:
        for kind in ("glm", "gbm"):
            with self.subTest(kind=kind):
                app, store, _model_id = self.make_app(kind)
                outside = Path(self.temp_dir.name) / f"outside-{kind}-model"
                outside.mkdir()
                store.write_json(outside / "manifest.json", {"model_id": "escape", "label": "escape"})
                link = store.model_dir("escape")
                try:
                    link.symlink_to(outside, target_is_directory=True)
                except OSError as exc:
                    self.skipTest(f"directory symlinks unavailable: {exc}")
                opened: list[Path] = []
                app.state.local_folder_opener_available = lambda: True
                app.state.local_folder_opener = opened.append
                status, _payload = asgi_request(app, "POST", f"/api/{kind}/models/escape/open-folder")
                self.assertEqual(status, 404)
                self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
