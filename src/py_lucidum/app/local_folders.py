from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from fastapi import Request


class LocalFolderOpenError(RuntimeError):
    """Raised when a desktop folder cannot be opened."""


class LocalFolderPathError(ValueError):
    """Raised when a requested folder is missing or escapes its workspace."""


def client_is_loopback(host: str | None) -> bool:
    text = str(host or "").strip().strip("[]")
    if not text:
        return False
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return text.casefold() == "localhost"
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def request_is_loopback(request: Request) -> bool:
    client = request.client
    return client_is_loopback(client.host if client else None)


def _linux_desktop_available(environ: dict[str, str] | None = None) -> bool:
    environment = os.environ if environ is None else environ
    return any(
        str(environment.get(name) or "").strip()
        for name in ("DISPLAY", "WAYLAND_DISPLAY")
    )


def folder_open_command(
    *,
    platform: str | None = None,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    selected_platform = sys.platform if platform is None else platform
    if selected_platform == "darwin":
        command = Path("/usr/bin/open")
        return str(command) if command.is_file() else None
    if selected_platform.startswith("linux") and _linux_desktop_available(environ):
        return which("xdg-open")
    return None


def local_folder_opening_available(
    *,
    platform: str | None = None,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> bool:
    selected_platform = sys.platform if platform is None else platform
    if selected_platform == "win32":
        return callable(getattr(os, "startfile", None))
    return folder_open_command(platform=selected_platform, environ=environ, which=which) is not None


def model_folder_opening_available(
    request: Request,
    dataset_path: str | Path,
    *,
    opener_available: Callable[[], bool] = local_folder_opening_available,
) -> bool:
    return request_is_loopback(request) and Path(dataset_path).is_file() and opener_available()


def confined_existing_directory(workspace_root: str | Path, directory: str | Path) -> Path:
    try:
        resolved_root = Path(workspace_root).resolve(strict=True)
        resolved_directory = Path(directory).resolve(strict=True)
    except OSError as exc:
        raise LocalFolderPathError("Choose an existing model folder") from exc
    if not resolved_directory.is_dir() or not resolved_directory.is_relative_to(resolved_root):
        raise LocalFolderPathError("Choose an existing model folder")
    return resolved_directory


def open_local_folder(
    directory: str | Path,
    *,
    platform: str | None = None,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> None:
    path = Path(directory)
    if not path.is_dir():
        raise LocalFolderPathError("Choose an existing model folder")
    selected_platform = sys.platform if platform is None else platform
    try:
        if selected_platform == "win32":
            startfile = getattr(os, "startfile", None)
            if not callable(startfile):
                raise LocalFolderOpenError("Opening folders is unavailable on this system")
            startfile(os.fspath(path), "explore")
            return
        command = folder_open_command(platform=selected_platform, environ=environ, which=which)
        if not command:
            raise LocalFolderOpenError("Opening folders is unavailable on this system")
        subprocess.Popen(
            [command, os.fspath(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except LocalFolderOpenError:
        raise
    except OSError as exc:
        raise LocalFolderOpenError("Could not open the model folder") from exc
