from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request

from py_lucidum.core import Dataset


@dataclass(frozen=True)
class AppContext:
    dataset: Dataset
    check_token: Callable[[Request], None]
