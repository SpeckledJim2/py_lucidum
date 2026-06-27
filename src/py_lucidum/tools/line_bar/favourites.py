from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from py_lucidum.core import Dataset, dataset_slug, dataset_workspace_metadata, is_numeric_kind


FAVOURITES_VERSION = 1
FAVOURITES_FILENAME = "favourites.json"
FAVOURITE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


class LineBarFavouriteError(ValueError):
    pass


class LineBarFavouriteStore:
    def __init__(
        self,
        dataset_path: str | Path,
        dataset: Dataset | None = None,
        favourites_path: str | Path | None = None,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.dataset = dataset
        self.favourites_path = Path(favourites_path).expanduser().resolve() if favourites_path else None

    @property
    def root(self) -> Path:
        if self.favourites_path is not None:
            return self.favourites_path.parent
        return self.dataset_path.parent / ".lucidum" / "datasets" / dataset_slug(self.dataset_path) / "line_bar"

    @property
    def path(self) -> Path:
        if self.favourites_path is not None:
            return self.favourites_path
        return self.root / FAVOURITES_FILENAME

    def read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": FAVOURITES_VERSION, "favourites": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LineBarFavouriteError(f"Favourites file is not valid JSON: {self.path}") from exc
        if isinstance(payload, list):
            favourites = payload
        elif isinstance(payload, dict):
            favourites = payload.get("favourites", [])
        else:
            favourites = []
        return {
            "version": FAVOURITES_VERSION,
            "favourites": [item for item in favourites if isinstance(item, dict)],
        }

    def write_payload(self, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)

    def list_favourites(
        self,
        *,
        saved_filters: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.favourite_payload(item, saved_filters=saved_filters, kpis=kpis)
            for item in self.read_payload()["favourites"]
        ]

    def create_favourite(
        self,
        name: Any,
        view: Any,
        *,
        saved_filters: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cleaned_name = self.clean_name(name)
        if not isinstance(view, dict):
            raise LineBarFavouriteError("Favourite view must be an object")
        payload = self.read_payload()
        favourites = payload["favourites"]
        self.ensure_unique_name(cleaned_name, favourites)
        validation = self.validate_view(view, saved_filters=saved_filters, kpis=kpis)
        if validation["errors"]:
            raise LineBarFavouriteError("; ".join(validation["errors"]))
        now = timestamp()
        item = {
            "id": uuid4().hex,
            "name": cleaned_name,
            "created_at": now,
            "updated_at": now,
            "dataset": self.dataset_metadata(),
            "view": json_safe(view),
        }
        favourites.append(item)
        self.write_payload(payload)
        return self.favourite_payload(item, saved_filters=saved_filters, kpis=kpis)

    def rename_favourite(
        self,
        favourite_id: str,
        name: Any,
        *,
        saved_filters: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cleaned_id = self.validate_id(favourite_id)
        cleaned_name = self.clean_name(name)
        payload = self.read_payload()
        favourites = payload["favourites"]
        item = self.find_favourite(favourites, cleaned_id)
        self.ensure_unique_name(cleaned_name, favourites, ignore_id=cleaned_id)
        item["name"] = cleaned_name
        item["updated_at"] = timestamp()
        self.write_payload(payload)
        return self.favourite_payload(item, saved_filters=saved_filters, kpis=kpis)

    def delete_favourite(self, favourite_id: str) -> dict[str, Any]:
        cleaned_id = self.validate_id(favourite_id)
        payload = self.read_payload()
        favourites = payload["favourites"]
        item = self.find_favourite(favourites, cleaned_id)
        payload["favourites"] = [fav for fav in favourites if fav.get("id") != cleaned_id]
        self.write_payload(payload)
        return dict(item)

    def reorder_favourites(
        self,
        ordered_ids: Any,
        *,
        saved_filters: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(ordered_ids, list):
            raise LineBarFavouriteError("Favourite order must be a list of ids")
        order = [self.validate_id(item) for item in ordered_ids]
        if len(order) != len(set(order)):
            raise LineBarFavouriteError("Favourite order includes duplicate ids")
        payload = self.read_payload()
        favourites = payload["favourites"]
        by_id = {str(item.get("id") or ""): item for item in favourites}
        missing = [favourite_id for favourite_id in order if favourite_id not in by_id]
        if missing:
            raise LineBarFavouriteError(f"Choose valid favourite ids: {', '.join(missing)}")
        payload["favourites"] = [by_id[favourite_id] for favourite_id in order] + [
            item for item in favourites if str(item.get("id") or "") not in set(order)
        ]
        self.write_payload(payload)
        return self.list_favourites(saved_filters=saved_filters, kpis=kpis)

    def favourite_payload(
        self,
        item: dict[str, Any],
        *,
        saved_filters: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = dict(item)
        view = payload.get("view") if isinstance(payload.get("view"), dict) else {}
        payload["validation"] = self.validate_view(view, saved_filters=saved_filters, kpis=kpis)
        return payload

    def validate_view(
        self,
        view: dict[str, Any],
        *,
        saved_filters: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        source = self.validate_source(view.get("source"), errors, label="data source")
        x_source = self.validate_source(view.get("xSource") or source, errors, label="x-axis source")
        self.validate_column(x_source, view.get("x"), errors, label="x-axis feature")
        actual = view.get("actual") if isinstance(view.get("actual"), dict) else {}
        actual_source = self.validate_source(actual.get("sourceId") or source, errors, label="Actual source")
        self.validate_column(actual_source, actual.get("value"), errors, label="Actual", numeric=True)
        denominator = str(view.get("denominator") or "__none__").strip() or "__none__"
        if denominator != "__none__":
            self.validate_column(source, denominator, errors, label="Weight", numeric=True)
        expected = view.get("expectedSelections")
        if isinstance(expected, list):
            for index, selection in enumerate(expected[:2], start=1):
                if not isinstance(selection, dict):
                    continue
                expected_source = self.validate_source(selection.get("sourceId") or source, errors, label=f"Expected {index} source")
                self.validate_column(expected_source, selection.get("value"), errors, label=f"Expected {index}", numeric=True)
        filter_sql = str(view.get("filter") or "").strip()
        if filter_sql and source:
            try:
                active_dataset(self.dataset_path, self.dataset).normalise_filter(filter_sql, source_id=source)
            except (ValueError, duckdb.Error) as exc:
                errors.append(f"Favourite filter is invalid: {exc}")
        saved_rows = view.get("savedFilterRows")
        if isinstance(saved_rows, list) and saved_rows:
            available = {saved_filter_key(row) for row in saved_filters or []}
            missing = [
                row for row in saved_rows
                if isinstance(row, dict) and saved_filter_key(row) not in available
            ]
            if missing:
                warnings.append(f"{len(missing)} saved FILTER selection{'s' if len(missing) != 1 else ''} no longer exist.")
        kpi = view.get("kpi") if isinstance(view.get("kpi"), dict) else {}
        if kpi:
            available_kpis = {kpi_key(row) for row in kpis or []}
            if kpi_key(kpi) not in available_kpis:
                warnings.append("Saved KPI selection no longer exists; Actual and Weight will still be restored.")
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_source(self, raw_source: Any, errors: list[str], *, label: str) -> str:
        raw = str(raw_source or "dataset").strip() or "dataset"
        try:
            return active_dataset(self.dataset_path, self.dataset).normalise_source(raw)
        except ValueError:
            errors.append(f"Favourite uses a missing {label}: {raw}")
            return raw

    def validate_column(
        self,
        source_id: str,
        name: Any,
        errors: list[str],
        *,
        label: str,
        numeric: bool = False,
    ) -> None:
        column_name = str(name or "").strip()
        if not column_name:
            errors.append(f"Favourite is missing {label}.")
            return
        try:
            columns = {
                column.name: column
                for column in active_dataset(self.dataset_path, self.dataset).schema_columns_for_source(source_id)
            }
        except ValueError:
            return
        column = columns.get(column_name)
        if column is None:
            errors.append(f"Favourite uses missing {label} column: {column_name}")
            return
        if numeric and not is_numeric_kind(column.kind):
            errors.append(f"Favourite {label} column is not numeric: {column_name}")

    def dataset_metadata(self) -> dict[str, Any]:
        metadata = dataset_workspace_metadata(self.dataset_path, self.dataset)
        return {
            "slug": metadata["slug"],
            "signature": metadata["signature"],
            "path": metadata["path"],
            "schema_fingerprint": metadata["schema_fingerprint"],
        }

    def validate_id(self, favourite_id: Any) -> str:
        text = str(favourite_id or "").strip()
        if not text or not FAVOURITE_ID_RE.fullmatch(text):
            raise LineBarFavouriteError("Choose a valid favourite id")
        return text

    def clean_name(self, name: Any) -> str:
        text = str(name or "").strip()
        if not text:
            raise LineBarFavouriteError("Choose a favourite name")
        if len(text) > 120:
            raise LineBarFavouriteError("Favourite names must be 120 characters or fewer")
        return text

    def ensure_unique_name(self, name: str, favourites: list[dict[str, Any]], *, ignore_id: str = "") -> None:
        target = name.casefold()
        for item in favourites:
            if ignore_id and item.get("id") == ignore_id:
                continue
            if str(item.get("name") or "").strip().casefold() == target:
                raise LineBarFavouriteError(f"Favourite already exists: {name}")

    def find_favourite(self, favourites: list[dict[str, Any]], favourite_id: str) -> dict[str, Any]:
        for item in favourites:
            if item.get("id") == favourite_id:
                return item
        raise LineBarFavouriteError("Choose a valid favourite")


def active_dataset(dataset_path: Path, dataset: Dataset | None) -> Dataset:
    return dataset if dataset is not None else Dataset(dataset_path)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def saved_filter_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("theme") or "General"),
        str(row.get("name") or ""),
        str(row.get("expression") or "").strip(),
    )


def kpi_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("group") or "General"),
        str(row.get("name") or ""),
        str(row.get("actual") or ""),
        str(row.get("denominator") or "__none__"),
    )


def json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


__all__ = [
    "LineBarFavouriteError",
    "LineBarFavouriteStore",
]
