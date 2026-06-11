from __future__ import annotations

import csv
from pathlib import Path


KPI_SPEC_COLUMNS = ["group", "name", "actual", "denominator", "decimals", "format"]
KPI_FORMATS = {"number", "currency", "percent"}


def resolve_kpis_path(kpis_path: str | Path | None, use_kpis: bool = True) -> Path | None:
    if not use_kpis:
        return None
    if kpis_path:
        return Path(kpis_path).expanduser().resolve()
    root_spec = (Path.cwd() / "kpi_spec.csv").resolve()
    if root_spec.exists():
        return root_spec
    return (Path.cwd() / "specs" / "kpi_spec.csv").resolve()


def load_kpis(kpis_path: str | Path | None, use_kpis: bool = True, missing_ok: bool = False) -> list[dict[str, str | int]]:
    path = resolve_kpis_path(kpis_path, use_kpis=use_kpis)
    if path is None:
        return []
    if not path.exists():
        if kpis_path and not missing_ok:
            raise FileNotFoundError(f"KPI specification file does not exist: {path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != KPI_SPEC_COLUMNS:
            raise ValueError("kpi_spec.csv must have exactly these columns: group,name,actual,denominator,decimals,format")
        kpis: list[dict[str, str | int]] = []
        for row_number, row in enumerate(reader, start=2):
            group = str(row.get("group") or "").strip()
            name = str(row.get("name") or "").strip()
            actual = str(row.get("actual") or "").strip()
            if not (group and name and actual):
                continue
            denominator = normalise_kpi_denominator(row.get("denominator"))
            decimals = normalise_kpi_decimals(row.get("decimals"), row_number)
            value_format = normalise_kpi_format(row.get("format"), row_number)
            kpis.append({
                "group": group,
                "name": name,
                "actual": actual,
                "denominator": denominator,
                "decimals": decimals,
                "format": value_format,
            })
        return kpis


def normalise_kpi_denominator(value: object) -> str:
    denominator = str(value or "").strip()
    if denominator.lower() in {"", "n", "average row value", "__none__"}:
        return "__none__"
    return denominator


def normalise_kpi_decimals(value: object, row_number: int) -> int:
    raw = str(value or "").strip()
    try:
        decimals = int(raw)
    except ValueError as exc:
        raise ValueError(f"kpi_spec.csv row {row_number} has invalid decimals: {raw!r}") from exc
    if decimals < 0 or decimals > 12:
        raise ValueError(f"kpi_spec.csv row {row_number} decimals must be between 0 and 12")
    return decimals


def normalise_kpi_format(value: object, row_number: int) -> str:
    value_format = str(value or "").strip().lower()
    if value_format not in KPI_FORMATS:
        accepted = ", ".join(sorted(KPI_FORMATS))
        raise ValueError(f"kpi_spec.csv row {row_number} has invalid format: {value_format!r}. Use one of: {accepted}")
    return value_format
