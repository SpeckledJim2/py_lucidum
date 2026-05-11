from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def infer_kind(duckdb_type: str) -> str:
    t = duckdb_type.upper()
    if any(part in t for part in ("INT", "DOUBLE", "FLOAT", "REAL", "DECIMAL", "HUGEINT", "UBIGINT")):
        return "numeric"
    if "TIMESTAMP" in t:
        return "datetime"
    if "DATE" in t or "TIME" in t:
        return "date"
    if "BOOL" in t:
        return "categorical"
    return "categorical"


def json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 9_007_199_254_740_992:
        return int(number)
    return number


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    duckdb_type: str
    kind: str
    band_suggestion: float | int | None = None


class Dataset:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self.path}")
        self.con = duckdb.connect(database=":memory:")
        self._schema: list[ColumnInfo] | None = None
        self._row_count: int | None = None
        self._band_suggestions: dict[str, float | int | None] | None = None

    def relation_sql(self) -> str:
        path = sql_literal(str(self.path))
        suffix = self.path.suffix.lower()
        if suffix == ".parquet":
            return f"read_parquet({path})"
        if suffix == ".csv":
            return f"read_csv_auto({path}, header=true, ignore_errors=true)"
        raise ValueError("Only .csv and .parquet files are supported in this prototype")

    def reload(self) -> None:
        self._schema = None
        self._row_count = None
        self._band_suggestions = None

    def schema(self) -> dict[str, Any]:
        if self._schema is None:
            rows = self.con.execute(f"DESCRIBE SELECT * FROM {self.relation_sql()}").fetchall()
            base_schema = [
                ColumnInfo(name=str(row[0]), duckdb_type=str(row[1]), kind=infer_kind(str(row[1])))
                for row in rows
            ]
            suggestions = self.band_suggestions(base_schema)
            self._schema = [
                ColumnInfo(
                    name=col.name,
                    duckdb_type=col.duckdb_type,
                    kind=col.kind,
                    band_suggestion=suggestions.get(col.name),
                )
                for col in base_schema
            ]
        if self._row_count is None:
            self._row_count = int(self.con.execute(f"SELECT COUNT(*) FROM {self.relation_sql()}").fetchone()[0])
        return {
            "path": str(self.path),
            "row_count": self._row_count,
            "columns": [
                {
                    "name": c.name,
                    "duckdb_type": c.duckdb_type,
                    "kind": c.kind,
                    "band_suggestion": c.band_suggestion,
                }
                for c in self._schema
            ],
        }

    def band_suggestions(self, schema: list[ColumnInfo]) -> dict[str, float | int | None]:
        if self._band_suggestions is not None:
            return self._band_suggestions
        numeric = [col for col in schema if col.kind == "numeric"]
        if not numeric:
            self._band_suggestions = {}
            return self._band_suggestions
        select_sql = ",\n    ".join(
            f"STDDEV_SAMP(TRY_CAST({quote_ident(col.name)} AS DOUBLE)) AS {quote_ident(col.name)}"
            for col in numeric
        )
        sql = f"""
WITH sample AS (
  SELECT * FROM {self.relation_sql()} LIMIT 10000
)
SELECT
    {select_sql}
FROM sample
"""
        row = self.con.execute(sql).fetchone()
        names = [description[0] for description in self.con.description]
        self._band_suggestions = {
            name: suggested_band_width(value)
            for name, value in zip(names, row)
        }
        return self._band_suggestions

    def column_map(self) -> dict[str, ColumnInfo]:
        return {c.name: c for c in self._schema_columns()}

    def _schema_columns(self) -> list[ColumnInfo]:
        self.schema()
        assert self._schema is not None
        return self._schema

    def chart(self, request: dict[str, Any]) -> dict[str, Any]:
        columns = self.column_map()
        x_col = str(request.get("x") or "")
        if x_col not in columns:
            raise ValueError("Choose a valid x-axis feature")

        responses = normalise_responses(request.get("responses"), columns)
        x_info = columns[x_col]
        x_sql = build_x_sql(
            x_col=x_col,
            kind=x_info.kind,
            band_width=request.get("bandWidth"),
            date_bucket=request.get("dateBucket"),
        )
        sigma_multiplier = float(request.get("sigma") or 0)
        include_sigma = sigma_multiplier > 0 and len(responses) >= 2
        sql = build_chart_sql(self.relation_sql(), x_sql, responses, include_sigma)
        raw_rows = [dict(zip([d[0] for d in self.con.description], row)) for row in self.con.execute(sql).fetchall()]

        grouped_rows = apply_low_weight_grouping(
            rows=raw_rows,
            responses=responses,
            x_kind=x_info.kind,
            threshold=str(request.get("lowGroup") or "0"),
        )
        sorted_rows = sort_rows(grouped_rows, x_info.kind, str(request.get("sort") or "original"))
        max_groups = int(request.get("maxGroups") or 2000)
        if len(sorted_rows) > max_groups:
            sorted_rows = sorted_rows[:max_groups]

        transform = str(request.get("transform") or "none")
        warnings: list[str] = []
        display_rows = apply_transform(sorted_rows, responses, transform, sigma_multiplier, warnings)

        return {
            "x": x_col,
            "x_kind": x_info.kind,
            "responses": [{"label": r["label"], "numerator": r["numerator"], "denominator": r["denominator"]} for r in responses],
            "rows": display_rows,
            "warnings": warnings,
        }


def normalise_responses(raw: Any, columns: dict[str, ColumnInfo]) -> list[dict[str, str | None]]:
    responses: list[dict[str, str | None]] = []
    if not isinstance(raw, list):
        return responses
    for item in raw:
        if not isinstance(item, dict):
            continue
        numerator = item.get("numerator")
        if not numerator or numerator not in columns:
            continue
        denominator = item.get("denominator") or None
        if denominator == "__none__":
            denominator = None
        if denominator not in (None, "__count__") and denominator not in columns:
            denominator = None
        label = item.get("label") or str(numerator)
        responses.append({"label": str(label), "numerator": str(numerator), "denominator": denominator})
    return responses[:2]


def build_x_sql(x_col: str, kind: str, band_width: Any, date_bucket: Any) -> dict[str, str]:
    col = quote_ident(x_col)
    if kind == "numeric":
        raw = f"TRY_CAST({col} AS DOUBLE)"
        width = parse_positive_float(band_width)
        if width:
            key = f"FLOOR({raw} / {width}) * {width}"
        else:
            key = raw
        return {
            "key": key,
            "label": f"CASE WHEN {key} IS NULL THEN '(missing)' ELSE CAST({key} AS VARCHAR) END",
            "sort": key,
        }
    if kind in ("date", "datetime"):
        bucket = str(date_bucket or "none").lower()
        if bucket in {"hour", "day", "week", "month", "year"}:
            key = f"DATE_TRUNC('{bucket}', {col})"
        else:
            key = col
        return {
            "key": key,
            "label": f"CASE WHEN {key} IS NULL THEN '(missing)' ELSE CAST({key} AS VARCHAR) END",
            "sort": key,
        }
    key = f"COALESCE(CAST({col} AS VARCHAR), '(missing)')"
    return {"key": key, "label": key, "sort": key}


def parse_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or not math.isfinite(parsed):
        return None
    return parsed


def nice_band_steps() -> list[float]:
    steps: list[float] = []
    for exponent in range(-8, 13):
        multiplier = 10 ** exponent
        steps.extend([1 * multiplier, 2 * multiplier, 5 * multiplier])
    return sorted(steps)


def suggested_band_width(stddev: Any) -> float | int | None:
    value = parse_positive_float(stddev)
    if not value:
        return None
    steps = nice_band_steps()
    lower_index = 0
    for index, step in enumerate(steps):
        if step <= value:
            lower_index = index
        else:
            break
    suggestion_index = max(0, lower_index - 1)
    return json_number(steps[suggestion_index])


def response_parts(response: dict[str, str | None], index: int) -> tuple[str, str, str]:
    numerator = f"TRY_CAST({quote_ident(str(response['numerator']))} AS DOUBLE)"
    denominator = response["denominator"]
    num_alias = f"resp{index}_num"
    den_alias = f"resp{index}_den"
    value_alias = f"resp{index}"
    if denominator == "__count__":
        num_expr = f"SUM({numerator})"
        den_expr = "COUNT(*)"
    elif denominator:
        den_col = f"TRY_CAST({quote_ident(str(denominator))} AS DOUBLE)"
        num_expr = f"SUM({numerator})"
        den_expr = f"SUM({den_col})"
    else:
        num_expr = f"SUM({numerator})"
        den_expr = f"COUNT({numerator})"
    value_expr = f"{num_alias} / NULLIF({den_alias}, 0)"
    return (
        f"{num_expr} AS {num_alias}",
        f"{den_expr} AS {den_alias}",
        f"{value_expr} AS {value_alias}",
    )


def build_chart_sql(relation: str, x_sql: dict[str, str], responses: list[dict[str, str | None]], include_sigma: bool) -> str:
    metric_selects: list[str] = []
    value_selects: list[str] = []
    for index, response in enumerate(responses):
        num_expr, den_expr, value_expr = response_parts(response, index)
        metric_selects.extend([num_expr, den_expr])
        value_selects.append(value_expr)

    metric_sql = ",\n      ".join(metric_selects)
    value_sql = ",\n    ".join(value_selects)
    if metric_sql:
        metric_sql = ",\n      " + metric_sql
    if value_sql:
        value_sql = ",\n    " + value_sql

    sigma_sql = ""
    sigma_join = ""
    sigma_output = ",\n    sigma.sigma_se,\n    sigma.valid_folds"
    if include_sigma:
        fold_metrics: list[str] = []
        fold_values: list[str] = []
        for index, response in enumerate(responses[:2]):
            num_expr, den_expr, value_expr = response_parts(response, index)
            fold_metrics.extend([num_expr, den_expr])
            fold_values.append(value_expr)
        sigma_sql = f""",
folds AS (
  SELECT
    x_key,
    x_label,
    __fold,
    {', '.join(fold_metrics)}
  FROM keyed
  GROUP BY x_key, x_label, __fold
),
fold_values AS (
  SELECT
    x_key,
    x_label,
    __fold,
    {', '.join(fold_values)}
  FROM folds
),
sigma AS (
  SELECT
    x_key,
    x_label,
    STDDEV_SAMP(resp0 - resp1) / SQRT(COUNT(*)) AS sigma_se,
    COUNT(*) AS valid_folds
  FROM fold_values
  WHERE resp0 IS NOT NULL AND resp1 IS NOT NULL
  GROUP BY x_key, x_label
)"""
        sigma_join = "LEFT JOIN sigma ON agg_values.x_key IS NOT DISTINCT FROM sigma.x_key AND agg_values.x_label = sigma.x_label"
    else:
        sigma_output = ",\n    NULL AS sigma_se,\n    NULL AS valid_folds"
        sigma_join = ""

    return f"""
WITH base AS (
  SELECT ROW_NUMBER() OVER () AS __rownum, * FROM {relation}
),
keyed AS (
  SELECT
    __rownum,
    {x_sql['key']} AS x_key,
    {x_sql['label']} AS x_label,
    {x_sql['sort']} AS x_sort,
    CAST(hash(__rownum) % 20 AS INTEGER) AS __fold,
    *
  FROM base
),
agg AS (
  SELECT
    x_key,
    x_label,
    MIN(x_sort) AS x_sort,
    MIN(__rownum) AS original_order,
    COUNT(*) AS volume
    {metric_sql}
  FROM keyed
  GROUP BY x_key, x_label
),
agg_values AS (
  SELECT
    *{value_sql}
  FROM agg
)
{sigma_sql}
SELECT
    agg_values.*{sigma_output}
FROM agg_values
{sigma_join}
"""


def parse_group_threshold(value: str, total_volume: float) -> float:
    raw = value.strip().lower()
    if raw in {"", "0", "none", "-"}:
        return 0
    if raw.endswith("%"):
        parsed = parse_positive_float(raw[:-1])
        return total_volume * parsed / 100 if parsed else 0
    return parse_positive_float(raw) or 0


def apply_low_weight_grouping(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str | None]],
    x_kind: str,
    threshold: str,
) -> list[dict[str, Any]]:
    total_volume = sum(float(row.get("volume") or 0) for row in rows)
    threshold_value = parse_group_threshold(threshold, total_volume)
    if threshold_value <= 0 or len(rows) < 3:
        return [normalise_row(row, responses) for row in rows]

    normalised = [normalise_row(row, responses) for row in rows]
    if x_kind in {"numeric", "date", "datetime"}:
        ordered = sorted(normalised, key=lambda r: (r["x_sort"] is None, r["x_sort"]))
        low: list[dict[str, Any]] = []
        high: list[dict[str, Any]] = []
        cumulative = 0.0
        for row in ordered:
            volume = float(row["volume"] or 0)
            if cumulative + volume <= threshold_value:
                low.append(row)
                cumulative += volume
            else:
                break
        cumulative = 0.0
        for row in reversed(ordered[len(low):]):
            volume = float(row["volume"] or 0)
            if cumulative + volume <= threshold_value:
                high.append(row)
                cumulative += volume
            else:
                break
        high = list(reversed(high))
        middle = ordered[len(low): len(ordered) - len(high) if high else len(ordered)]
        result: list[dict[str, Any]] = []
        if len(low) > 1:
            result.append(combine_rows(low, "Low tail", responses, is_tail=True))
        else:
            result.extend(low)
        result.extend(middle)
        if len(high) > 1:
            result.append(combine_rows(high, "High tail", responses, is_tail=True))
        else:
            result.extend(high)
        return result

    rare = [row for row in normalised if float(row["volume"] or 0) <= threshold_value]
    common = [row for row in normalised if float(row["volume"] or 0) > threshold_value]
    if len(rare) > 1:
        common.append(combine_rows(rare, "Other", responses, is_tail=True))
    else:
        common.extend(rare)
    return common


def normalise_row(row: dict[str, Any], responses: list[dict[str, str | None]]) -> dict[str, Any]:
    result = {
        "x": str(row.get("x_label")),
        "x_sort": row.get("x_sort"),
        "original_order": int(row.get("original_order") or 0),
        "volume": int(row.get("volume") or 0),
        "is_tail": False,
        "sigma_se": json_number(row.get("sigma_se")),
        "valid_folds": json_number(row.get("valid_folds")),
    }
    for index, _ in enumerate(responses):
        result[f"resp{index}_num"] = json_number(row.get(f"resp{index}_num"))
        result[f"resp{index}_den"] = json_number(row.get(f"resp{index}_den"))
        result[f"resp{index}"] = json_number(row.get(f"resp{index}"))
    return result


def combine_rows(rows: list[dict[str, Any]], label: str, responses: list[dict[str, str | None]], is_tail: bool) -> dict[str, Any]:
    combined = {
        "x": label,
        "x_sort": rows[0].get("x_sort"),
        "original_order": min(int(row.get("original_order") or 0) for row in rows),
        "volume": sum(int(row.get("volume") or 0) for row in rows),
        "is_tail": is_tail,
        "sigma_se": None,
        "valid_folds": None,
    }
    for index, _ in enumerate(responses):
        num = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        den = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        combined[f"resp{index}_num"] = json_number(num)
        combined[f"resp{index}_den"] = json_number(den)
        combined[f"resp{index}"] = json_number(num / den) if den else None
    return combined


def sort_rows(rows: list[dict[str, Any]], x_kind: str, sort: str) -> list[dict[str, Any]]:
    if x_kind not in {"categorical"}:
        return sorted(rows, key=lambda r: (r["x_sort"] is None, r["x_sort"]))
    if sort == "alpha":
        return sorted(rows, key=lambda r: str(r["x"]).lower())
    if sort == "volume":
        return sorted(rows, key=lambda r: (not r.get("is_tail"), -(r["volume"] or 0), str(r["x"]).lower()))
    if sort == "response":
        return sorted(rows, key=lambda r: (r.get("resp0") is None, -(r.get("resp0") or 0), str(r["x"]).lower()))
    return sorted(rows, key=lambda r: r.get("original_order") or 0)


def apply_transform(
    rows: list[dict[str, Any]],
    responses: list[dict[str, str | None]],
    transform: str,
    sigma_multiplier: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    averages: dict[int, float | None] = {}
    for index, _ in enumerate(responses):
        num = sum(float(row.get(f"resp{index}_num") or 0) for row in rows)
        den = sum(float(row.get(f"resp{index}_den") or 0) for row in rows)
        averages[index] = num / den if den else None

    display: list[dict[str, Any]] = []
    invalid_count = 0
    for row in rows:
        out = {
            "x": row["x"],
            "volume": row["volume"],
            "is_tail": bool(row.get("is_tail")),
            "valid_folds": row.get("valid_folds"),
        }
        for index, _ in enumerate(responses):
            out[f"resp{index}"] = transform_value(row.get(f"resp{index}"), transform, averages[index])
            if row.get(f"resp{index}") is not None and out[f"resp{index}"] is None:
                invalid_count += 1
        if sigma_multiplier > 0 and len(responses) >= 2 and row.get("sigma_se") is not None and row.get("resp1") is not None:
            se = float(row["sigma_se"])
            expected = float(row["resp1"])
            out["resp1_low"] = transform_value(expected - sigma_multiplier * se, transform, averages[1])
            out["resp1_high"] = transform_value(expected + sigma_multiplier * se, transform, averages[1])
        display.append(out)

    if invalid_count:
        warnings.append(f"{invalid_count} response values could not be shown because they are outside the {transform} transform domain.")
    return display


def transform_value(value: Any, transform: str, average: float | None) -> float | int | None:
    number = json_number(value)
    if number is None:
        return None
    x = float(number)
    try:
        if transform == "log":
            return json_number(math.log(x)) if x > 0 else None
        if transform == "exp":
            return json_number(math.exp(x))
        if transform == "logit":
            return json_number(math.log(x / (1 - x))) if 0 < x < 1 else None
        if transform == "zero":
            return json_number(x - average) if average is not None else None
        if transform == "one":
            return json_number(x / average) if average not in (None, 0) else None
    except (OverflowError, ValueError):
        return None
    return json_number(x)
