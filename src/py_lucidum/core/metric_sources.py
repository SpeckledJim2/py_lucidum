from __future__ import annotations

from typing import Any, Iterable

from .dataset import Dataset, ModelPredictionSource
from .schema import ColumnInfo, is_numeric_kind
from .sql import quote_ident


MODEL_PREDICTION_COLUMNS = {
    "gbm_prediction",
    "gbm_prediction_rate",
    "gbm_tabulated_prediction",
    "glm_prediction",
    "glm_prediction_rate",
    "glm_tabulated_prediction",
}
PRIMARY_MODEL_PREDICTION_COLUMNS = {"gbm_prediction", "glm_prediction"}


def has_denominator_column(raw: Any) -> bool:
    return str(raw or "").strip().lower() not in {
        "",
        "__none__",
        "n",
        "average row value",
    }


def field_source_id(dataset: Dataset, raw_source: Any, fallback_source: str = "dataset") -> str:
    raw = str(raw_source or "").strip()
    if raw:
        return dataset.normalise_source(raw)
    return fallback_source or "dataset"


def normalise_denominator_source(dataset: Dataset, raw_source: Any, denominator: Any) -> str:
    source_id = field_source_id(dataset, raw_source, "dataset")
    column_name = str(denominator or "").strip()
    if not has_denominator_column(column_name):
        return "dataset"
    if source_id == "dataset":
        return source_id
    prediction_source = dataset.model_prediction_source(source_id)
    if prediction_source is None or column_name not in PRIMARY_MODEL_PREDICTION_COLUMNS:
        raise ValueError("Choose an active primary model prediction as Denominator")
    source_columns = dataset.column_map_for_source(source_id)
    column = source_columns.get(column_name)
    if column is None or not is_numeric_kind(column.kind):
        raise ValueError("Choose an active primary model prediction as Denominator")
    return source_id


def metric_relation_context(
    dataset: Dataset,
    *,
    source_id: Any = None,
    fields: Iterable[tuple[str, Any]] = (),
) -> dict[str, Any]:
    base_source = dataset.normalise_source(source_id)
    dataset_columns = dataset.column_map()
    columns = dict(dataset_columns)
    prediction_sources: dict[str, ModelPredictionSource] = {}
    resolved_sources: list[str] = []
    for column_name, raw_source in fields:
        resolved_source = field_source_id(dataset, raw_source, base_source)
        resolved_sources.append(resolved_source)
        add_metric_field(
            dataset,
            columns,
            dataset_columns,
            prediction_sources,
            str(column_name or "").strip(),
            resolved_source,
        )

    if prediction_sources:
        relation = mixed_metric_relation_sql(dataset, list(prediction_sources.values()))
    elif base_source == "dataset":
        relation = dataset.relation_sql()
    else:
        relation = dataset.relation_sql_for_source(base_source)
        columns = dataset.column_map_for_source(base_source)
    return {
        "source_id": base_source,
        "relation": relation,
        "columns": columns,
        "row_count": relation_row_count(dataset, relation),
        "field_sources": resolved_sources,
    }


def add_metric_field(
    dataset: Dataset,
    columns: dict[str, ColumnInfo],
    dataset_columns: dict[str, ColumnInfo],
    prediction_sources: dict[str, ModelPredictionSource],
    column_name: str,
    source_id: str,
) -> None:
    if not column_name:
        return
    prediction_source = dataset.model_prediction_source(source_id)
    if prediction_source is not None:
        source_columns = dataset.column_map_for_source(source_id)
        source_column = source_columns.get(column_name)
        if column_name in MODEL_PREDICTION_COLUMNS and source_column is not None and is_numeric_kind(source_column.kind):
            if not prediction_source.relation_sql:
                raise ValueError("Choose a valid model prediction source")
            prediction_sources[f"{prediction_source.source_id}:{column_name}"] = ModelPredictionSource(
                source_id=prediction_source.source_id,
                column=column_name,
                relation_sql=prediction_source.relation_sql,
                active=prediction_source.active,
                binding=prediction_source.bindings.get(column_name),
                bindings=prediction_source.bindings,
            )
            columns[column_name] = ColumnInfo(name=column_name, duckdb_type="DOUBLE", kind="numeric")
            return
        if column_name == prediction_source.column:
            if not prediction_source.column or not prediction_source.relation_sql:
                raise ValueError("Choose a valid model prediction source")
            prediction_sources[prediction_source.source_id] = prediction_source
            columns[column_name] = ColumnInfo(name=column_name, duckdb_type="DOUBLE", kind="numeric")
            return
        raise ValueError("Choose a valid model prediction column")
    if column_name in dataset_columns:
        columns[column_name] = dataset_columns[column_name]
        return
    raise ValueError("Choose a valid dataset column")


def mixed_metric_relation_sql(dataset: Dataset, prediction_sources: list[ModelPredictionSource]) -> str:
    positional_sql = positional_mixed_metric_relation_sql(dataset, prediction_sources)
    if positional_sql:
        return positional_sql
    return keyed_mixed_metric_relation_sql(dataset, prediction_sources)


def positional_mixed_metric_relation_sql(
    dataset: Dataset,
    prediction_sources: list[ModelPredictionSource],
) -> str:
    if not prediction_sources:
        return ""
    bindings = []
    for source in prediction_sources:
        binding = source.binding
        if binding is None or source.column not in binding.columns:
            return ""
        if not dataset.model_source_binding_eligible(binding):
            return ""
        bindings.append(binding)
    base_where_sql = bindings[0].base_where_sql
    if any(binding.base_where_sql != base_where_sql for binding in bindings):
        return ""

    prediction_columns = {source.column for source in prediction_sources}
    dataset_columns = [
        column.name for column in dataset.valid_schema_columns() if column.name not in prediction_columns
    ]
    source_column_sql = ",\n    ".join(quote_ident(name) for name in dataset_columns)
    source_column_suffix = f",\n    {source_column_sql}" if source_column_sql else ""
    where_sql = f"\n  WHERE {base_where_sql}" if base_where_sql else ""
    joins: list[str] = []
    selects = [f"base.{quote_ident(name)}" for name in dataset_columns]
    for index, source in enumerate(prediction_sources):
        alias = f"prediction_{index}"
        joins.append(f"POSITIONAL JOIN {source.binding.relation_sql} {alias}")
        selects.append(f"{alias}.{quote_ident(source.column)} AS {quote_ident(source.column)}")
    select_sql = ",\n  ".join(selects) if selects else "*"
    join_sql = "\n".join(joins)
    return f"""(
SELECT
  {select_sql}
FROM (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{source_column_suffix}
  FROM {dataset.relation_sql()}
  {where_sql}
) base
{join_sql}
)"""


def keyed_mixed_metric_relation_sql(
    dataset: Dataset,
    prediction_sources: list[ModelPredictionSource],
) -> str:
    prediction_columns = {source.column for source in prediction_sources}
    dataset_columns = [
        column.name for column in dataset.valid_schema_columns() if column.name not in prediction_columns
    ]
    source_column_sql = ",\n    ".join(quote_ident(name) for name in dataset_columns)
    source_column_suffix = f",\n    {source_column_sql}" if source_column_sql else ""
    joins: list[str] = []
    selects = [f"base.{quote_ident(name)}" for name in dataset_columns]
    for index, source in enumerate(prediction_sources):
        alias = f"prediction_{index}"
        joins.append(f"LEFT JOIN {source.relation_sql} {alias} USING (__lucidum_row_id)")
        selects.append(f"{alias}.{quote_ident(source.column)} AS {quote_ident(source.column)}")
    select_sql = ",\n  ".join(selects) if selects else "*"
    join_sql = "\n".join(joins)
    return f"""(
WITH dataset_rows AS (
  SELECT
    ROW_NUMBER() OVER () AS __lucidum_row_id{source_column_suffix}
  FROM {dataset.relation_sql()}
)
SELECT
  {select_sql}
FROM dataset_rows base
{join_sql}
)"""


def relation_row_count(dataset: Dataset, relation: str, filter_sql: str = "") -> int:
    where_sql = f" WHERE ({filter_sql})" if filter_sql else ""
    value = dataset.con.execute(f"SELECT COUNT(*) FROM {relation}{where_sql}").fetchone()[0]
    return int(value)
