from __future__ import annotations

from .dataset import Dataset, ModelPredictionSource, ModelSourceBinding
from .features import load_features, resolve_features_path
from .filters import load_saved_filters, resolve_filters_path
from .kpis import load_kpis, resolve_kpis_path
from .metrics import (
    build_denominator_summary_sql,
    build_response_summary_sql,
    denominator_exclusion_warnings,
    denominator_valid_condition,
    denominator_value_sql,
    denominator_warnings,
    missing_response_condition,
    normalise_denominator,
    response_parts,
    response_summary,
    response_summary_for_relation,
    response_value_sql,
    summarize_denominator,
    summarize_denominator_for_relation,
    weighted_value_sql,
)
from .metric_sources import (
    PRIMARY_MODEL_PREDICTION_COLUMNS,
    add_metric_field,
    field_source_id,
    has_denominator_column,
    metric_relation_context,
    mixed_metric_relation_sql,
    normalise_denominator_source,
    relation_row_count,
)
from .schema import ColumnInfo, duckdb_error_message, infer_kind, is_numeric_kind, json_number, parse_positive_float, suggested_band_width
from .sql import quote_ident, sql_literal
from .workspace import dataset_slug, dataset_workspace_metadata, dataset_workspace_root

__all__ = [
    "build_denominator_summary_sql",
    "build_response_summary_sql",
    "ColumnInfo",
    "Dataset",
    "dataset_slug",
    "dataset_workspace_metadata",
    "dataset_workspace_root",
    "denominator_exclusion_warnings",
    "denominator_valid_condition",
    "denominator_value_sql",
    "denominator_warnings",
    "duckdb_error_message",
    "infer_kind",
    "is_numeric_kind",
    "json_number",
    "load_features",
    "load_kpis",
    "load_saved_filters",
    "metric_relation_context",
    "mixed_metric_relation_sql",
    "ModelPredictionSource",
    "ModelSourceBinding",
    "missing_response_condition",
    "normalise_denominator",
    "normalise_denominator_source",
    "parse_positive_float",
    "quote_ident",
    "response_parts",
    "response_summary",
    "response_summary_for_relation",
    "response_value_sql",
    "relation_row_count",
    "resolve_features_path",
    "resolve_filters_path",
    "resolve_kpis_path",
    "sql_literal",
    "summarize_denominator",
    "summarize_denominator_for_relation",
    "suggested_band_width",
    "weighted_value_sql",
    "add_metric_field",
    "field_source_id",
    "has_denominator_column",
    "PRIMARY_MODEL_PREDICTION_COLUMNS",
]
