from __future__ import annotations

import re
from typing import Any

import duckdb
from fastapi import FastAPI

from py_lucidum.core import (
    Dataset,
    ModelPredictionSource,
    ModelSourceBinding,
    is_numeric_kind,
    quote_ident,
    sql_literal,
)


RATIO_COLUMN = "gbm_to_glm_ratio"
PREDICTION_RATIO_COLUMN = "prediction_ratio"
RATIO_KIND = "model_ratio"
SOURCE_RE = re.compile(r"^model_ratio:gbm_to_glm_ratio:([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+)$")
PREDICTION_RATIO_SOURCE_RE = re.compile(
    r"^model_ratio:prediction_ratio:(glm|gbm|other):([A-Za-z0-9_.-]+):"
    r"(glm|gbm|other):([A-Za-z0-9_.-]+)$"
)
PRIMARY_PREDICTION_COLUMNS = {
    "glm": "glm_prediction",
    "gbm": "gbm_prediction",
}


def encode_prediction_ratio_column(column_name: str) -> str:
    return str(column_name).encode("utf-8").hex()


def decode_prediction_ratio_column(token: str) -> str:
    try:
        return bytes.fromhex(str(token)).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return ""


class ModelPredictionRatioSourceProvider:
    def __init__(
        self,
        gbm_store: Any | None = None,
        glm_store: Any | None = None,
        dataset: Dataset | None = None,
        prediction_providers: tuple[Any, ...] = (),
    ):
        self.gbm_store = gbm_store
        self.glm_store = glm_store
        self.dataset = dataset
        self.prediction_providers = tuple(prediction_providers)

    def active_source_id(self) -> str:
        if self.gbm_store is None or self.glm_store is None:
            return ""
        gbm_model_id = str(self.gbm_store.active_model_id() or "").strip()
        glm_model_id = str(self.glm_store.active_model_id() or "").strip()
        if not gbm_model_id or not glm_model_id:
            return ""
        gbm_source_id = self.gbm_store.source_id(gbm_model_id, "predictions")
        glm_source_id = self.glm_store.source_id(glm_model_id)
        if self.gbm_store.source_ref(gbm_source_id) is None or self.glm_store.source_ref(glm_source_id) is None:
            return ""
        return f"model_ratio:{RATIO_COLUMN}:{gbm_model_id}:{glm_model_id}"

    @staticmethod
    def prediction_ratio_source_id(
        challenger_family: str,
        challenger_model_id: str,
        baseline_family: str,
        baseline_model_id: str,
    ) -> str:
        challenger_token = (
            encode_prediction_ratio_column(challenger_model_id)
            if challenger_family == "other"
            else challenger_model_id
        )
        baseline_token = (
            encode_prediction_ratio_column(baseline_model_id)
            if baseline_family == "other"
            else baseline_model_id
        )
        return (
            f"model_ratio:{PREDICTION_RATIO_COLUMN}:"
            f"{challenger_family}:{challenger_token}:{baseline_family}:{baseline_token}"
        )

    def has_source(self, source_id: str) -> bool:
        context = self._source_context(source_id)
        if context is None:
            return False
        return self._side_valid(context, "challenger") and self._side_valid(context, "baseline")

    def relation_sql(self, source_id: str) -> str:
        context = self._source_context(source_id)
        if context is None:
            raise ValueError("Choose a valid data source")
        if not self._side_valid(context, "challenger") or not self._side_valid(context, "baseline"):
            raise ValueError("Choose a valid data source")
        challenger_column = quote_ident(context["challenger_column"])
        baseline_column = quote_ident(context["baseline_column"])
        ratio_sql = quote_ident(context["output_column"])
        challenger_relation = self._side_keyed_relation(context, "challenger")
        baseline_relation = self._side_keyed_relation(context, "baseline")
        return f"""(
SELECT
  challenger.__lucidum_row_id,
  CASE
    WHEN TRY_CAST(baseline.{baseline_column} AS DOUBLE) IS NULL THEN NULL
    WHEN TRY_CAST(baseline.{baseline_column} AS DOUBLE) = 0 THEN NULL
    ELSE TRY_CAST(challenger.{challenger_column} AS DOUBLE)
      / TRY_CAST(baseline.{baseline_column} AS DOUBLE)
  END AS {ratio_sql}
FROM {challenger_relation} challenger
INNER JOIN {baseline_relation} baseline USING (__lucidum_row_id)
)"""

    def prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        if not self.has_source(source_id):
            return None
        context = self._source_context(source_id)
        if context is None:
            return None
        output_column = context["output_column"]
        binding = self._ratio_binding(context)
        return ModelPredictionSource(
            source_id=source_id,
            column=output_column,
            relation_sql=self.relation_sql(source_id),
            active=True,
            binding=binding,
            bindings={output_column: binding} if binding is not None else {},
        )

    def data_sources(self, dataset: Dataset) -> list[dict[str, Any]]:
        # Keep publishing the legacy active GBM/GLM source so old URLs and
        # favourites retain their active-model rebinding behaviour. Exact
        # model-pair sources are resolved dynamically and need no schema row.
        source_id = self.active_source_id()
        if not source_id or self.gbm_store is None or self.glm_store is None:
            return []
        context = self._source_context(source_id) or {}
        try:
            gbm_ref = self.gbm_store.source_ref(str(context.get("challenger_source_id") or ""))
            glm_ref = self.glm_store.source_ref(str(context.get("baseline_source_id") or ""))
            if gbm_ref is None or glm_ref is None:
                raise ValueError("Choose valid GBM and GLM prediction sources")
            gbm_path = self.gbm_store.source_path(gbm_ref.model_id, "predictions")
            glm_path = self.glm_store.source_path(glm_ref.model_id, "predictions")
            gbm_metadata = dataset.parquet_artifact_metadata(gbm_path)
            glm_metadata = dataset.parquet_artifact_metadata(glm_path)
            if "__lucidum_row_id" not in {column.name for column in gbm_metadata.columns}:
                raise ValueError("Unsupported GBM prediction artifact schema")
            if "__lucidum_row_id" not in {column.name for column in glm_metadata.columns}:
                raise ValueError("Unsupported GLM prediction artifact schema")
            with dataset.lock:
                row_count = int(
                    dataset.con.execute(
                        f"""
SELECT COUNT(*)
FROM read_parquet({sql_literal(str(gbm_path))}) gbm
INNER JOIN read_parquet({sql_literal(str(glm_path))}) glm USING (__lucidum_row_id)
"""
                    ).fetchone()[0]
                )
        except (duckdb.Error, FileNotFoundError, KeyError, ValueError):
            row_count = int(dataset.schema_for_source(source_id)["row_count"])
        return [
            {
                "id": source_id,
                "label": "GBM / GLM prediction ratio",
                "kind": RATIO_KIND,
                "active": True,
                "gbm_model_id": context.get("challenger_model_id"),
                "glm_model_id": context.get("baseline_model_id"),
                "row_count": row_count,
                "columns": [
                    {
                        "name": RATIO_COLUMN,
                        "duckdb_type": "DOUBLE",
                        "kind": "numeric",
                        "band_suggestion": None,
                    }
                ],
            }
        ]

    def _source_context(self, source_id: str) -> dict[str, str] | None:
        value = str(source_id or "")
        legacy_match = SOURCE_RE.match(value)
        if legacy_match:
            if value != self.active_source_id() or self.gbm_store is None or self.glm_store is None:
                return None
            gbm_model_id, glm_model_id = legacy_match.groups()
            return {
                "challenger_family": "gbm",
                "challenger_model_id": gbm_model_id,
                "challenger_source_id": self.gbm_store.source_id(gbm_model_id, "predictions"),
                "challenger_column": PRIMARY_PREDICTION_COLUMNS["gbm"],
                "baseline_family": "glm",
                "baseline_model_id": glm_model_id,
                "baseline_source_id": self.glm_store.source_id(glm_model_id),
                "baseline_column": PRIMARY_PREDICTION_COLUMNS["glm"],
                "output_column": RATIO_COLUMN,
            }

        match = PREDICTION_RATIO_SOURCE_RE.match(value)
        if not match:
            return None
        challenger_family, challenger_model_id, baseline_family, baseline_model_id = match.groups()
        if challenger_family == baseline_family and challenger_model_id == baseline_model_id:
            return None
        challenger = self._side_context(challenger_family, challenger_model_id)
        baseline = self._side_context(baseline_family, baseline_model_id)
        if challenger is None or baseline is None:
            return None
        return {
            **{f"challenger_{key}": item for key, item in challenger.items()},
            **{f"baseline_{key}": item for key, item in baseline.items()},
            "output_column": PREDICTION_RATIO_COLUMN,
        }

    def _side_context(self, family: str, identity: str) -> dict[str, str] | None:
        if family == "other":
            column = decode_prediction_ratio_column(identity)
            if not column or encode_prediction_ratio_column(column) != identity:
                return None
            if not self._dataset_baseline_column(column):
                return None
            return {
                "family": "other",
                "model_id": identity,
                "source_id": "dataset",
                "column": column,
            }
        source_id = self._prediction_source_id(family, identity)
        if not source_id:
            return None
        return {
            "family": family,
            "model_id": identity,
            "source_id": source_id,
            "column": PRIMARY_PREDICTION_COLUMNS[family],
        }

    def _prediction_source_id(self, family: str, model_id: str) -> str:
        if family == "gbm" and self.gbm_store is not None:
            return str(self.gbm_store.source_id(model_id, "predictions"))
        if family == "glm" and self.glm_store is not None:
            return str(self.glm_store.source_id(model_id))
        if family in PRIMARY_PREDICTION_COLUMNS and self.prediction_providers:
            return f"{family}:{model_id}:predictions"
        return ""

    def _prediction_source(self, family: str, source_id: str) -> ModelPredictionSource | None:
        for provider in self.prediction_providers:
            prediction_source = getattr(provider, "prediction_source", None)
            if not callable(prediction_source):
                continue
            resolved = prediction_source(source_id)
            if resolved is not None:
                return resolved
        if family == "gbm" and self.gbm_store is not None:
            from py_lucidum.tools.gbm.store import GbmSourceProvider

            return GbmSourceProvider(self.gbm_store).prediction_source(source_id)
        if family == "glm" and self.glm_store is not None:
            from py_lucidum.tools.glm.store import GlmSourceProvider

            return GlmSourceProvider(self.glm_store).prediction_source(source_id)
        return None

    def _dataset_baseline_column(self, column_name: str) -> bool:
        if self.dataset is None:
            return False
        return any(
            column.name == column_name and is_numeric_kind(column.kind)
            for column in self.dataset.valid_schema_columns()
        )

    def _side_valid(self, context: dict[str, str], role: str) -> bool:
        family = context[f"{role}_family"]
        if family == "other":
            return self._dataset_baseline_column(context[f"{role}_column"])
        return self._prediction_source(family, context[f"{role}_source_id"]) is not None

    def _side_keyed_relation(self, context: dict[str, str], role: str) -> str:
        family = context[f"{role}_family"]
        if family != "other":
            source = self._prediction_source(family, context[f"{role}_source_id"])
            if source is None:
                raise ValueError(f"Choose a valid {role.title()} source")
            return source.relation_sql
        if self.dataset is None:
            raise ValueError(f"Choose a valid {role.title()} source")
        column = quote_ident(context[f"{role}_column"])
        return f"""(
SELECT
  ROW_NUMBER() OVER () AS __lucidum_row_id,
  {column}
FROM {self.dataset.relation_sql()}
)"""

    def _ratio_binding(self, context: dict[str, str]) -> ModelSourceBinding | None:
        bindings: dict[str, ModelSourceBinding] = {}
        for role in ("challenger", "baseline"):
            family = context[f"{role}_family"]
            if family == "other":
                continue
            source = self._prediction_source(family, context[f"{role}_source_id"])
            if source is None:
                return None
            binding = source.bindings.get(context[f"{role}_column"]) or source.binding
            if binding is None:
                return None
            bindings[role] = binding
        if not bindings:
            return None
        anchor_binding = next(iter(bindings.values()))
        if any(
            binding.base_where_sql != anchor_binding.base_where_sql
            or binding.base_columns != anchor_binding.base_columns
            for binding in bindings.values()
        ):
            return None
        side_relations: dict[str, str] = {}
        identity_sqls: list[str] = []
        side_cache_keys: list[tuple[Any, ...]] = []
        for role in ("challenger", "baseline"):
            binding = bindings.get(role)
            if binding is not None:
                side_relations[role] = binding.relation_sql
                identity_sqls.extend(binding.identity_sqls)
                side_cache_keys.append(binding.cache_key)
                continue
            if self.dataset is None or not self._dataset_baseline_column(context[f"{role}_column"]):
                return None
            column = quote_ident(context[f"{role}_column"])
            where_sql = f"\nWHERE {anchor_binding.base_where_sql}" if anchor_binding.base_where_sql else ""
            side_relations[role] = f"""(
SELECT {column}
FROM {self.dataset.relation_sql()}{where_sql}
)"""
            side_cache_keys.append(("dataset_column", context[f"{role}_column"], self.dataset.relation_sql()))
        challenger_column = quote_ident(context["challenger_column"])
        baseline_column = quote_ident(context["baseline_column"])
        output_column = context["output_column"]
        ratio_sql = quote_ident(output_column)
        return ModelSourceBinding(
            relation_sql=f"""(
SELECT
  CASE
    WHEN TRY_CAST(baseline.{baseline_column} AS DOUBLE) IS NULL THEN NULL
    WHEN TRY_CAST(baseline.{baseline_column} AS DOUBLE) = 0 THEN NULL
    ELSE TRY_CAST(challenger.{challenger_column} AS DOUBLE)
      / TRY_CAST(baseline.{baseline_column} AS DOUBLE)
  END AS {ratio_sql}
FROM {side_relations["challenger"]} challenger
POSITIONAL JOIN {side_relations["baseline"]} baseline
)""",
            columns=(output_column,),
            identity_sqls=tuple(identity_sqls),
            base_where_sql=anchor_binding.base_where_sql,
            base_columns=anchor_binding.base_columns,
            cache_key=(
                "prediction_ratio",
                context["challenger_source_id"],
                context["baseline_source_id"],
                *side_cache_keys,
            ),
        )


# Retain the public name used by the original mixed-model implementation.
GbmGlmRatioSourceProvider = ModelPredictionRatioSourceProvider


def register_model_ratio_source_provider(app: FastAPI, dataset: Dataset) -> None:
    gbm_store = getattr(app.state, "gbm_store", None)
    glm_store = getattr(app.state, "glm_store", None)
    if gbm_store is None and glm_store is None:
        return
    dataset.register_data_source_provider(ModelPredictionRatioSourceProvider(gbm_store, glm_store, dataset))


__all__ = [
    "GbmGlmRatioSourceProvider",
    "ModelPredictionRatioSourceProvider",
    "PREDICTION_RATIO_COLUMN",
    "PREDICTION_RATIO_SOURCE_RE",
    "RATIO_COLUMN",
    "RATIO_KIND",
    "SOURCE_RE",
    "decode_prediction_ratio_column",
    "encode_prediction_ratio_column",
    "register_model_ratio_source_provider",
]
