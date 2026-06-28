from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI

from py_lucidum.core import Dataset, ModelPredictionSource, ModelSourceBinding, quote_ident


RATIO_COLUMN = "gbm_to_glm_ratio"
RATIO_KIND = "model_ratio"
SOURCE_RE = re.compile(r"^model_ratio:gbm_to_glm_ratio:([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+)$")


class GbmGlmRatioSourceProvider:
    def __init__(self, gbm_store: Any, glm_store: Any):
        self.gbm_store = gbm_store
        self.glm_store = glm_store

    def active_source_id(self) -> str:
        gbm_model_id = str(self.gbm_store.active_model_id() or "").strip()
        glm_model_id = str(self.glm_store.active_model_id() or "").strip()
        if not gbm_model_id or not glm_model_id:
            return ""
        gbm_source_id = self.gbm_store.source_id(gbm_model_id, "predictions")
        glm_source_id = self.glm_store.source_id(glm_model_id)
        if self.gbm_store.source_ref(gbm_source_id) is None or self.glm_store.source_ref(glm_source_id) is None:
            return ""
        return f"model_ratio:{RATIO_COLUMN}:{gbm_model_id}:{glm_model_id}"

    def has_source(self, source_id: str) -> bool:
        return bool(source_id and source_id == self.active_source_id())

    def relation_sql(self, source_id: str) -> str:
        context = self._source_context(source_id)
        if context is None:
            raise ValueError("Choose a valid data source")
        gbm_source = self._gbm_prediction_source(context["gbm_source_id"])
        glm_source = self._glm_prediction_source(context["glm_source_id"])
        if gbm_source is None or glm_source is None:
            raise ValueError("Choose a valid data source")
        gbm_relation = gbm_source.relation_sql
        glm_relation = glm_source.relation_sql
        ratio_sql = quote_ident(RATIO_COLUMN)
        return f"""(
SELECT
  gbm.__lucidum_row_id,
  CASE
    WHEN TRY_CAST(glm.glm_prediction AS DOUBLE) IS NULL THEN NULL
    WHEN TRY_CAST(glm.glm_prediction AS DOUBLE) = 0 THEN NULL
    ELSE TRY_CAST(gbm.gbm_prediction AS DOUBLE) / TRY_CAST(glm.glm_prediction AS DOUBLE)
  END AS {ratio_sql}
FROM {gbm_relation} gbm
INNER JOIN {glm_relation} glm USING (__lucidum_row_id)
)"""

    def prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        if not self.has_source(source_id):
            return None
        context = self._source_context(source_id)
        binding = self._ratio_binding(context) if context else None
        return ModelPredictionSource(
            source_id=source_id,
            column=RATIO_COLUMN,
            relation_sql=self.relation_sql(source_id),
            active=True,
            binding=binding,
            bindings={RATIO_COLUMN: binding} if binding is not None else {},
        )

    def data_sources(self, dataset: Dataset) -> list[dict[str, Any]]:
        source_id = self.active_source_id()
        if not source_id:
            return []
        schema = dataset.schema_for_source(source_id)
        context = self._source_context(source_id) or {}
        return [
            {
                "id": source_id,
                "label": "GBM / GLM prediction ratio",
                "kind": RATIO_KIND,
                "active": True,
                "gbm_model_id": context.get("gbm_model_id"),
                "glm_model_id": context.get("glm_model_id"),
                "row_count": schema["row_count"],
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
        match = SOURCE_RE.match(str(source_id or ""))
        if not match:
            return None
        gbm_model_id, glm_model_id = match.groups()
        active_source_id = self.active_source_id()
        if source_id != active_source_id:
            return None
        return {
            "gbm_model_id": gbm_model_id,
            "glm_model_id": glm_model_id,
            "gbm_source_id": self.gbm_store.source_id(gbm_model_id, "predictions"),
            "glm_source_id": self.glm_store.source_id(glm_model_id),
        }

    def _gbm_prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        from py_lucidum.tools.gbm.store import GbmSourceProvider

        return GbmSourceProvider(self.gbm_store).prediction_source(source_id)

    def _glm_prediction_source(self, source_id: str) -> ModelPredictionSource | None:
        from py_lucidum.tools.glm.store import GlmSourceProvider

        return GlmSourceProvider(self.glm_store).prediction_source(source_id)

    def _ratio_binding(self, context: dict[str, str]) -> ModelSourceBinding | None:
        gbm_source = self._gbm_prediction_source(context["gbm_source_id"])
        glm_source = self._glm_prediction_source(context["glm_source_id"])
        if gbm_source is None or glm_source is None:
            return None
        gbm_binding = gbm_source.bindings.get("gbm_prediction") or gbm_source.binding
        glm_binding = glm_source.bindings.get("glm_prediction") or glm_source.binding
        if gbm_binding is None or glm_binding is None:
            return None
        if gbm_binding.base_where_sql != glm_binding.base_where_sql:
            return None
        if gbm_binding.base_columns != glm_binding.base_columns:
            return None
        ratio_sql = quote_ident(RATIO_COLUMN)
        return ModelSourceBinding(
            relation_sql=f"""(
SELECT
  CASE
    WHEN TRY_CAST(glm.glm_prediction AS DOUBLE) IS NULL THEN NULL
    WHEN TRY_CAST(glm.glm_prediction AS DOUBLE) = 0 THEN NULL
    ELSE TRY_CAST(gbm.gbm_prediction AS DOUBLE) / TRY_CAST(glm.glm_prediction AS DOUBLE)
  END AS {ratio_sql}
FROM {gbm_binding.relation_sql} gbm
POSITIONAL JOIN {glm_binding.relation_sql} glm
)""",
            columns=(RATIO_COLUMN,),
            identity_sqls=(*gbm_binding.identity_sqls, *glm_binding.identity_sqls),
            base_where_sql=gbm_binding.base_where_sql,
            base_columns=gbm_binding.base_columns,
            cache_key=("ratio", gbm_binding.cache_key, glm_binding.cache_key),
        )


def register_model_ratio_source_provider(app: FastAPI, dataset: Dataset) -> None:
    gbm_store = getattr(app.state, "gbm_store", None)
    glm_store = getattr(app.state, "glm_store", None)
    if gbm_store is None or glm_store is None:
        return
    dataset.register_data_source_provider(GbmGlmRatioSourceProvider(gbm_store, glm_store))


__all__ = [
    "GbmGlmRatioSourceProvider",
    "RATIO_COLUMN",
    "RATIO_KIND",
    "register_model_ratio_source_provider",
]
