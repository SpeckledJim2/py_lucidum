from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal
from py_lucidum.model_metrics import split_gini_metrics
from py_lucidum.tools.glm.store import GlmModelStore

from .interaction_group_model import (
    NoInteractionGroupTreesError,
    extract_lightgbm_interaction_group,
    interaction_group_model_filename,
)
from .sample import (
    SAMPLE_COLUMN,
    create_generated_sample,
    dataset_sample_column,
    generated_sample_is_current,
    generated_sample_relation_sql,
)
from .store import GbmModelStore
from .validation import (
    INIT_SCORE_NONE,
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
    bool_parameter,
    detect_sample_column,
    display_monotonicity,
    feature_interaction_constraint_groups,
    metric,
    normalise_init_score_value,
    normalise_feature_grouping_map,
    normalise_feature_interaction_features,
    normalise_feature_interaction_groupings,
    normalise_feature_interaction_pairs,
    normalise_training_mode,
    objective,
    init_score_requested,
    init_score_transform,
    normalise_features,
    normalise_parameters,
    selected_offset_column,
    selected_response_column,
    uses_log_offset,
    validate_request,
)


ProgressCallback = Callable[[dict[str, Any]], None]
DependencyProgressCallback = Callable[[str], None]
EBM_INITIAL_LEARNING_RATE = 0.3
DEFAULT_SHAP_SAMPLE_SEED = 2026
DEPENDENCY_IMPORT_PACKAGES = {
    "importing_lightgbm": "LightGBM",
    "importing_numpy": "NumPy",
    "importing_pandas": "pandas",
    "importing_polars": "Polars",
    "importing_pyarrow": "PyArrow",
    "importing_cffi": "CFFI",
}


class MissingGbmDependency(RuntimeError):
    def __init__(self, missing: str, hint: str | None = None):
        message = f"Install GBM dependencies with `pip install 'py-lucidum[gbm]'` to train LightGBM models. Missing: {missing}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)
        self.missing = missing
        self.hint = hint


def _dependency_is_loaded(module_name: str) -> bool:
    return module_name in sys.modules


def _report_dependency_import(
    progress_callback: DependencyProgressCallback | None,
    *,
    module_name: str,
    stage: str,
) -> None:
    if progress_callback is not None and not _dependency_is_loaded(module_name):
        progress_callback(stage)


def gbm_dependencies(
    dependency_progress: DependencyProgressCallback | None = None,
) -> tuple[Any, Any, Any]:
    missing: list[str] = []
    _report_dependency_import(
        dependency_progress,
        module_name="lightgbm",
        stage="importing_lightgbm",
    )
    try:
        import lightgbm as lgb  # type: ignore[import-not-found]
    except ImportError:
        lgb = None
        missing.append("lightgbm")
        hint = None
    except OSError as exc:
        lgb = None
        missing.append("lightgbm runtime")
        hint = "On macOS, install the OpenMP runtime with `brew install libomp`." if "libomp" in str(exc) else str(exc)
    else:
        hint = None
    _report_dependency_import(
        dependency_progress,
        module_name="numpy",
        stage="importing_numpy",
    )
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None
        missing.append("numpy")
    _report_dependency_import(
        dependency_progress,
        module_name="pandas",
        stage="importing_pandas",
    )
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        pd = None
        missing.append("pandas")
    if missing:
        raise MissingGbmDependency(", ".join(missing), hint)
    return lgb, np, pd


def gbm_training_dependencies(
    dependency_progress: DependencyProgressCallback | None = None,
) -> tuple[Any, Any, Any, Any]:
    lgb, np, pd = gbm_dependencies(dependency_progress=dependency_progress)
    missing: list[str] = []
    _report_dependency_import(
        dependency_progress,
        module_name="polars",
        stage="importing_polars",
    )
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError:
        pl = None
        missing.append("polars")
    _report_dependency_import(
        dependency_progress,
        module_name="pyarrow",
        stage="importing_pyarrow",
    )
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
    except ImportError:
        pa = None
        missing.append("pyarrow")
    _report_dependency_import(
        dependency_progress,
        module_name="cffi",
        stage="importing_cffi",
    )
    try:
        import cffi  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        missing.append("LightGBM Arrow runtime (cffi)")
    if missing:
        raise MissingGbmDependency(", ".join(missing))
    return lgb, np, pd, pl


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 6)


class EbmStageController:
    def __init__(
        self,
        *,
        lgb: Any,
        metric_name: str,
        target_num_leaves: int,
        configured_learning_rate: Any,
        early_stopping_rounds: int,
        total_iterations: int,
    ):
        self.lgb = lgb
        self.metric_name = metric_name
        self.target_num_leaves = max(2, int(target_num_leaves or 2))
        self.configured_learning_rate = configured_learning_rate
        self.early_stopping_rounds = max(1, int(early_stopping_rounds or 1))
        self.total_iterations = max(1, int(total_iterations or 1))
        self.current_num_leaves = 2
        self.stage_start_iteration = 1
        self.stage_best_iteration: int | None = None
        self.stage_best_score: float | None = None
        self.best_iteration: int | None = None
        self.best_score: float | None = None
        self.best_result_list: list[Any] | None = None
        self.higher_is_better: bool | None = None
        self.stages: list[dict[str, Any]] = []
        self._closed = False
        self.order = 40
        self.before_iteration = False

    def __call__(self, env: Any) -> None:
        current_iteration = int(getattr(env, "iteration", 0) or 0) - int(getattr(env, "begin_iteration", 0) or 0) + 1
        selected = self._selected_metric(getattr(env, "evaluation_result_list", None))
        if not selected:
            return
        score = json_safe_number(selected[2])
        if score is None:
            return
        higher_is_better = bool(selected[3])
        result_list = list(getattr(env, "evaluation_result_list", None) or [])
        self.higher_is_better = higher_is_better
        if self._is_improvement(score, self.stage_best_score, higher_is_better):
            self.stage_best_score = score
            self.stage_best_iteration = current_iteration
        if self._is_improvement(score, self.best_score, higher_is_better):
            self.best_score = score
            self.best_iteration = current_iteration
            self.best_result_list = result_list
        if current_iteration >= self.total_iterations:
            return
        if self.stage_best_iteration is None:
            return
        if current_iteration - self.stage_best_iteration < self.early_stopping_rounds:
            return
        if self.current_num_leaves < self.target_num_leaves:
            self._close_stage(current_iteration, "plateau")
            self.current_num_leaves += 1
            self.stage_start_iteration = current_iteration + 1
            self.stage_best_iteration = None
            self.stage_best_score = None
            reset_params = {"num_leaves": self.current_num_leaves, "learning_rate": self.configured_learning_rate}
            env.model.reset_parameter(reset_params)
            env.params.update(reset_params)
            return
        self._close_stage(current_iteration, "final_plateau")
        self._closed = True
        best_iteration = (self.best_iteration or current_iteration) - 1
        best_score = self.best_result_list or result_list
        raise self.lgb.callback.EarlyStopException(best_iteration, best_score)

    def _selected_metric(self, evaluation_result_list: Any) -> Any | None:
        test_rows = [
            item
            for item in evaluation_result_list or []
            if len(item) >= 4 and str(item[0]).lower() == "test"
        ]
        for item in test_rows:
            if str(item[1]) == self.metric_name:
                return item
        return test_rows[0] if test_rows else None

    def _is_improvement(self, score: float, best_score: float | None, higher_is_better: bool) -> bool:
        if best_score is None:
            return True
        return score > best_score if higher_is_better else score < best_score

    def _close_stage(self, end_iteration: int, reason: str) -> None:
        if self.stages and self.stages[-1].get("end_iteration") is None:
            self.stages[-1].update(
                {
                    "end_iteration": end_iteration,
                    "best_iteration": self.stage_best_iteration,
                    "best_score": self.stage_best_score,
                    "stop_reason": reason,
                }
            )
            return
        self.stages.append(
            {
                "num_leaves": self.current_num_leaves,
                "learning_rate": EBM_INITIAL_LEARNING_RATE if self.current_num_leaves == 2 else self.configured_learning_rate,
                "start_iteration": self.stage_start_iteration,
                "end_iteration": end_iteration,
                "best_iteration": self.stage_best_iteration,
                "best_score": self.stage_best_score,
                "metric": self.metric_name,
                "dataset": "test",
                "stop_reason": reason,
            }
        )

    def finish(self, final_iteration: int) -> None:
        if self._closed:
            return
        self._close_stage(max(1, int(final_iteration or 1)), "iteration_cap")
        self._closed = True

    def progress_context(self) -> dict[str, Any]:
        return {
            "leaf_stage": self.current_num_leaves,
            "target_leaf_stage": self.target_num_leaves,
            "stage_start_iteration": self.stage_start_iteration,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "target_num_leaves": self.target_num_leaves,
            "initial_learning_rate": EBM_INITIAL_LEARNING_RATE,
            "configured_learning_rate": self.configured_learning_rate,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "higher_is_better": self.higher_is_better,
            "stages": self.stages,
        }


def write_dataframe_parquet(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("artifact_df", frame)
        con.execute(f"COPY artifact_df TO {sql_literal(str(path))} (FORMAT PARQUET)")
    finally:
        con.close()


def init_score_kind(selected_init_score: str) -> str:
    value = normalise_init_score_value(selected_init_score)
    if value.lower() == INIT_SCORE_NONE:
        return "none"
    if value.startswith("glm:"):
        return "glm_prediction"
    return "dataset_column"


def attach_glm_init_score(work_frame: Any, dataset: Dataset, selected_init_score: str, pl: Any) -> Any:
    glm_store = GlmModelStore(dataset.path)
    ref = glm_store.source_ref(selected_init_score)
    if ref is None or ref.source_kind != "predictions":
        raise ValueError(f"Choose a current fitted GLM prediction source for init_score: {selected_init_score}")
    prediction_path = glm_store.source_path(ref.model_id, "predictions")
    con = duckdb.connect(database=":memory:")
    try:
        con.register("score_rows", work_frame.select("__lucidum_row_id"))
        init_frame = con.execute(
            f"""
SELECT score_rows.__lucidum_row_id, prediction.glm_prediction AS __lucidum_init_score_prediction
FROM score_rows
LEFT JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)
"""
        ).pl()
    finally:
        con.close()
    return work_frame.join(init_frame, on="__lucidum_row_id", how="left", maintain_order="left")


def init_score_arrays(
    *,
    work_frame: Any,
    selected_init_score: str,
    selected_objective: str,
    dataset: Dataset,
    np: Any,
    pl: Any,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    value = normalise_init_score_value(selected_init_score)
    kind = init_score_kind(value)
    transform = init_score_transform(selected_objective)
    metadata: dict[str, Any] = {"value": value, "kind": kind, "transform": None if kind == "none" else transform}
    if kind == "none":
        return None, None, metadata
    if kind == "glm_prediction":
        work_frame = attach_glm_init_score(work_frame, dataset, value, pl)
        prediction_values = polars_numeric_array(work_frame, "__lucidum_init_score_prediction", pl)
        glm_store = GlmModelStore(dataset.path)
        ref = glm_store.source_ref(value)
        if ref is not None:
            metadata.update({"source_id": value, "model_id": ref.model_id})
    else:
        if value not in work_frame.columns:
            raise ValueError(f"Choose a valid numeric dataset column for init_score: {value}")
        prediction_values = polars_numeric_array(work_frame, value, pl)
        metadata.update({"column": value})
    prediction_array = np.asarray(prediction_values, dtype="float64")
    linear_array = init_score_to_linear(np, prediction_array, transform, value)
    metadata.update(
        {
            "artifact": "init_score.parquet",
            "space": "linear_predictor",
            "prediction_space": True,
            "replaces_offset": True,
            "boost_from_average": "ignored",
            "status": "current",
        }
    )
    return linear_array, prediction_array, metadata


def init_score_to_linear(np: Any, values: Any, transform: str, label: str) -> Any:
    array = np.asarray(values, dtype="float64")
    finite = np.isfinite(array)
    if transform == "log":
        valid = finite & (array > 0)
        if not bool(valid.all()):
            raise ValueError(f"init_score {label} must contain positive numeric values for all scored rows")
        return np.log(array)
    if transform == "logit":
        valid = finite & (array > 0) & (array < 1)
        if not bool(valid.all()):
            raise ValueError(f"init_score {label} must contain numeric values between 0 and 1 for all scored rows")
        return np.log(array / (1.0 - array))
    if not bool(finite.all()):
        raise ValueError(f"init_score {label} must contain finite numeric values for all scored rows")
    return array


def raw_score_to_prediction(np: Any, raw_values: Any, transform: str) -> Any:
    if transform == "log":
        return np.exp(raw_values)
    if transform == "logit":
        return 1.0 / (1.0 + np.exp(-raw_values))
    return raw_values


def predict_response_values(
    *,
    booster: Any,
    feature_data: Any,
    best_iteration: int,
    np: Any,
    use_supplied_init_score: bool,
    init_score_linear: Any | None,
    init_score_transform_name: str,
    use_offset_init_score: bool,
    log_offset: Any | None,
) -> Any:
    if use_supplied_init_score:
        if init_score_linear is None:
            raise ValueError("Supplied GBM init_score values are unavailable")
        raw_score = booster.predict(feature_data, raw_score=True, num_iteration=best_iteration)
        return raw_score_to_prediction(
            np,
            np.asarray(init_score_linear, dtype="float64") + np.asarray(raw_score, dtype="float64"),
            init_score_transform_name,
        )
    if use_offset_init_score:
        if log_offset is None:
            raise ValueError("GBM offset init_score values are unavailable")
        raw_score = booster.predict(feature_data, raw_score=True, num_iteration=best_iteration)
        return np.exp(np.asarray(log_offset, dtype="float64") + np.asarray(raw_score, dtype="float64"))
    return booster.predict(feature_data, num_iteration=best_iteration)


def init_score_dataframe(pl: Any, work_frame: Any, linear_values: Any, prediction_values: Any) -> Any:
    return pl.DataFrame(
        {
            "__lucidum_row_id": work_frame.get_column("__lucidum_row_id").cast(pl.Int64),
            "init_score": linear_values,
            "init_score_prediction": prediction_values,
        }
    )


def polars_numeric_array(frame: Any, column: str, pl: Any) -> Any:
    return (
        frame.get_column(column)
        .cast(pl.Float64, strict=False)
        .fill_null(float("nan"))
        .to_numpy()
    )


def polars_feature_frame(
    frame: Any,
    feature_names: list[str],
    categorical_features: list[str],
    pl: Any,
) -> tuple[Any, dict[str, list[str]]]:
    labels: dict[str, list[str]] = {}
    categorical_set = set(categorical_features)
    expressions: list[Any] = [
        pl.col(name).cast(pl.Float64, strict=False).alias(name)
        for name in feature_names
        if name not in categorical_set
    ]
    for name in categorical_features:
        values = (
            frame.get_column(name)
            .cast(pl.String, strict=False)
            .drop_nulls()
            .unique()
            .sort()
            .to_list()
        )
        categories = [str(value) for value in values]
        labels[name] = categories
        if categories:
            expressions.append(
                pl.col(name)
                .cast(pl.String, strict=False)
                .cast(pl.Enum(categories), strict=False)
                .to_physical()
                .cast(pl.Int32)
                .alias(name)
            )
        else:
            expressions.append(pl.lit(None, dtype=pl.Int32).alias(name))
    feature_frame = frame.select(feature_names)
    if expressions:
        feature_frame = feature_frame.with_columns(expressions)
    return feature_frame, labels


def arrow_rows(frame: Any, mask: Any, pl: Any) -> Any:
    return frame.filter(pl.Series("__lucidum_mask", mask)).to_arrow()


def training_projection_columns(
    *,
    response_col: str,
    offset_col: str | None,
    sample_column: str | None,
    feature_names: list[str],
    columns: dict[str, Any],
    init_score_col: str | None = None,
) -> list[str]:
    names = [response_col, offset_col, sample_column, init_score_col, *feature_names]
    projected: list[str] = []
    seen: set[str] = set()
    for name in names:
        text = str(name or "").strip()
        if not text or text in seen or text not in columns:
            continue
        projected.append(text)
        seen.add(text)
    return projected


def training_select_sql(
    relation_sql: str,
    projection_columns: list[str],
    where_sql: str = "",
    generated_sample_path: Path | None = None,
) -> str:
    base_select_parts = ["ROW_NUMBER() OVER () AS __lucidum_row_id"]
    base_select_parts.extend(quote_ident(name) for name in projection_columns)
    base_select_sql = ",\n      ".join(base_select_parts)
    select_parts = ["base.__lucidum_row_id"]
    select_parts.extend(f"base.{quote_ident(name)}" for name in projection_columns)
    join_sql = ""
    if generated_sample_path:
        select_parts.append(f"sample.{quote_ident(SAMPLE_COLUMN)} AS {quote_ident(SAMPLE_COLUMN)}")
        join_sql = f"\nLEFT JOIN {generated_sample_relation_sql(generated_sample_path)} sample USING (__lucidum_row_id)"
    select_sql = ",\n  ".join(select_parts)
    return f"""
SELECT
  {select_sql}
FROM (
  SELECT
      {base_select_sql}
  FROM {relation_sql}
) base{join_sql}
{where_sql}
ORDER BY base.__lucidum_row_id
"""


def train_model(
    dataset: Dataset,
    store: GbmModelStore,
    payload: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
    *,
    activate: bool = True,
    grid_search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dependency_started = time.perf_counter()

    def dependency_progress(stage: str) -> None:
        package = DEPENDENCY_IMPORT_PACKAGES.get(stage, stage.replace("_", " ").title())
        emit_preparing_progress(
            progress_callback,
            f"Preparing GBM: importing {package}...",
            0,
            stage=stage,
        )

    lgb, np, pd, pl = gbm_training_dependencies(
        dependency_progress=dependency_progress,
    )
    timings = {"dependency_seconds": elapsed_seconds(dependency_started)}
    started = time.perf_counter()
    emit_preparing_progress(progress_callback, "Preparing GBM: validating request...", 0, stage="validating_request")
    validation_started = time.perf_counter()
    validation = validate_request(dataset, payload, generated_sample_path=store.generated_sample_path)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    build_warnings = list(validation.warnings)
    timings["validation_seconds"] = elapsed_seconds(validation_started)
    create_interaction_group_models = bool_parameter(payload, "create_feature_interaction_group_models")

    emit_preparing_progress(progress_callback, "Preparing GBM: resolving parameters...", 0, stage="resolving_parameters")
    params = normalise_parameters(payload.get("parameters"))
    selected_init_score = normalise_init_score_value(params.pop("init_score", INIT_SCORE_NONE))
    training_mode = normalise_training_mode(payload.get("training_mode"))
    selected_objective = objective(params)
    params["objective"] = selected_objective
    num_boost_round = max(1, int(params.pop("num_iterations", 200) or 200))
    early_stopping_rounds = max(0, int(params.pop("early_stopping_rounds", 0) or 0))
    selected_metric = metric(params)
    params["metric"] = selected_metric
    configured_num_leaves = max(2, int(params.get("num_leaves", 31) or 31))
    configured_learning_rate = params.get("learning_rate", 0.05)
    stored_params = dict(params)
    stored_params["num_iterations"] = num_boost_round
    stored_params["early_stopping_rounds"] = early_stopping_rounds
    if training_mode == "ebm":
        params["num_leaves"] = 2
        params["learning_rate"] = EBM_INITIAL_LEARNING_RATE

    data_load_started = time.perf_counter()
    emit_preparing_progress(
        progress_callback,
        "Preparing GBM: waiting for dataset access...",
        0,
        stage="waiting_for_dataset",
    )
    with dataset.lock:
        emit_preparing_progress(progress_callback, "Preparing GBM: resolving selected features...", 0, stage="resolving_features")
        columns = dataset.column_map()
        response_col = selected_response_column(payload, columns)
        offset_col = selected_offset_column(payload, columns)
        features = normalise_features(payload.get("features"), columns)
        feature_names = [feature["name"] for feature in features]
        feature_grouping_map = normalise_feature_grouping_map(payload.get("feature_groupings"))
        selected_interaction_groupings = normalise_feature_interaction_groupings(payload.get("feature_interaction_groupings"))
        selected_interaction_features = normalise_feature_interaction_features(payload.get("feature_interaction_features"))
        selected_interaction_pairs = normalise_feature_interaction_pairs(payload.get("feature_interaction_pairs"))
        dataset_sample = dataset_sample_column(dataset)
        if not dataset_sample and payload.get("create_sample"):
            emit_preparing_progress(
                progress_callback,
                "Preparing GBM: creating generated SAMPLE split...",
                0,
                stage="creating_sample",
            )
            create_generated_sample(dataset, store.generated_sample_path)
        generated_sample_path = (
            store.generated_sample_path
            if not dataset_sample and generated_sample_is_current(dataset, store.generated_sample_path)
            else None
        )
        sample_column = SAMPLE_COLUMN if dataset_sample else None
        init_score_col = selected_init_score if selected_init_score in columns else None
        projection_columns = training_projection_columns(
            response_col=response_col,
            offset_col=offset_col,
            sample_column=sample_column,
            feature_names=feature_names,
            columns=columns,
            init_score_col=init_score_col,
        )
        where_sql = f"\nWHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        emit_preparing_progress(
            progress_callback,
            f"Preparing GBM: loading selected data from DuckDB ({len(projection_columns):,} columns)...",
            0,
            stage="loading_data",
            feature_count=len(feature_names),
            projection_column_count=len(projection_columns),
        )
        select_sql = training_select_sql(
            dataset.relation_sql(),
            projection_columns,
            where_sql,
            generated_sample_path=generated_sample_path,
        )
        score_frame = dataset.con.execute(select_sql).pl()
    timings["data_load_seconds"] = elapsed_seconds(data_load_started)

    if score_frame.is_empty():
        raise ValueError("No rows are available for GBM training")
    scored_rows = score_frame.height

    emit_preparing_progress(
        progress_callback,
        f"Preparing GBM: loaded {scored_rows:,} rows; creating model workspace...",
        0,
        stage="creating_workspace",
        scored_rows=scored_rows,
        feature_count=len(feature_names),
    )
    model_label = str(payload.get("label") or "GBM").strip() or "GBM"
    model_id = store.create_model_id(model_label)
    model_dir = store.create_model_dir(model_id)
    categorical_features = [feature["name"] for feature in features if feature["kind"] == "categorical"]
    monotone_constraints = [int(feature["monotonicity"]) for feature in features]
    if any(monotone_constraints):
        params["monotone_constraints"] = monotone_constraints
        stored_params["monotone_constraints"] = monotone_constraints
    interaction_group_constraints: list[dict[str, Any]] = []
    if selected_interaction_pairs:
        pair_feature_names = {
            feature_name
            for pair in selected_interaction_pairs
            for feature_name in (pair["left"], pair["right"])
        }
        interaction_feature_constraints = [
            feature_name
            for feature_name in selected_interaction_features
            if feature_name not in pair_feature_names
        ]
        interaction_groups = feature_interaction_constraint_groups(
            features,
            selected_interaction_groupings,
            feature_grouping_map,
            interaction_feature_constraints,
        )
        interaction_constraints = lightgbm_pair_interaction_constraints(feature_names, selected_interaction_pairs, interaction_groups)
        interaction_group_constraints = [
            {"grouping": str(group["grouping"]), "features": group["features"]}
            for group in interaction_groups
            if group.get("kind") != "feature"
        ]
        feature_interaction_constraints = (
            {
                "mode": "pairs",
                "pairs": selected_interaction_pairs,
                "uncovered_policy": "singletons",
                **({"groupings": [str(group["grouping"]) for group in interaction_group_constraints]} if interaction_group_constraints else {}),
                **({"groups": interaction_group_constraints} if interaction_group_constraints else {}),
                **({"features": interaction_feature_constraints} if interaction_feature_constraints else {}),
            }
            if interaction_constraints
            else None
        )
    else:
        interaction_groups = feature_interaction_constraint_groups(
            features,
            selected_interaction_groupings,
            feature_grouping_map,
            selected_interaction_features,
        )
        interaction_constraints = lightgbm_interaction_constraints(feature_names, interaction_groups)
        interaction_group_constraints = [
            {"grouping": str(group["grouping"]), "features": group["features"]}
            for group in interaction_groups
            if group.get("kind") != "feature"
        ]
        interaction_feature_constraints = [
            str(group["features"][0])
            for group in interaction_groups
            if group.get("kind") == "feature" and group.get("features")
        ]
        feature_interaction_constraints = (
            {
                "groupings": [str(group["grouping"]) for group in interaction_group_constraints],
                "features": interaction_feature_constraints,
                "groups": interaction_group_constraints,
            }
            if interaction_constraints
            else None
        )
    if interaction_constraints:
        params["interaction_constraints"] = interaction_constraints
        stored_params["interaction_constraints"] = interaction_constraints

    matrix_prep_started = time.perf_counter()
    emit_preparing_progress(
        progress_callback,
        "Preparing GBM: coercing response and denominator columns...",
        0,
        stage="coercing_columns",
        scored_rows=scored_rows,
    )
    numeric_expressions = [pl.col(response_col).cast(pl.Float64, strict=False).alias(response_col)]
    if offset_col:
        numeric_expressions.append(pl.col(offset_col).cast(pl.Float64, strict=False).alias(offset_col))
    work_frame = score_frame.with_columns(numeric_expressions)
    del score_frame
    response_values = polars_numeric_array(work_frame, response_col, pl)
    response_present = ~np.isnan(response_values)
    sample_mode = "none"
    sample_source = "none"
    sample_values = None
    if sample_column or generated_sample_path:
        sample_values = (
            work_frame.get_column(SAMPLE_COLUMN)
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
            .to_numpy()
        )
        sample_mode = SAMPLE_COLUMN
        sample_source = "dataset" if sample_column else "generated"
        train_mask = response_present & (sample_values == "training")
        test_mask = response_present & (sample_values == "test")
        validation_mask = response_present & (sample_values == "validation")
    else:
        train_mask = response_present.copy()
        test_mask = np.zeros(work_frame.height, dtype=bool)
        validation_mask = np.zeros(work_frame.height, dtype=bool)
    if not bool(train_mask.any()):
        raise ValueError("No training rows are available after validation")

    emit_preparing_progress(
        progress_callback,
        (
            "Preparing GBM: applying SAMPLE split "
            f"({int(train_mask.sum()):,} train, {int(test_mask.sum()):,} test, {int(validation_mask.sum()):,} validation)..."
        ),
        0,
        stage="splitting_sample",
        scored_rows=scored_rows,
        training_rows=int(train_mask.sum()),
        test_rows=int(test_mask.sum()),
        validation_rows=int(validation_mask.sum()),
    )
    emit_preparing_progress(
        progress_callback,
        f"Preparing GBM: encoding {len(categorical_features):,} categorical features...",
        0,
        stage="encoding_categoricals",
        feature_count=len(feature_names),
        categorical_feature_count=len(categorical_features),
    )
    feature_frame, categorical_labels = polars_feature_frame(work_frame, feature_names, categorical_features, pl)

    emit_preparing_progress(
        progress_callback,
        "Preparing GBM: preparing init scores...",
        0,
        stage="preparing_init_scores",
        scored_rows=scored_rows,
    )
    offset_values = polars_numeric_array(work_frame, offset_col, pl) if offset_col else None
    log_offset = np.log(offset_values) if offset_values is not None else None
    use_supplied_init_score = init_score_requested({"init_score": selected_init_score})
    init_score_linear, init_score_prediction, init_score_metadata = init_score_arrays(
        work_frame=work_frame,
        selected_init_score=selected_init_score,
        selected_objective=selected_objective,
        dataset=dataset,
        np=np,
        pl=pl,
    )
    use_offset_init_score = bool(not use_supplied_init_score and should_use_offset_init_score(params, offset_col))
    active_init_score = init_score_linear if use_supplied_init_score else log_offset if use_offset_init_score else None
    train_init = active_init_score[train_mask] if active_init_score is not None else None
    valid_init = active_init_score[test_mask] if active_init_score is not None and bool(test_mask.any()) else None
    timings["matrix_prep_seconds"] = elapsed_seconds(matrix_prep_started)

    emit_preparing_progress(
        progress_callback,
        "Preparing GBM: building LightGBM datasets...",
        0,
        stage="constructing_datasets",
        training_rows=int(train_mask.sum()),
    )
    dataset_construct_started = time.perf_counter()
    train_set = lgb.Dataset(
        arrow_rows(feature_frame, train_mask, pl),
        label=np.asarray(response_values[train_mask], dtype="float64"),
        feature_name=feature_names,
        categorical_feature=categorical_features,
        init_score=train_init,
        params=params,
        free_raw_data=True,
    )
    valid_sets = [train_set]
    valid_names = ["training"]
    if bool(test_mask.any()):
        valid_sets.append(
            lgb.Dataset(
                arrow_rows(feature_frame, test_mask, pl),
                label=np.asarray(response_values[test_mask], dtype="float64"),
                feature_name=feature_names,
                categorical_feature=categorical_features,
                init_score=valid_init,
                reference=train_set,
                params=params,
                free_raw_data=True,
            )
        )
        valid_names.append("test")
    for lightgbm_dataset in valid_sets:
        lightgbm_dataset.construct()
    timings["dataset_construct_seconds"] = elapsed_seconds(dataset_construct_started)

    evaluation_result: dict[str, dict[str, list[float | None]]] = {}
    ebm_controller: EbmStageController | None = None
    if training_mode == "ebm":
        ebm_controller = EbmStageController(
            lgb=lgb,
            metric_name=selected_metric,
            target_num_leaves=configured_num_leaves,
            configured_learning_rate=configured_learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            total_iterations=num_boost_round,
        )
    callbacks = [lgb.record_evaluation(evaluation_result)]
    callbacks.append(
        lightgbm_progress_callback(
            metric_name=selected_metric,
            total_iterations=num_boost_round,
            evaluation_result=evaluation_result,
            progress_callback=progress_callback,
            stage_provider=ebm_controller.progress_context if ebm_controller else None,
        )
    )
    if ebm_controller:
        callbacks.append(ebm_controller)
    elif len(valid_sets) > 1 and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    callbacks.append(lgb.log_evaluation(period=0))

    emit_preparing_progress(
        progress_callback,
        f"Preparing GBM: starting LightGBM training ({num_boost_round:,} trees)...",
        0,
        stage="fitting",
    )
    fit_started = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    timings["fit_seconds"] = elapsed_seconds(fit_started)
    final_iteration = int(booster.current_iteration() if hasattr(booster, "current_iteration") else num_boost_round)
    if ebm_controller:
        ebm_controller.finish(final_iteration)
    best_iteration = int(
        (ebm_controller.best_iteration if ebm_controller and ebm_controller.best_iteration else None)
        or getattr(booster, "best_iteration", 0)
        or num_boost_round
    )
    emit_progress(
        progress_callback,
        phase_progress_payload(
            phase="scoring",
            message=f"best iteration {best_iteration}, scoring...",
            percent=90,
            iteration=best_iteration,
            total_iterations=num_boost_round,
            metric=selected_metric,
            evaluation=evaluation_result,
        ),
    )
    score_started = time.perf_counter()
    full_feature_table = feature_frame.to_arrow()
    prediction = predict_response_values(
        booster=booster,
        feature_data=full_feature_table,
        best_iteration=best_iteration,
        np=np,
        use_supplied_init_score=use_supplied_init_score,
        init_score_linear=init_score_linear,
        init_score_transform_name=str(init_score_metadata.get("transform") or "identity"),
        use_offset_init_score=use_offset_init_score,
        log_offset=log_offset,
    )
    del full_feature_table
    prediction_data = {
        "__lucidum_row_id": work_frame.get_column("__lucidum_row_id").cast(pl.Int64),
        "gbm_prediction": prediction,
    }
    if offset_col:
        prediction_array = np.asarray(prediction, dtype="float64")
        offset_array = np.asarray(offset_values, dtype="float64")
        rate = np.full(prediction_array.shape, np.nan, dtype="float64")
        valid_rate = np.isfinite(prediction_array) & np.isfinite(offset_array) & (offset_array > 0)
        rate[valid_rate] = prediction_array[valid_rate] / offset_array[valid_rate]
        prediction_data["gbm_prediction_rate"] = rate
    predictions = pl.DataFrame(prediction_data)
    validation_warning = append_holdout_evaluation(
        lgb=lgb,
        np=np,
        response=response_values,
        prediction=prediction,
        validation_mask=validation_mask,
        parameters=params,
        evaluation_result=evaluation_result,
        best_iteration=best_iteration,
        metric_name=selected_metric,
    )
    if validation_warning:
        build_warnings.append(validation_warning)
    if offset_values is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            gini_actual = response_values / offset_values
            gini_prediction = prediction / offset_values
        gini_weight = offset_values
    else:
        gini_actual = response_values
        gini_prediction = prediction
        gini_weight = None
    gini_metrics, gini_warnings = split_gini_metrics(
        np,
        actual=gini_actual,
        prediction=gini_prediction,
        weight=gini_weight,
        sample_roles=sample_values,
    )
    build_warnings.extend(gini_warnings)
    saved_init_score_frame = (
        init_score_dataframe(pl, work_frame, init_score_linear, init_score_prediction)
        if use_supplied_init_score and init_score_linear is not None and init_score_prediction is not None
        else None
    )
    timings["score_seconds"] = elapsed_seconds(score_started)

    gain_values = booster.feature_importance(importance_type="gain")
    gains = {name: float(value or 0.0) for name, value in zip(feature_names, gain_values)}
    feature_config = [
        {
            **feature,
            "include": True,
            "monotonicity": display_monotonicity(feature.get("monotonicity")),
            "monotonicity_value": int(feature["monotonicity"]),
            "gain": round(gains.get(feature["name"], 0.0), 3),
        }
        for feature in features
    ]
    feature_config = sorted(feature_config, key=lambda item: (-float(item["gain"]), str(item["name"]).lower()))

    shap_mode = str(payload.get("shap_rows") or "0").strip().lower()
    shap_seed = shap_sampling_seed(params.get("seed"))
    shap_written_rows = 0
    shap_summary_rows: list[dict[str, Any]] = []
    shap_frame = None
    timings["shap_seconds"] = 0.0
    if shap_mode not in {"zero", "0", "none"}:
        shap_started = time.perf_counter()
        emit_progress(
            progress_callback,
            phase_progress_payload(
                phase="shap",
                message=f"best iteration {best_iteration}, SHAP values...",
                percent=96,
                iteration=best_iteration,
                total_iterations=num_boost_round,
                metric=selected_metric,
                evaluation=evaluation_result,
            ),
        )
        shap_frame, shap_summary = shap_dataframes(
            np=np,
            pl=pl,
            booster=booster,
            feature_frame=feature_frame,
            score_frame=work_frame,
            feature_names=feature_names,
            model_id=model_id,
            shap_mode=shap_mode,
            shap_seed=shap_seed,
            best_iteration=best_iteration,
            shap_interaction_groups=interaction_group_constraints,
        )
        shap_written_rows = shap_frame.height
        write_dataframe_parquet(shap_frame, store.artifact_path(model_id, "shap_long"))
        write_dataframe_parquet(shap_summary, store.artifact_path(model_id, "shap_summary"))
        shap_summary_rows = shap_summary.to_dicts()
        feature_config = feature_config_with_mean_abs_shap(feature_config, shap_summary_rows)
        timings["shap_seconds"] = elapsed_seconds(shap_started)

    emit_progress(
        progress_callback,
        phase_progress_payload(
            phase="artifacts",
            message=f"best iteration {best_iteration}, tree artifacts...",
            percent=98,
            iteration=best_iteration,
            total_iterations=num_boost_round,
            metric=selected_metric,
            evaluation=evaluation_result,
        ),
    )
    artifact_write_started = time.perf_counter()
    model_path = store.artifact_path(model_id, "model")
    booster.save_model(str(model_path), num_iteration=best_iteration)
    interaction_group_model_results: list[dict[str, Any]] = []
    timings["interaction_group_model_seconds"] = 0.0
    if create_interaction_group_models:
        interaction_group_model_started = time.perf_counter()
        interaction_group_model_results = create_and_verify_interaction_group_models(
            lgb=lgb,
            np=np,
            pl=pl,
            source_model=model_path,
            output_dir=model_dir,
            groups=interaction_group_constraints,
            feature_frame=feature_frame,
            feature_names=feature_names,
            shap_frame=shap_frame,
            shap_mode=shap_mode,
            shap_seed=shap_seed,
        )
        timings["interaction_group_model_seconds"] = elapsed_seconds(interaction_group_model_started)
    write_dataframe_parquet(predictions, store.artifact_path(model_id, "predictions"))
    if saved_init_score_frame is not None:
        write_dataframe_parquet(saved_init_score_frame, store.artifact_path(model_id, "init_score"))
    write_dataframe_parquet(evaluation_dataframe(pd, evaluation_result), store.artifact_path(model_id, "evaluation"))
    write_dataframe_parquet(tree_dataframe(pd, booster, categorical_labels=categorical_labels), store.artifact_path(model_id, "tree_table"))

    ebm_metadata = ebm_controller.metadata() if ebm_controller else None
    feature_scenario = normalise_feature_scenario(payload.get("feature_scenario"))
    manifest = {
        "model_id": model_id,
        "label": model_label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_mode": training_mode,
        "response_column": response_col,
        "offset_column": offset_col,
        "best_iteration": best_iteration,
        "training_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "scored_rows": scored_rows,
        "sample_column": sample_mode if sample_mode != "none" else None,
        "sample_source": sample_source,
        "shap_rows": shap_written_rows,
        **gini_metrics,
        "timings": {},
        "warnings": build_warnings,
    }
    if feature_scenario:
        manifest["feature_scenario"] = feature_scenario
    if feature_interaction_constraints:
        manifest["feature_interaction_constraints"] = feature_interaction_constraints
    manifest["feature_interaction_group_models"] = {
        "enabled": create_interaction_group_models,
        "error_metric": "max_absolute_error",
        "groups": interaction_group_model_results,
    }
    if ebm_metadata:
        manifest["ebm"] = ebm_metadata
    if grid_search:
        manifest["grid_search"] = grid_search
    manifest["init_score"] = init_score_metadata
    feature_config_frame = pd.DataFrame(feature_config)
    feature_config_columns = ["name", "kind", "include", "monotonicity", "monotonicity_value", "gain", "mean_abs_shap"]
    for column in feature_config_columns:
        if column not in feature_config_frame:
            feature_config_frame[column] = None
    store.write_json(store.artifact_path(model_id, "features"), feature_names)
    write_dataframe_parquet(feature_config_frame[feature_config_columns], store.artifact_path(model_id, "feature_config"))
    store.write_json(store.artifact_path(model_id, "parameters"), stored_params)
    timings["artifact_write_seconds"] = elapsed_seconds(artifact_write_started)
    timings["training_seconds"] = elapsed_seconds(started)
    manifest["timings"] = timings
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)
    if activate:
        result = store.activate_model(model_id)
    else:
        result = store.model_list_item(model_dir, manifest, store.active_model_id())
    emit_progress(
        progress_callback,
        phase_progress_payload(
            phase="succeeded",
            message="GBM training complete",
            percent=100,
            iteration=best_iteration,
            total_iterations=num_boost_round,
            metric=selected_metric,
            evaluation=evaluation_result,
        ),
    )
    return result


def normalise_feature_scenario(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    features: list[str] = []
    seen: set[str] = set()
    raw_features = raw.get("features")
    if isinstance(raw_features, list):
        for item in raw_features:
            feature = str(item or "").strip()
            if feature and feature not in seen:
                features.append(feature)
                seen.add(feature)
    return {"name": name, "features": features}


def lightgbm_interaction_constraints(feature_names: list[str], groups: list[dict[str, Any]]) -> list[list[int]]:
    if not feature_names or not groups:
        return []
    feature_indexes = {name: index for index, name in enumerate(feature_names)}
    constrained_indexes: set[int] = set()
    constraints: list[list[int]] = []
    for group in groups:
        indexes: list[int] = []
        seen: set[int] = set()
        for feature in group.get("features", []):
            index = feature_indexes.get(str(feature))
            if index is not None and index not in seen:
                indexes.append(index)
                seen.add(index)
        if indexes:
            constraints.append(indexes)
            constrained_indexes.update(indexes)
    if not constraints:
        return []
    remainder = [index for index in range(len(feature_names)) if index not in constrained_indexes]
    if remainder:
        constraints.append(remainder)
    return constraints


def lightgbm_pair_interaction_constraints(
    feature_names: list[str],
    pairs: list[dict[str, str]],
    groups: list[dict[str, Any]] | None = None,
) -> list[list[int]]:
    if not feature_names or not pairs:
        return []
    feature_indexes = {name: index for index, name in enumerate(feature_names)}
    constraints: list[list[int]] = []
    constrained_indexes: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for pair in pairs:
        left_index = feature_indexes.get(str(pair.get("left") or ""))
        right_index = feature_indexes.get(str(pair.get("right") or ""))
        if left_index is None or right_index is None or left_index == right_index:
            continue
        key = tuple(sorted((left_index, right_index)))
        if key in seen_pairs:
            continue
        constraints.append([left_index, right_index])
        constrained_indexes.update(key)
        seen_pairs.add(key)
    seen_groups: set[tuple[int, ...]] = set()
    for group in groups or []:
        indexes: list[int] = []
        seen_indexes: set[int] = set()
        for feature in group.get("features", []):
            index = feature_indexes.get(str(feature))
            if index is not None and index not in seen_indexes:
                indexes.append(index)
                seen_indexes.add(index)
        if not indexes:
            continue
        key = tuple(indexes)
        if key in seen_groups:
            continue
        constraints.append(indexes)
        constrained_indexes.update(indexes)
        seen_groups.add(key)
    if not constraints:
        return []
    for index in range(len(feature_names)):
        if index not in constrained_indexes:
            constraints.append([index])
    return constraints


def emit_progress(progress_callback: ProgressCallback | None, progress: dict[str, Any]) -> None:
    if progress_callback:
        progress_callback(progress)


def emit_preparing_progress(
    progress_callback: ProgressCallback | None,
    message: str,
    percent: float | int | None,
    stage: str | None = None,
    **extra: Any,
) -> None:
    payload = phase_progress_payload(
        phase="preparing",
        stage=stage,
        message=message,
        percent=percent,
    )
    payload.update({key: value for key, value in extra.items() if value is not None})
    emit_progress(progress_callback, payload)


def phase_progress_payload(
    *,
    phase: str,
    stage: str | None = None,
    message: str,
    percent: float | int | None,
    iteration: int | None = None,
    total_iterations: int | None = None,
    metric: str | None = None,
    evaluation: dict[str, dict[str, list[float]]] | None = None,
) -> dict[str, Any]:
    safe_evaluation = json_safe_evaluation(evaluation or {})
    return {
        "phase": phase,
        "stage": stage or phase,
        "message": message,
        "iteration": iteration,
        "total_iterations": total_iterations,
        "percent": json_safe_number(percent),
        "metric": metric,
        "latest": latest_from_evaluation(safe_evaluation, metric),
        "evaluation": safe_evaluation,
    }


def lightgbm_progress_callback(
    *,
    metric_name: str,
    total_iterations: int,
    evaluation_result: dict[str, dict[str, list[float]]],
    progress_callback: ProgressCallback | None,
    stage_provider: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    def callback(env: Any) -> None:
        emit_progress(
            progress_callback,
            lightgbm_progress_payload(
                env,
                metric_name=metric_name,
                total_iterations=total_iterations,
                evaluation_result=evaluation_result,
                stage=stage_provider() if stage_provider else None,
            ),
        )

    callback.order = 30
    callback.before_iteration = False
    return callback


def append_holdout_evaluation(
    *,
    lgb: Any,
    np: Any,
    response: Any,
    prediction: Any,
    validation_mask: Any,
    parameters: dict[str, Any],
    evaluation_result: dict[str, dict[str, list[float | None]]],
    best_iteration: int,
    metric_name: str,
) -> str | None:
    if not bool(validation_mask.any()):
        return None
    metric_booster = None
    try:
        actual_values = np.asarray(response[validation_mask], dtype="float64")
        prediction_values = np.asarray(prediction[validation_mask], dtype="float64")
        if actual_values.shape != prediction_values.shape or actual_values.size == 0:
            raise ValueError("Validation actuals and predictions must have the same non-zero length")
        if not bool(np.isfinite(actual_values).all() and np.isfinite(prediction_values).all()):
            raise ValueError("Validation actuals and predictions must be finite")

        raw_prediction = prediction_to_raw_score(
            np,
            prediction_values,
            str(parameters.get("objective") or "regression").strip().lower(),
            parameters,
        )
        validation_set = lgb.Dataset(
            np.arange(actual_values.size, dtype="float64").reshape(-1, 1),
            label=actual_values,
            init_score=raw_prediction,
            free_raw_data=False,
            params={"min_data_in_leaf": 1, "min_data_in_bin": 1, "verbosity": -1},
        )
        metric_booster = lgb.Booster(
            params=metric_evaluation_parameters(parameters),
            train_set=validation_set,
        )
        results = metric_booster.eval_train()
        requested_metric = str(metric_name or "").strip().lower()
        selected = next(
            (
                item
                for item in results or []
                if len(item) >= 3 and str(item[1]).strip().lower() == requested_metric
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"LightGBM returned no {metric_name} result")
        value = json_safe_number(selected[2])
        if value is None:
            raise ValueError(f"LightGBM returned a non-finite {metric_name} result")
        values: list[float | None] = [None] * max(0, int(best_iteration) - 1)
        values.append(value)
        evaluation_result.setdefault("validation", {})[str(selected[1])] = values
        return None
    except Exception as exc:
        return f"Validation {metric_name} metric could not be calculated: {exc}"
    finally:
        free_dataset = getattr(metric_booster, "free_dataset", None)
        if callable(free_dataset):
            free_dataset()


def prediction_to_raw_score(
    np: Any,
    prediction: Any,
    objective_name: str,
    parameters: dict[str, Any],
) -> Any:
    """Undo LightGBM's objective transform for metric-only evaluation."""

    values = np.asarray(prediction, dtype="float64")
    if objective_name in {"poisson", "gamma", "tweedie"}:
        if not bool((values > 0).all()):
            raise ValueError(f"{objective_name} predictions must be positive")
        return np.log(values)
    if objective_name in {"binary", "cross_entropy"}:
        if not bool(((values > 0) & (values < 1)).all()):
            raise ValueError(f"{objective_name} predictions must be between zero and one")
        sigmoid = float(parameters.get("sigmoid", 1.0) or 1.0)
        return np.log(values / (1.0 - values)) / sigmoid
    if objective_name == "cross_entropy_lambda":
        if not bool((values > 0).all()):
            raise ValueError("cross_entropy_lambda predictions must be positive")
        raw = np.empty_like(values)
        large = values > 20
        raw[large] = values[large] + np.log1p(-np.exp(-values[large]))
        raw[~large] = np.log(np.expm1(values[~large]))
        return raw
    return values


def metric_evaluation_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Keep only settings which change LightGBM's metric calculation."""

    names = {
        "objective",
        "metric",
        "sigmoid",
        "alpha",
        "fair_c",
        "tweedie_variance_power",
    }
    return {
        **{name: value for name, value in parameters.items() if name in names},
        "boost_from_average": False,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1,
        "verbosity": -1,
    }


def lightgbm_progress_payload(
    env: Any,
    *,
    metric_name: str,
    total_iterations: int,
    evaluation_result: dict[str, dict[str, list[float]]],
    stage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    begin_iteration = int(getattr(env, "begin_iteration", 0) or 0)
    current_iteration = int(getattr(env, "iteration", 0) or 0) - begin_iteration + 1
    total = max(1, int(total_iterations or 1))
    latest = latest_from_evaluation_result_list(getattr(env, "evaluation_result_list", None))
    preferred = preferred_progress_metric(latest, metric_name)
    message = training_progress_message(current_iteration, total, preferred, metric_name, stage)
    payload = {
        "phase": "training",
        "stage": "fitting",
        "message": message,
        "iteration": current_iteration,
        "total_iterations": total,
        "percent": round(min(90.0, max(0.0, 90.0 * current_iteration / total)), 1),
        "metric": metric_name,
        "latest": latest,
        "evaluation": json_safe_evaluation(evaluation_result),
    }
    if stage:
        payload.update(stage)
    return payload


def latest_from_evaluation_result_list(evaluation_result_list: Any) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for item in evaluation_result_list or []:
        if len(item) < 3:
            continue
        latest.append(
            {
                "dataset": str(item[0]),
                "metric": str(item[1]),
                "value": json_safe_number(item[2]),
            }
        )
    latest.sort(key=lambda item: progress_metric_sort_key(item, None))
    return latest


def latest_from_evaluation(evaluation: dict[str, dict[str, list[Any]]], metric_name: str | None = None) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for dataset_name, metrics in evaluation.items():
        for name, values in metrics.items():
            value = last_json_safe_number(values)
            if value is not None:
                latest.append({"dataset": dataset_name, "metric": name, "value": value})
    latest.sort(key=lambda item: progress_metric_sort_key(item, metric_name))
    return latest


def preferred_progress_metric(latest: list[dict[str, Any]], metric_name: str | None) -> dict[str, Any] | None:
    if not latest:
        return None
    return sorted(latest, key=lambda item: progress_metric_sort_key(item, metric_name))[0]


def progress_metric_sort_key(item: dict[str, Any], metric_name: str | None) -> tuple[int, int, str]:
    dataset = str(item.get("dataset") or "").lower()
    metric = str(item.get("metric") or "")
    dataset_rank = 0 if dataset == "test" else 1 if dataset in {"validation", "valid"} else 2 if dataset in {"training", "train"} else 3
    metric_rank = 0 if metric_name and metric == metric_name else 1
    return dataset_rank, metric_rank, metric


def training_progress_message(iteration: int, total: int, preferred: dict[str, Any] | None, metric_name: str, stage: dict[str, Any] | None = None) -> str:
    prefix = f"training, leaves {stage['leaf_stage']}, tree {iteration}/{total}" if stage and stage.get("leaf_stage") else f"training, tree {iteration}/{total}"
    if not preferred:
        return prefix
    dataset = "train" if str(preferred.get("dataset") or "").lower() == "training" else str(preferred.get("dataset") or "metric")
    metric = str(preferred.get("metric") or metric_name)
    value = format_progress_value(preferred.get("value"))
    return f"{prefix}, {dataset} {metric} {value}".rstrip()


def json_safe_evaluation(evaluation: dict[str, dict[str, list[Any]]]) -> dict[str, dict[str, list[float | None]]]:
    safe: dict[str, dict[str, list[float | None]]] = {}
    for dataset_name, metrics in evaluation.items():
        safe_metrics: dict[str, list[float | None]] = {}
        for metric_name, values in metrics.items():
            safe_metrics[str(metric_name)] = [json_safe_number(value) for value in values]
        safe[str(dataset_name)] = safe_metrics
    return safe


def json_safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def last_json_safe_number(values: list[Any]) -> float | None:
    for value in reversed(values):
        number = json_safe_number(value)
        if number is not None:
            return number
    return None


def format_progress_value(value: Any) -> str:
    number = json_safe_number(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def evaluation_dataframe(
    pd: Any,
    evaluation_result: dict[str, dict[str, list[float | None]]],
) -> Any:
    rows: list[dict[str, Any]] = []
    for dataset_name, metrics in evaluation_result.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                if json_safe_number(value) is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "iteration": iteration,
                        "value": canonical_gbm_metric(value),
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "metric", "iteration", "value"])


def canonical_gbm_metric(value: Any) -> float | None:
    """Remove immaterial platform-level reduction noise from saved metrics."""

    number = json_safe_number(value)
    return float(f"{number:.15g}") if number is not None else None


def should_use_offset_init_score(params: dict[str, Any], offset_col: str | None) -> bool:
    return bool(offset_col and uses_log_offset(params))


def tree_dataframe(pd: Any, booster: Any, categorical_labels: dict[str, list[str]] | None = None) -> Any:
    try:
        frame = booster.trees_to_dataframe()
    except Exception:
        frame = pd.DataFrame(
            columns=[
                "tree_index",
                "node_depth",
                "node_index",
                "left_child",
                "right_child",
                "parent_index",
                "split_feature",
                "split_gain",
                "threshold",
                "threshold_label",
                "decision_type",
                "missing_direction",
                "missing_type",
                "value",
                "weight",
                "count",
            ]
        )
    return tree_dataframe_with_threshold_labels(pd, frame, categorical_labels or {})


def categorical_feature_labels(feature_frame: Any, categorical_features: list[str]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for name in categorical_features:
        if name not in feature_frame:
            continue
        categories = getattr(getattr(feature_frame[name], "cat", None), "categories", None)
        if categories is not None:
            labels[name] = [str(value) for value in categories]
    return labels


def tree_dataframe_with_threshold_labels(pd: Any, frame: Any, categorical_labels: dict[str, list[str]]) -> Any:
    if frame is None:
        frame = pd.DataFrame()
    frame = frame.copy()
    if "threshold_label" not in frame.columns:
        frame["threshold_label"] = None
    if not categorical_labels or frame.empty:
        return frame
    required = {"split_feature", "decision_type", "threshold"}
    if not required.issubset(set(frame.columns)):
        return frame
    labels: list[str | None] = []
    for _, row in frame.iterrows():
        feature = str(row.get("split_feature") or "").strip()
        decision_type = str(row.get("decision_type") or "").strip()
        threshold = row.get("threshold")
        if decision_type == "==" and feature in categorical_labels and not bool(pd.isna(threshold)):
            labels.append(decode_categorical_threshold(threshold, categorical_labels[feature]))
        else:
            labels.append(None)
    frame["threshold_label"] = labels
    return frame


def decode_categorical_threshold(value: Any, categories: list[str]) -> str | None:
    labels: list[str] = []
    for part in str(value).split("||"):
        try:
            index = int(float(part))
        except ValueError:
            continue
        labels.append(categories[index] if 0 <= index < len(categories) else str(index))
    return " / ".join(labels) if labels else None


def shap_row_limit(mode: str, row_count: int) -> int:
    if mode in {"all", "*"}:
        return row_count
    if mode in {"10k", "10000", "10,000"}:
        return min(10000, row_count)
    if mode in {"100k", "100000", "100,000"}:
        return min(100000, row_count)
    try:
        return min(max(0, int(mode)), row_count)
    except ValueError:
        return 0


def shap_sampling_seed(raw: Any) -> int:
    try:
        seed = int(raw)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_SHAP_SAMPLE_SEED
    return seed if seed >= 0 else DEFAULT_SHAP_SAMPLE_SEED


def shap_sample_positions(np: Any, *, mode: str, row_count: int, seed: int) -> Any:
    limit = shap_row_limit(mode, row_count)
    if limit <= 0:
        return np.array([], dtype="int64")
    if limit >= row_count:
        return np.arange(row_count, dtype="int64")
    return np.sort(np.random.default_rng(seed).choice(row_count, size=limit, replace=False))


def shap_dataframes(
    *,
    np: Any,
    pl: Any,
    booster: Any,
    feature_frame: Any,
    score_frame: Any,
    feature_names: list[str],
    model_id: str,
    shap_mode: str,
    shap_seed: int,
    best_iteration: int,
    shap_interaction_groups: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any]:
    group_columns = shap_interaction_group_columns(shap_interaction_groups or [], feature_names)
    positions = shap_sample_positions(np, mode=shap_mode, row_count=feature_frame.height, seed=shap_seed)
    shap_schema = {
        "__lucidum_row_id": pl.Int64,
        **{feature_name: pl.Float64 for feature_name in feature_names},
        **{group["name"]: pl.Float64 for group in group_columns},
    }
    summary_schema = {
        "feature": pl.String,
        "mean_abs_shap": pl.Float64,
        "mean_shap": pl.Float64,
        "row_count": pl.Int64,
    }
    if len(positions) == 0:
        return pl.DataFrame(schema=shap_schema), pl.DataFrame(schema=summary_schema)

    sampled = feature_frame.gather(positions)
    sampled_scores = score_frame.gather(positions)
    contributions = booster.predict(sampled.to_arrow(), pred_contrib=True, num_iteration=best_iteration)
    if contributions.ndim == 3:
        contributions = contributions[:, :, 0]
    feature_contribs = contributions[:, : len(feature_names)]

    shap_frame = pl.DataFrame(
        {
            "__lucidum_row_id": sampled_scores.get_column("__lucidum_row_id").cast(pl.Int64),
            **{
                feature_name: feature_contribs[:, feature_index].astype(float)
                for feature_index, feature_name in enumerate(feature_names)
            },
        }
    )
    for group in group_columns:
        shap_frame = shap_frame.with_columns(
            pl.sum_horizontal([pl.col(name) for name in group["features"]]).alias(group["name"])
        )
    if shap_frame.is_empty():
        summary = pl.DataFrame(schema=summary_schema)
    else:
        summary = pl.DataFrame(
            [
                {
                    "feature": feature_name,
                    "mean_abs_shap": float(
                        np.mean(np.abs(feature_contribs[:, feature_index]))
                    ),
                    "mean_shap": float(np.mean(feature_contribs[:, feature_index])),
                    "row_count": int(feature_contribs.shape[0]),
                }
                for feature_index, feature_name in enumerate(feature_names)
            ],
            schema=summary_schema,
        )
        summary = summary.sort(
            ["mean_abs_shap", "feature"],
            descending=[True, False],
        )
    return shap_frame, summary


def shap_interaction_group_columns(groups: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    feature_set = set(feature_names)
    used_names = {"__lucidum_row_id", *feature_names}
    columns: list[dict[str, Any]] = []
    for group in groups:
        grouping = str(group.get("grouping") or "").strip()
        raw_features = group.get("features")
        if not grouping or not isinstance(raw_features, list):
            continue
        features: list[str] = []
        seen_features: set[str] = set()
        for raw_feature in raw_features:
            feature = str(raw_feature or "").strip()
            if feature and feature in feature_set and feature not in seen_features:
                features.append(feature)
                seen_features.add(feature)
        if not features:
            continue
        column_name = unique_shap_group_column_name(f"{grouping}_INTERACTION_GROUP", used_names)
        columns.append({"name": column_name, "grouping": grouping, "features": features})
    return columns


def unique_shap_group_column_name(base_name: str, used_names: set[str]) -> str:
    base = str(base_name or "INTERACTION_GROUP").strip() or "INTERACTION_GROUP"
    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def create_and_verify_interaction_group_models(
    *,
    lgb: Any,
    np: Any,
    pl: Any,
    source_model: Path,
    output_dir: Path,
    groups: list[dict[str, Any]],
    feature_frame: Any,
    feature_names: list[str],
    shap_frame: Any,
    shap_mode: str,
    shap_seed: int,
) -> list[dict[str, Any]]:
    """Create selected interaction-group models and verify them against grouped SHAP."""

    if shap_frame is None or shap_frame.is_empty():
        raise ValueError("Constraint group model verification requires saved SHAP rows")
    group_columns = {
        str(group["grouping"]): group
        for group in shap_interaction_group_columns(groups, feature_names)
    }
    positions = shap_sample_positions(
        np,
        mode=shap_mode,
        row_count=feature_frame.height,
        seed=shap_seed,
    )
    if len(positions) != shap_frame.height:
        raise ValueError("Constraint group model verification rows do not match saved SHAP rows")
    sampled_features = feature_frame.gather(positions)
    results: list[dict[str, Any]] = []
    used_filenames: list[str] = []
    for group in groups:
        grouping = str(group.get("grouping") or "").strip()
        group_column = group_columns.get(grouping)
        if not grouping or group_column is None:
            raise ValueError(f"Saved SHAP values are missing for constraint group: {grouping or '<blank>'}")
        filename = interaction_group_model_filename(grouping, used_filenames)
        output_model = output_dir / filename
        try:
            extract_lightgbm_interaction_group(
                source_model,
                group_column["features"],
                output_model,
            )
        except NoInteractionGroupTreesError:
            results.append(
                {
                    "grouping": grouping,
                    "status": "no_trees",
                    "artifact": None,
                    "tree_count": 0,
                    "verified_rows": 0,
                    "max_absolute_error": None,
                }
            )
            continue

        extracted = lgb.Booster(model_file=str(output_model))
        extracted_features = list(extracted.feature_name())
        if set(extracted_features) != set(group_column["features"]):
            raise ValueError(f"Extracted constraint group model features are inconsistent for: {grouping}")
        raw_prediction = np.asarray(
            extracted.predict(
                sampled_features.select(extracted_features).to_arrow(),
                raw_score=True,
            ),
            dtype="float64",
        )
        if raw_prediction.ndim == 2:
            raw_prediction = raw_prediction[:, 0]
        if raw_prediction.ndim != 1 or raw_prediction.shape[0] != shap_frame.height:
            raise ValueError(f"Extracted constraint group model prediction shape is invalid for: {grouping}")
        expected = np.asarray(shap_frame.get_column(group_column["name"]).to_numpy(), dtype="float64")
        if not bool(np.isfinite(raw_prediction).all()) or not bool(np.isfinite(expected).all()):
            raise ValueError(f"Extracted constraint group model verification contains non-finite values for: {grouping}")
        max_absolute_error = float(np.max(np.abs(raw_prediction - expected)))
        if not math.isfinite(max_absolute_error):
            raise ValueError(f"Extracted constraint group model verification error is invalid for: {grouping}")
        results.append(
            {
                "grouping": grouping,
                "status": "verified",
                "artifact": filename,
                "tree_count": int(extracted.num_trees()),
                "verified_rows": int(shap_frame.height),
                "max_absolute_error": max_absolute_error,
            }
        )
        used_filenames.append(filename)
    return results


def feature_config_with_mean_abs_shap(
    feature_config: list[dict[str, Any]],
    shap_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mean_abs_shap = {
        str(row.get("feature")): number
        for row in shap_summary_rows
        if row.get("feature") and (number := json_safe_number(row.get("mean_abs_shap"))) is not None
    }
    if not mean_abs_shap:
        return feature_config
    return [
        {**feature, "mean_abs_shap": mean_abs_shap[feature["name"]]}
        if feature.get("name") in mean_abs_shap
        else feature
        for feature in feature_config
    ]


__all__ = [
    "MissingGbmDependency",
    "gbm_dependencies",
    "gbm_training_dependencies",
    "lightgbm_progress_payload",
    "lightgbm_interaction_constraints",
    "lightgbm_pair_interaction_constraints",
    "create_and_verify_interaction_group_models",
    "feature_config_with_mean_abs_shap",
    "normalise_feature_scenario",
    "polars_feature_frame",
    "predict_response_values",
    "should_use_offset_init_score",
    "shap_interaction_group_columns",
    "train_model",
    "training_projection_columns",
    "training_select_sql",
    "write_dataframe_parquet",
]
