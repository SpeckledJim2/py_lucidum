from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal
from py_lucidum.tools.glm.store import GlmModelStore

from .sample import (
    SAMPLE_COLUMN,
    create_generated_sample,
    dataset_sample_column,
    generated_sample_is_current,
    generated_sample_relation_sql,
)
from .store import GbmModelStore, best_metrics_from_evaluation
from .validation import (
    INIT_SCORE_NONE,
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
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
EBM_INITIAL_LEARNING_RATE = 0.3
DEFAULT_SHAP_SAMPLE_SEED = 2026


class MissingGbmDependency(RuntimeError):
    def __init__(self, missing: str, hint: str | None = None):
        message = f"Install GBM dependencies with `pip install 'py-lucidum[gbm]'` to train LightGBM models. Missing: {missing}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)
        self.missing = missing
        self.hint = hint


def gbm_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
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
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        pd = None
        missing.append("pandas")
    if missing:
        raise MissingGbmDependency(", ".join(missing), hint)
    return lgb, np, pd


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


def attach_glm_init_score(work_frame: Any, dataset: Dataset, selected_init_score: str) -> Any:
    glm_store = GlmModelStore(dataset.path)
    ref = glm_store.source_ref(selected_init_score)
    if ref is None or ref.source_kind != "predictions":
        raise ValueError(f"Choose a current fitted GLM prediction source for init_score: {selected_init_score}")
    prediction_path = glm_store.source_path(ref.model_id, "predictions")
    con = duckdb.connect(database=":memory:")
    try:
        con.register("score_rows", work_frame[["__lucidum_row_id"]])
        init_frame = con.execute(
            f"""
SELECT score_rows.__lucidum_row_id, prediction.glm_prediction AS __lucidum_init_score_prediction
FROM score_rows
LEFT JOIN read_parquet({sql_literal(str(prediction_path))}) prediction USING (__lucidum_row_id)
"""
        ).fetch_df()
    finally:
        con.close()
    return work_frame.merge(init_frame, on="__lucidum_row_id", how="left", sort=False)


def init_score_arrays(
    *,
    work_frame: Any,
    selected_init_score: str,
    selected_objective: str,
    dataset: Dataset,
    np: Any,
    pd: Any,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    value = normalise_init_score_value(selected_init_score)
    kind = init_score_kind(value)
    transform = init_score_transform(selected_objective)
    metadata: dict[str, Any] = {"value": value, "kind": kind, "transform": None if kind == "none" else transform}
    if kind == "none":
        return None, None, metadata
    if kind == "glm_prediction":
        work_frame = attach_glm_init_score(work_frame, dataset, value)
        prediction_values = pd.to_numeric(work_frame["__lucidum_init_score_prediction"], errors="coerce")
        glm_store = GlmModelStore(dataset.path)
        ref = glm_store.source_ref(value)
        if ref is not None:
            metadata.update({"source_id": value, "model_id": ref.model_id})
    else:
        if value not in work_frame:
            raise ValueError(f"Choose a valid numeric dataset column for init_score: {value}")
        prediction_values = pd.to_numeric(work_frame[value], errors="coerce")
        metadata.update({"column": value})
    prediction_array = prediction_values.to_numpy(dtype="float64")
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


def init_score_dataframe(pd: Any, work_frame: Any, linear_values: Any, prediction_values: Any) -> Any:
    return pd.DataFrame(
        {
            "__lucidum_row_id": work_frame["__lucidum_row_id"].astype("int64").to_numpy(),
            "init_score": linear_values,
            "init_score_prediction": prediction_values,
        }
    )


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
    lgb, np, pd = gbm_dependencies()
    started = time.perf_counter()
    emit_progress(progress_callback, {"phase": "preparing", "message": "preparing GBM...", "percent": 0})
    validation = validate_request(dataset, payload, generated_sample_path=store.generated_sample_path)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

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
    stored_params["init_score"] = selected_init_score
    stored_params["num_iterations"] = num_boost_round
    stored_params["early_stopping_rounds"] = early_stopping_rounds
    stored_params["training_mode"] = training_mode
    if training_mode == "ebm":
        params["num_leaves"] = 2
        params["learning_rate"] = EBM_INITIAL_LEARNING_RATE

    with dataset.lock:
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
            create_generated_sample(dataset, store.generated_sample_path)
        generated_sample_path = (
            store.generated_sample_path
            if not dataset_sample and generated_sample_is_current(dataset, store.generated_sample_path)
            else None
        )
        sample_column = SAMPLE_COLUMN if dataset_sample else None
        init_score_col = selected_init_score if selected_init_score in columns else None
        source_columns = list(columns)
        projection_columns = training_projection_columns(
            response_col=response_col,
            offset_col=offset_col,
            sample_column=sample_column,
            feature_names=feature_names,
            columns=columns,
            init_score_col=init_score_col,
        )
        where_sql = f"\nWHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        select_sql = training_select_sql(
            dataset.relation_sql(),
            projection_columns,
            where_sql,
            generated_sample_path=generated_sample_path,
        )
        score_frame = dataset.con.execute(select_sql).fetch_df()

    if score_frame.empty:
        raise ValueError("No rows are available for GBM training")

    model_label = str(payload.get("label") or "GBM").strip() or "GBM"
    model_id = store.create_model_id(model_label)
    model_dir = store.create_model_dir(model_id)
    categorical_features = [feature["name"] for feature in features if feature["kind"] == "categorical"]
    monotone_constraints = [int(feature["monotonicity"]) for feature in features]
    if any(monotone_constraints):
        params["monotone_constraints"] = monotone_constraints
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

    work_frame = score_frame.copy()
    work_frame[response_col] = pd.to_numeric(work_frame[response_col], errors="coerce")
    if offset_col:
        work_frame[offset_col] = pd.to_numeric(work_frame[offset_col], errors="coerce")
    response_present = work_frame[response_col].notna()
    sample_mode = "none"
    sample_source = "none"
    if sample_column or generated_sample_path:
        sample_values = work_frame[SAMPLE_COLUMN].astype(str).str.strip().str.lower()
        sample_mode = SAMPLE_COLUMN
        sample_source = "dataset" if sample_column else "generated"
    else:
        sample_values = pd.Series(["training"] * len(work_frame), index=work_frame.index)

    train_mask = response_present & (sample_values == "training")
    test_mask = response_present & (sample_values == "test")
    validation_mask = response_present & (sample_values == "validation")
    if not bool(train_mask.any()):
        raise ValueError("No training rows are available after validation")

    feature_frame = work_frame[feature_names].copy()
    for name in categorical_features:
        feature_frame[name] = feature_frame[name].astype("category")
    categorical_labels = categorical_feature_labels(feature_frame, categorical_features)

    offset_values = work_frame[offset_col].astype("float64") if offset_col else pd.Series([1.0] * len(work_frame), index=work_frame.index)
    log_offset = np.log(offset_values.to_numpy(dtype="float64"))
    use_supplied_init_score = init_score_requested({"init_score": selected_init_score})
    init_score_linear, init_score_prediction, init_score_metadata = init_score_arrays(
        work_frame=work_frame,
        selected_init_score=selected_init_score,
        selected_objective=selected_objective,
        dataset=dataset,
        np=np,
        pd=pd,
    )
    use_offset_init_score = bool(not use_supplied_init_score and should_use_offset_init_score(params, offset_col))
    active_init_score = init_score_linear if use_supplied_init_score else log_offset if use_offset_init_score else None
    train_init = active_init_score[train_mask.to_numpy()] if active_init_score is not None else None
    valid_init = active_init_score[test_mask.to_numpy()] if active_init_score is not None and bool(test_mask.any()) else None

    train_set = lgb.Dataset(
        feature_frame.loc[train_mask],
        label=work_frame.loc[train_mask, response_col].astype("float64"),
        categorical_feature=categorical_features or "auto",
        init_score=train_init,
        free_raw_data=False,
    )
    valid_sets = [train_set]
    valid_names = ["training"]
    if bool(test_mask.any()):
        valid_sets.append(
            lgb.Dataset(
                feature_frame.loc[test_mask],
                label=work_frame.loc[test_mask, response_col].astype("float64"),
                categorical_feature=categorical_features or "auto",
                init_score=valid_init,
                reference=train_set,
                free_raw_data=False,
            )
        )
        valid_names.append("test")
    validation_init = active_init_score[validation_mask.to_numpy()] if active_init_score is not None and bool(validation_mask.any()) else None

    evaluation_result: dict[str, dict[str, list[float]]] = {}
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

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    final_iteration = int(booster.current_iteration() if hasattr(booster, "current_iteration") else num_boost_round)
    if ebm_controller:
        ebm_controller.finish(final_iteration)
    best_iteration = int(
        (ebm_controller.best_iteration if ebm_controller and ebm_controller.best_iteration else None)
        or getattr(booster, "best_iteration", 0)
        or num_boost_round
    )
    append_holdout_evaluation(
        lgb=lgb,
        booster=booster,
        feature_frame=feature_frame,
        response=work_frame[response_col].astype("float64"),
        validation_mask=validation_mask,
        categorical_features=categorical_features,
        validation_init=validation_init,
        train_set=train_set,
        evaluation_result=evaluation_result,
    )
    model_path = store.artifact_path(model_id, "model")
    booster.save_model(str(model_path), num_iteration=best_iteration)

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
    raw_score = booster.predict(feature_frame, raw_score=True, num_iteration=best_iteration)
    if use_supplied_init_score and init_score_linear is not None:
        prediction = raw_score_to_prediction(np, init_score_linear + raw_score, str(init_score_metadata.get("transform") or "identity"))
    elif use_offset_init_score:
        prediction = np.exp(log_offset + raw_score)
    else:
        prediction = booster.predict(feature_frame, num_iteration=best_iteration)
    prediction_data = {
        "__lucidum_row_id": work_frame["__lucidum_row_id"].astype("int64").to_numpy(),
        "gbm_prediction": prediction,
    }
    if offset_col:
        prediction_array = np.asarray(prediction, dtype="float64")
        offset_array = offset_values.to_numpy(dtype="float64")
        rate = np.full(prediction_array.shape, np.nan, dtype="float64")
        valid_rate = np.isfinite(prediction_array) & np.isfinite(offset_array) & (offset_array > 0)
        rate[valid_rate] = prediction_array[valid_rate] / offset_array[valid_rate]
        prediction_data["gbm_prediction_rate"] = rate
    predictions = pd.DataFrame(prediction_data)
    write_dataframe_parquet(predictions, store.artifact_path(model_id, "predictions"))
    if use_supplied_init_score and init_score_linear is not None and init_score_prediction is not None:
        write_dataframe_parquet(
            init_score_dataframe(pd, work_frame, init_score_linear, init_score_prediction),
            store.artifact_path(model_id, "init_score"),
        )
        init_score_metadata["artifact_path"] = str(store.artifact_path(model_id, "init_score"))

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
    shap_written_rows = 0
    shap_summary_rows: list[dict[str, Any]] = []
    if shap_mode not in {"zero", "0", "none"}:
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
            pd=pd,
            booster=booster,
            feature_frame=feature_frame,
            score_frame=work_frame,
            feature_names=feature_names,
            model_id=model_id,
            shap_mode=shap_mode,
            shap_seed=shap_sampling_seed(params.get("seed")),
            best_iteration=best_iteration,
            shap_interaction_groups=interaction_group_constraints,
        )
        shap_written_rows = int(len(shap_frame))
        write_dataframe_parquet(shap_frame, store.artifact_path(model_id, "shap_long"))
        write_dataframe_parquet(shap_summary, store.artifact_path(model_id, "shap_summary"))
        shap_summary_rows = shap_summary.to_dict("records")
        feature_config = feature_config_with_mean_abs_shap(feature_config, shap_summary_rows)

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
    evaluation_frame = evaluation_dataframe(pd, evaluation_result)
    write_dataframe_parquet(evaluation_frame, store.artifact_path(model_id, "evaluation"))
    tree_table = tree_dataframe(pd, booster, categorical_labels=categorical_labels)
    write_dataframe_parquet(tree_table, store.artifact_path(model_id, "tree_table"))

    elapsed = time.perf_counter() - started
    ebm_metadata = ebm_controller.metadata() if ebm_controller else None
    best_metrics = best_metrics_from_evaluation(evaluation_result, selected_metric, best_iteration)
    feature_scenario = normalise_feature_scenario(payload.get("feature_scenario"))
    manifest = {
        "model_id": model_id,
        "label": model_label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_mode": training_mode,
        "objective": str(params.get("objective")),
        "metric": selected_metric,
        "response_column": response_col,
        "offset_column": offset_col,
        "best_iteration": best_iteration,
        "best_metrics": best_metrics,
        "num_iterations": num_boost_round,
        "training_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "scored_rows": int(len(score_frame)),
        "sample_column": sample_mode if sample_mode != "none" else None,
        "sample_source": sample_source,
        "source_columns": source_columns,
        "shap_rows": shap_written_rows,
        "timings": {"training_seconds": elapsed},
        "warnings": validation.warnings,
        "feature_importance": feature_config,
        "sources": {
            "predictions": store.source_id(model_id, "predictions"),
            **({"shap_long": store.source_id(model_id, "shap_long"), "shap_summary": store.source_id(model_id, "shap_summary")} if shap_summary_rows else {}),
        },
    }
    if feature_scenario:
        manifest["feature_scenario"] = feature_scenario
    if feature_interaction_constraints:
        manifest["feature_interaction_constraints"] = feature_interaction_constraints
    if ebm_metadata:
        manifest["ebm"] = ebm_metadata
    if grid_search:
        manifest["grid_search"] = grid_search
    manifest["init_score"] = init_score_metadata
    stored_params["init_score_metadata"] = init_score_metadata
    store.write_json(store.artifact_path(model_id, "feature_config"), feature_config)
    store.write_json(store.artifact_path(model_id, "parameters"), stored_params)
    store.write_json(
        store.artifact_path(model_id, "training_log"),
        {"evaluation": evaluation_result, "warnings": validation.warnings, **({"ebm": ebm_metadata} if ebm_metadata else {})},
    )
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)
    if activate:
        store.activate_model(model_id)
        manifest["active"] = True
    else:
        manifest["active"] = False
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
    return manifest


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


def phase_progress_payload(
    *,
    phase: str,
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
    booster: Any,
    feature_frame: Any,
    response: Any,
    validation_mask: Any,
    categorical_features: list[str],
    validation_init: Any,
    train_set: Any,
    evaluation_result: dict[str, dict[str, list[float]]],
) -> None:
    if not bool(validation_mask.any()):
        return
    validation_set = lgb.Dataset(
        feature_frame.loc[validation_mask],
        label=response.loc[validation_mask],
        categorical_feature=categorical_features or "auto",
        init_score=validation_init,
        reference=train_set,
        free_raw_data=False,
    )
    try:
        results = booster.eval(validation_set, name="validation")
    except Exception:
        return
    for item in results or []:
        if len(item) < 3:
            continue
        dataset_name = str(item[0])
        metric_name = str(item[1])
        value = json_safe_number(item[2])
        if value is None:
            continue
        evaluation_result.setdefault(dataset_name, {}).setdefault(metric_name, []).append(value)


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


def evaluation_dataframe(pd: Any, evaluation_result: dict[str, dict[str, list[float]]]) -> Any:
    rows: list[dict[str, Any]] = []
    for dataset_name, metrics in evaluation_result.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "iteration": iteration,
                        "value": float(value) if value is not None and math.isfinite(float(value)) else None,
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "metric", "iteration", "value"])


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
    pd: Any,
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
    positions = shap_sample_positions(np, mode=shap_mode, row_count=len(feature_frame), seed=shap_seed)
    if len(positions) == 0:
        return (
            pd.DataFrame(
                {
                    "__lucidum_row_id": pd.Series(dtype="int64"),
                    **{feature_name: pd.Series(dtype="float64") for feature_name in feature_names},
                    **{group["name"]: pd.Series(dtype="float64") for group in group_columns},
                }
            ),
            pd.DataFrame(columns=["gbm_model_id", "feature", "mean_abs_shap", "mean_shap", "row_count"]),
        )

    sampled = feature_frame.iloc[positions]
    sampled_scores = score_frame.iloc[positions]
    contributions = booster.predict(sampled, pred_contrib=True, num_iteration=best_iteration)
    if contributions.ndim == 3:
        contributions = contributions[:, :, 0]
    feature_contribs = contributions[:, : len(feature_names)]

    shap_frame = pd.DataFrame(
        {
            "__lucidum_row_id": sampled_scores["__lucidum_row_id"].astype("int64").to_numpy(),
            **{
                feature_name: feature_contribs[:, feature_index].astype(float)
                for feature_index, feature_name in enumerate(feature_names)
            },
        }
    )
    for group in group_columns:
        shap_frame[group["name"]] = shap_frame[group["features"]].sum(axis=1)
    if shap_frame.empty:
        summary = pd.DataFrame(columns=["gbm_model_id", "feature", "mean_abs_shap", "mean_shap", "row_count"])
    else:
        summary = pd.DataFrame(
            [
                {
                    "gbm_model_id": model_id,
                    "feature": feature_name,
                    "mean_abs_shap": float(np.abs(shap_frame[feature_name]).mean()),
                    "mean_shap": float(shap_frame[feature_name].mean()),
                    "row_count": int(shap_frame[feature_name].count()),
                }
                for feature_name in feature_names
            ]
        )
        summary = summary.sort_values("mean_abs_shap", ascending=False)
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
    "lightgbm_progress_payload",
    "lightgbm_interaction_constraints",
    "lightgbm_pair_interaction_constraints",
    "feature_config_with_mean_abs_shap",
    "normalise_feature_scenario",
    "should_use_offset_init_score",
    "shap_interaction_group_columns",
    "train_model",
    "training_projection_columns",
    "training_select_sql",
    "write_dataframe_parquet",
]
