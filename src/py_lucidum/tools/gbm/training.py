from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal

from .sample import (
    SAMPLE_COLUMN,
    create_generated_sample,
    dataset_sample_column,
    generated_sample_is_current,
    generated_sample_relation_sql,
)
from .store import GbmModelStore, best_metrics_from_evaluation
from .validation import (
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
    detect_sample_column,
    display_monotonicity,
    metric,
    normalise_training_mode,
    objective,
    normalise_features,
    normalise_parameters,
    selected_offset_column,
    selected_response_column,
    uses_log_offset,
    validate_request,
)


ProgressCallback = Callable[[dict[str, Any]], None]
EBM_INITIAL_LEARNING_RATE = 0.3


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


def training_projection_columns(
    *,
    response_col: str,
    offset_col: str | None,
    sample_column: str | None,
    feature_names: list[str],
    columns: dict[str, Any],
) -> list[str]:
    names = [response_col, offset_col, sample_column, *feature_names]
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
) -> dict[str, Any]:
    lgb, np, pd = gbm_dependencies()
    started = time.perf_counter()
    emit_progress(progress_callback, {"phase": "preparing", "message": "preparing GBM...", "percent": 0})
    validation = validate_request(dataset, payload, generated_sample_path=store.generated_sample_path)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    params = normalise_parameters(payload.get("parameters"))
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
        dataset_sample = dataset_sample_column(dataset)
        if not dataset_sample and payload.get("create_sample"):
            create_generated_sample(dataset, store.generated_sample_path)
        generated_sample_path = (
            store.generated_sample_path
            if not dataset_sample and generated_sample_is_current(dataset, store.generated_sample_path)
            else None
        )
        sample_column = SAMPLE_COLUMN if dataset_sample else None
        source_columns = list(columns)
        projection_columns = training_projection_columns(
            response_col=response_col,
            offset_col=offset_col,
            sample_column=sample_column,
            feature_names=feature_names,
            columns=columns,
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
    use_offset_init_score = should_use_offset_init_score(params, offset_col)
    train_init = log_offset[train_mask.to_numpy()] if use_offset_init_score else None
    valid_init = log_offset[test_mask.to_numpy()] if use_offset_init_score and bool(test_mask.any()) else None

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
    validation_init = log_offset[validation_mask.to_numpy()] if use_offset_init_score and bool(validation_mask.any()) else None

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
    if use_offset_init_score:
        prediction = np.exp(log_offset + raw_score)
    else:
        prediction = booster.predict(feature_frame, num_iteration=best_iteration)
    predictions = pd.DataFrame(
        {
            "__lucidum_row_id": work_frame["__lucidum_row_id"].astype("int64").to_numpy(),
            "gbm_prediction": prediction,
        }
    )
    write_dataframe_parquet(predictions, store.artifact_path(model_id, "predictions"))

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
            best_iteration=best_iteration,
        )
        write_dataframe_parquet(shap_frame, store.artifact_path(model_id, "shap_long"))
        write_dataframe_parquet(shap_summary, store.artifact_path(model_id, "shap_summary"))
        shap_summary_rows = shap_summary.to_dict("records")

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
        "shap_rows": int(len(shap_summary_rows) and min(len(score_frame), shap_row_limit(shap_mode, len(score_frame)))),
        "timings": {"training_seconds": elapsed},
        "warnings": validation.warnings,
        "feature_importance": feature_config,
        "sources": {
            "predictions": store.source_id(model_id, "predictions"),
            **({"shap_long": store.source_id(model_id, "shap_long"), "shap_summary": store.source_id(model_id, "shap_summary")} if shap_summary_rows else {}),
        },
    }
    if ebm_metadata:
        manifest["ebm"] = ebm_metadata
    store.write_json(store.artifact_path(model_id, "feature_config"), feature_config)
    store.write_json(store.artifact_path(model_id, "parameters"), stored_params)
    store.write_json(
        store.artifact_path(model_id, "training_log"),
        {"evaluation": evaluation_result, "warnings": validation.warnings, **({"ebm": ebm_metadata} if ebm_metadata else {})},
    )
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)
    store.activate_model(model_id)
    manifest["active"] = True
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
    best_iteration: int,
) -> tuple[Any, Any]:
    limit = shap_row_limit(shap_mode, len(feature_frame))
    sampled = feature_frame.head(limit)
    sampled_scores = score_frame.head(limit)
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


__all__ = [
    "MissingGbmDependency",
    "gbm_dependencies",
    "lightgbm_progress_payload",
    "should_use_offset_init_score",
    "train_model",
    "training_projection_columns",
    "training_select_sql",
    "write_dataframe_parquet",
]
