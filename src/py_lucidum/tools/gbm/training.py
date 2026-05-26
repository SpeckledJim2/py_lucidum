from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import duckdb

from py_lucidum.core import Dataset, quote_ident, sql_literal

from .store import GbmModelStore
from .validation import (
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
    detect_sample_column,
    display_monotonicity,
    metric,
    objective,
    normalise_features,
    normalise_parameters,
    selected_offset_column,
    selected_response_column,
    uses_log_offset,
    validate_request,
)


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


def write_dataframe_parquet(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("artifact_df", frame)
        con.execute(f"COPY artifact_df TO {sql_literal(str(path))} (FORMAT PARQUET)")
    finally:
        con.close()


def train_model(dataset: Dataset, store: GbmModelStore, payload: dict[str, Any]) -> dict[str, Any]:
    lgb, np, pd = gbm_dependencies()
    started = time.perf_counter()
    validation = validate_request(dataset, payload)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    params = normalise_parameters(payload.get("parameters"))
    selected_objective = objective(params)
    params["objective"] = selected_objective
    num_boost_round = max(1, int(params.pop("num_iterations", 200) or 200))
    early_stopping_rounds = max(0, int(params.pop("early_stopping_rounds", 0) or 0))
    selected_metric = metric(params)
    params["metric"] = selected_metric

    with dataset.lock:
        columns = dataset.column_map()
        response_col = selected_response_column(payload, columns)
        offset_col = selected_offset_column(payload, columns)
        features = normalise_features(payload.get("features"), columns)
        where_sql = f"\nWHERE TRY_CAST({quote_ident(offset_col)} AS DOUBLE) > 0" if offset_col else ""
        select_sql = f"""
SELECT
  ROW_NUMBER() OVER () AS __lucidum_row_id,
  *
FROM {dataset.relation_sql()}
{where_sql}
"""
        score_frame = dataset.con.execute(select_sql).fetch_df()

    if score_frame.empty:
        raise ValueError("No rows are available for GBM training")

    model_label = str(payload.get("label") or "GBM").strip() or "GBM"
    model_id = store.create_model_id(model_label)
    model_dir = store.create_model_dir(model_id)
    feature_names = [feature["name"] for feature in features]
    categorical_features = [feature["name"] for feature in features if feature["kind"] == "categorical"]
    monotone_constraints = [int(feature["monotonicity"]) for feature in features]
    if any(monotone_constraints):
        params["monotone_constraints"] = monotone_constraints

    work_frame = score_frame.copy()
    work_frame[response_col] = pd.to_numeric(work_frame[response_col], errors="coerce")
    if offset_col:
        work_frame[offset_col] = pd.to_numeric(work_frame[offset_col], errors="coerce")
    response_present = work_frame[response_col].notna()
    sample_column = detect_sample_column(dataset, payload.get("sample_column"))
    sample_mode = "none"
    if sample_column:
        sample_values = work_frame[sample_column].astype(str).str.strip().str.lower()
        sample_mode = sample_column
    elif payload.get("create_sample"):
        sample_values = np.where(work_frame["__lucidum_row_id"].astype("int64") % 5 == 0, "test", "training")
        work_frame["__gbm_sample"] = sample_values
        sample_mode = "__gbm_sample"
    else:
        sample_values = pd.Series(["training"] * len(work_frame), index=work_frame.index)

    train_mask = response_present & (sample_values == "training")
    test_mask = response_present & (sample_values == "test")
    if not bool(train_mask.any()):
        raise ValueError("No training rows are available after validation")

    feature_frame = work_frame[feature_names].copy()
    for name in categorical_features:
        feature_frame[name] = feature_frame[name].astype("category")

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

    evaluation_result: dict[str, dict[str, list[float]]] = {}
    callbacks = [lgb.record_evaluation(evaluation_result)]
    if len(valid_sets) > 1 and early_stopping_rounds > 0:
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
    best_iteration = int(getattr(booster, "best_iteration", 0) or num_boost_round)
    model_path = store.artifact_path(model_id, "model")
    booster.save_model(str(model_path), num_iteration=best_iteration)

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

    evaluation_frame = evaluation_dataframe(pd, evaluation_result)
    write_dataframe_parquet(evaluation_frame, store.artifact_path(model_id, "evaluation"))
    tree_dump = booster.dump_model(num_iteration=best_iteration)
    store.write_json(store.artifact_path(model_id, "tree_dump"), tree_dump)
    tree_table = tree_dataframe(pd, booster)
    write_dataframe_parquet(tree_table, store.artifact_path(model_id, "tree_table"))

    shap_mode = str(payload.get("shap_rows") or "0").strip().lower()
    shap_summary_rows: list[dict[str, Any]] = []
    if shap_mode not in {"zero", "0", "none"}:
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

    elapsed = time.perf_counter() - started
    manifest = {
        "model_id": model_id,
        "label": model_label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objective": str(params.get("objective")),
        "metric": selected_metric,
        "response_column": response_col,
        "offset_column": offset_col,
        "best_iteration": best_iteration,
        "num_iterations": num_boost_round,
        "training_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "scored_rows": int(len(score_frame)),
        "sample_column": sample_mode if sample_mode != "none" else None,
        "shap_rows": int(len(shap_summary_rows) and min(len(score_frame), shap_row_limit(shap_mode, len(score_frame)))),
        "timings": {"training_seconds": elapsed},
        "warnings": validation.warnings,
        "feature_importance": feature_config,
        "sources": {
            "predictions": store.source_id(model_id, "predictions"),
            **({"shap_long": store.source_id(model_id, "shap_long"), "shap_summary": store.source_id(model_id, "shap_summary")} if shap_summary_rows else {}),
        },
    }
    store.write_json(store.artifact_path(model_id, "feature_config"), feature_config)
    store.write_json(store.artifact_path(model_id, "parameters"), params | {"num_iterations": num_boost_round, "early_stopping_rounds": early_stopping_rounds})
    store.write_json(store.artifact_path(model_id, "training_log"), {"evaluation": evaluation_result, "warnings": validation.warnings})
    store.write_json(store.artifact_path(model_id, "manifest"), manifest)
    store.activate_model(model_id)
    manifest["active"] = True
    return manifest


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


def tree_dataframe(pd: Any, booster: Any) -> Any:
    try:
        return booster.trees_to_dataframe()
    except Exception:
        return pd.DataFrame(columns=["tree_index", "node_depth", "node_index", "split_feature", "threshold", "value"])


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
    "should_use_offset_init_score",
    "train_model",
    "write_dataframe_parquet",
]
