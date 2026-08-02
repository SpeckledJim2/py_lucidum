from __future__ import annotations

from collections.abc import Callable
from typing import Any

import duckdb

from py_lucidum.core import Dataset, json_number, sql_literal

from .sample import SAMPLE_COLUMN, sample_metadata
from .store import GbmModelStore
from .validation import (
    DATA_SAMPLE_STRATEGIES,
    DEFAULT_TRAINING_MODE,
    GBM_METRICS,
    GBM_OBJECTIVES,
    INIT_SCORE_NONE,
    INIT_SCORE_PARAMETER,
    OFFSET_COLUMN,
    RESPONSE_COLUMN,
    default_parameters,
    detect_sample_column,
    ebm_available,
    feature_rows,
    init_score_current_options,
    normalise_feature_interaction_pairs,
    normalise_training_mode,
)


class GbmConfigBuilder:
    def __init__(self, dataset: Dataset, store: GbmModelStore, feature_spec: Callable[[], Any] | None = None):
        self.dataset = dataset
        self.store = store
        self.feature_spec = feature_spec or (lambda: {})

    def active_gains(self) -> dict[str, float]:
        model_id = self.store.active_model_id()
        if not model_id:
            return {}
        try:
            features = self.store.model_feature_config(model_id)
        except ValueError:
            return {}
        return {
            str(item.get("name")): float(item.get("gain") or 0)
            for item in features
            if isinstance(item, dict) and item.get("name")
        }

    def active_shap_importance(self, model_id: str) -> dict[str, float]:
        path = self.store.artifact_path(model_id, "shap_summary")
        if not path.exists():
            return {}
        try:
            with self.dataset.lock:
                columns = {
                    str(row[0])
                    for row in self.dataset.con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})"
                    ).fetchall()
                }
                if not {"feature", "mean_abs_shap"}.issubset(columns):
                    return {}
                records = self.dataset.con.execute(
                    f"""
SELECT feature, mean_abs_shap
FROM read_parquet({sql_literal(str(path))})
WHERE feature IS NOT NULL
"""
                ).fetchall()
        except duckdb.Error:
            return {}
        values: dict[str, float] = {}
        for feature, value in records:
            name = str(feature or "").strip()
            number = json_number(value)
            if name and number is not None:
                values[name] = float(number)
        return values

    def feature_config_with_shap(self, features: list[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
        shap_importance = self.active_shap_importance(model_id)
        if not shap_importance:
            return features
        enriched: list[dict[str, Any]] = []
        for item in features:
            row = dict(item)
            name = str(row.get("name") or "").strip()
            if name in shap_importance and json_number(row.get("mean_abs_shap")) is None:
                row["mean_abs_shap"] = shap_importance[name]
            enriched.append(row)
        return enriched

    def active_feature_config(self) -> list[dict[str, Any]] | None:
        model_id = self.store.active_model_id()
        if not model_id:
            return None
        try:
            features = self.store.model_feature_config(model_id)
            if features:
                return self.feature_config_with_shap(features, model_id)
        except ValueError:
            return None
        return None

    def active_training_mode(self) -> str:
        model_id = self.store.active_model_id()
        if not model_id:
            return DEFAULT_TRAINING_MODE
        try:
            manifest = self.store.manifest(model_id)
        except ValueError:
            return DEFAULT_TRAINING_MODE
        return normalise_training_mode(manifest.get("training_mode"))

    def parameter_rows(self) -> list[dict[str, Any]]:
        values: dict[str, Any] = {}
        model_id = self.store.active_model_id()
        active_init_score = self.active_init_score_value()
        if model_id:
            try:
                stored = self.store.read_json(self.store.artifact_path(model_id, "parameters"), {})
            except ValueError:
                stored = {}
            if isinstance(stored, dict):
                values = stored

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in default_parameters():
            item = dict(row)
            name = str(item["name"])
            if name == INIT_SCORE_PARAMETER:
                item["value"] = active_init_score
            elif name in values:
                item["value"] = values[name]
            rows.append(item)
            seen.add(name)

        for name, value in values.items():
            text_name = str(name)
            if text_name in {
                INIT_SCORE_PARAMETER,
                "training_mode",
                "init_score_metadata",
                "interaction_constraints",
            }:
                continue
            if text_name not in seen:
                rows.append({"name": text_name, "value": value, "important": False})
        return rows

    def active_init_score_value(self) -> str:
        model_id = self.store.active_model_id()
        if not model_id:
            return INIT_SCORE_NONE
        try:
            manifest = self.store.manifest(model_id)
        except ValueError:
            return INIT_SCORE_NONE
        init_score = manifest.get("init_score") if isinstance(manifest.get("init_score"), dict) else {}
        value = str(init_score.get("value") or "").strip()
        return value or INIT_SCORE_NONE

    def feature_spec_payload(self) -> dict[str, Any]:
        spec = self.feature_spec()
        return spec if isinstance(spec, dict) else {"rows": [], "scenarios": []}

    def feature_groupings(self) -> dict[str, str]:
        rows = self.feature_spec_payload().get("rows", [])
        if not isinstance(rows, list):
            return {}
        return {
            str(row.get("feature")): str(row.get("grouping") or "")
            for row in rows
            if isinstance(row, dict) and row.get("feature")
        }

    def feature_bases(self) -> dict[str, str]:
        rows = self.feature_spec_payload().get("rows", [])
        if not isinstance(rows, list):
            return {}
        return {
            str(row.get("feature")): str(row.get("base") or "").strip()
            for row in rows
            if isinstance(row, dict) and row.get("feature") and str(row.get("base") or "").strip()
        }

    def feature_interaction_groupings(self, current_groupings: dict[str, str] | None = None) -> list[str]:
        values = (current_groupings if current_groupings is not None else self.feature_groupings()).values()
        return sorted({str(grouping).strip() for grouping in values if str(grouping).strip()}, key=str.lower)

    def feature_scenarios(self) -> list[dict[str, Any]]:
        scenarios = self.feature_spec_payload().get("scenarios", [])
        if not isinstance(scenarios, list):
            return []
        return [
            {
                "name": str(scenario.get("name") or ""),
                "features": [
                    str(feature)
                    for feature in scenario.get("features", [])
                    if str(feature).strip()
                ],
            }
            for scenario in scenarios
            if isinstance(scenario, dict) and scenario.get("name")
        ]

    def active_feature_scenario(self, current_scenarios: list[dict[str, Any]]) -> dict[str, Any] | None:
        model_id = self.store.active_model_id()
        if not model_id:
            return None
        try:
            manifest = self.store.manifest(model_id)
        except ValueError:
            return None
        stored = manifest.get("feature_scenario")
        if not isinstance(stored, dict):
            return None
        name = str(stored.get("name") or "").strip()
        if not name:
            return None
        stored_features = self.scenario_feature_list(stored.get("features"))
        current = {scenario["name"]: scenario for scenario in current_scenarios}.get(name)
        payload: dict[str, Any] = {"name": name, "features": stored_features}
        if not current:
            payload["status"] = "missing"
            return payload
        current_features = self.scenario_feature_list(current.get("features"))
        if self.scenario_feature_set(stored_features) == self.scenario_feature_set(current_features):
            payload["status"] = "current"
        else:
            payload["status"] = "stale"
            payload["current_features"] = current_features
        return payload

    @staticmethod
    def scenario_feature_list(raw_features: Any) -> list[str]:
        if not isinstance(raw_features, list):
            return []
        features: list[str] = []
        seen: set[str] = set()
        for item in raw_features:
            feature = str(item or "").strip()
            if feature and feature not in seen:
                features.append(feature)
                seen.add(feature)
        return features

    @staticmethod
    def scenario_feature_set(features: list[str]) -> set[str]:
        return {feature for feature in features if feature}

    def active_feature_interaction_constraints(self, current_groupings: dict[str, str], valid_groupings: list[str]) -> dict[str, Any] | None:
        model_id = self.store.active_model_id()
        if not model_id:
            return None
        try:
            manifest = self.store.manifest(model_id)
        except ValueError:
            return None
        stored = manifest.get("feature_interaction_constraints")
        if not isinstance(stored, dict):
            return None
        raw_group_model_metadata = manifest.get("feature_interaction_group_models")
        group_model_metadata = self.normalise_interaction_group_models(raw_group_model_metadata)
        has_group_model_metadata = isinstance(raw_group_model_metadata, dict)
        group_models_by_name = {
            group["grouping"]: group
            for group in group_model_metadata["groups"]
        }
        pairs = normalise_feature_interaction_pairs(stored.get("pairs"))
        groups = self.normalise_interaction_constraint_groups(stored.get("groups"))
        features = self.scenario_feature_list(stored.get("features"))
        current_grouping_set = set(valid_groupings)
        payload_groups: list[dict[str, Any]] = []
        for group in groups:
            grouping = group["grouping"]
            group_features = group["features"]
            payload: dict[str, Any] = {"grouping": grouping, "features": group_features}
            if grouping not in current_grouping_set:
                payload["status"] = "missing"
            elif any(current_groupings.get(feature, "") != grouping for feature in group_features):
                payload["status"] = "stale"
            else:
                payload["status"] = "current"
            if grouping in group_models_by_name:
                payload["group_model"] = group_models_by_name[grouping]
            payload_groups.append(payload)
        if str(stored.get("mode") or "").strip().lower() == "pairs" or pairs:
            uncovered_policy = str(stored.get("uncovered_policy") or "").strip().lower()
            policy_inferred = uncovered_policy not in {"singletons", "remainder"}
            if policy_inferred:
                uncovered_policy = self.infer_pair_uncovered_policy(
                    model_id,
                    pairs=pairs,
                    groups=groups,
                    explicit_features=features,
                )
            return {
                "mode": "pairs",
                "pairs": pairs,
                "groupings": [group["grouping"] for group in payload_groups],
                "features": features,
                "groups": payload_groups,
                "uncovered_policy": uncovered_policy,
                "policy_inferred": policy_inferred,
                **({"create_group_models": group_model_metadata["enabled"]} if has_group_model_metadata else {}),
            } if pairs else None
        if not groups and not features:
            return None
        return {
            "mode": "groups",
            "groupings": [group["grouping"] for group in payload_groups],
            "features": features,
            "groups": payload_groups,
            **({"create_group_models": group_model_metadata["enabled"]} if has_group_model_metadata else {}),
        }

    def normalise_interaction_group_models(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {"enabled": False, "groups": []}
        groups: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_group in raw.get("groups", []):
            if not isinstance(raw_group, dict):
                continue
            grouping = str(raw_group.get("grouping") or "").strip()
            status = str(raw_group.get("status") or "").strip().lower()
            if not grouping or grouping in seen or status not in {"verified", "no_trees"}:
                continue
            error = json_number(raw_group.get("max_absolute_error"))
            groups.append(
                {
                    "grouping": grouping,
                    "status": status,
                    "artifact": str(raw_group.get("artifact") or "").strip() or None,
                    "tree_count": max(0, int(json_number(raw_group.get("tree_count")) or 0)),
                    "verified_rows": max(0, int(json_number(raw_group.get("verified_rows")) or 0)),
                    "max_absolute_error": float(error) if error is not None else None,
                }
            )
            seen.add(grouping)
        return {"enabled": raw.get("enabled") is True, "groups": groups}

    def infer_pair_uncovered_policy(
        self,
        model_id: str,
        *,
        pairs: list[dict[str, str]],
        groups: list[dict[str, Any]],
        explicit_features: list[str],
    ) -> str:
        feature_names = self.store.model_feature_names(model_id)
        if not feature_names:
            return "unknown"

        covered_names = set(explicit_features)
        for pair in pairs:
            covered_names.update((pair["left"], pair["right"]))
        for group in groups:
            covered_names.update(group["features"])
        uncovered_indexes = [
            index
            for index, feature_name in enumerate(feature_names)
            if feature_name not in covered_names
        ]
        if not uncovered_indexes:
            return "singletons"

        parameters = self.store.model_parameters(model_id)
        raw_constraints = parameters.get("interaction_constraints")
        if not isinstance(raw_constraints, list):
            return "unknown"

        constraint_sets: list[frozenset[int]] = []
        for raw_group in raw_constraints:
            if not isinstance(raw_group, list):
                continue
            indexes: set[int] = set()
            valid = True
            for raw_index in raw_group:
                if isinstance(raw_index, bool):
                    valid = False
                    break
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    valid = False
                    break
                if index < 0 or index >= len(feature_names):
                    valid = False
                    break
                indexes.add(index)
            if valid and indexes:
                constraint_sets.append(frozenset(indexes))

        uncovered_set = frozenset(uncovered_indexes)
        if len(uncovered_indexes) > 1 and uncovered_set in constraint_sets:
            return "remainder"
        if all(frozenset({index}) in constraint_sets for index in uncovered_indexes):
            return "singletons"
        return "unknown"

    def normalise_interaction_constraint_groups(self, raw_groups: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_groups, list):
            return []
        groups: list[dict[str, Any]] = []
        seen_groupings: set[str] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                continue
            grouping = str(raw_group.get("grouping") or "").strip()
            if not grouping or grouping in seen_groupings:
                continue
            features = self.scenario_feature_list(raw_group.get("features"))
            if not features:
                continue
            groups.append({"grouping": grouping, "features": features})
            seen_groupings.add(grouping)
        return groups

    def payload(self) -> dict[str, Any]:
        model_features = self.active_feature_config()
        scenarios = self.feature_scenarios()
        current_feature_groupings = self.feature_groupings()
        interaction_groupings = self.feature_interaction_groupings(current_feature_groupings)
        with self.dataset.lock:
            sample = sample_metadata(self.dataset, self.store.generated_sample_path)
            sample_reserved = {SAMPLE_COLUMN} if sample.get("source") == "dataset" else set()
            features = feature_rows(
                self.dataset,
                self.active_gains(),
                model_features=model_features,
                reserved_names=sample_reserved,
                feature_groupings=current_feature_groupings,
            )
            sample_column = detect_sample_column(self.dataset)
            can_use_ebm = ebm_available(self.dataset, generated_sample_path=self.store.generated_sample_path)
            current_init_score_options = init_score_current_options(
                self.dataset,
                self.active_init_score_value(),
                response_column=RESPONSE_COLUMN,
                sample_column=sample_column,
            )
        return {
            "tool": "gbm",
            "status": "ready",
            "response": RESPONSE_COLUMN,
            "offset": OFFSET_COLUMN,
            "sample_column": sample_column,
            "sample": sample,
            "training_mode": self.active_training_mode(),
            "ebm_available": can_use_ebm,
            "parameters": self.parameter_rows(),
            "parameter_options": {
                "init_score": current_init_score_options,
                "objective": sorted(GBM_OBJECTIVES),
                "metric": sorted(GBM_METRICS),
                "data_sample_strategy": list(DATA_SAMPLE_STRATEGIES),
            },
            "features": features,
            "feature_scenarios": scenarios,
            "active_feature_scenario": self.active_feature_scenario(scenarios),
            "feature_interaction_groupings": interaction_groupings,
            "active_feature_interaction_constraints": self.active_feature_interaction_constraints(current_feature_groupings, interaction_groupings),
            "models": self.store.list_models(),
            "active_model_id": self.store.active_model_id(),
            "shap_options": [
                {"value": "0", "label": "0"},
                {"value": "10k", "label": "10k"},
                {"value": "100k", "label": "100k"},
                {"value": "all", "label": "All"},
            ],
        }


__all__ = ["GbmConfigBuilder"]
