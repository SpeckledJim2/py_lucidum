from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import duckdb

from py_lucidum.core import sql_literal

from .store import GbmModelStore


MAX_SPLIT_LABEL_LENGTH = 160


def tree_summary(store: GbmModelStore, model_id: str) -> dict[str, Any]:
    store.manifest(model_id)
    rows = tree_summary_from_table(store, model_id)
    if not rows:
        rows = tree_summary_from_dump(store, model_id)
    return {"model_id": model_id, "trees": rows}


def tree_detail(store: GbmModelStore, model_id: str, tree_index: int) -> dict[str, Any]:
    store.manifest(model_id)
    dump = store.read_json(store.artifact_path(model_id, "tree_dump"), {})
    if not isinstance(dump, dict):
        dump = {}
    tree_info = selected_tree_info(dump, tree_index)
    if not tree_info:
        return {"model_id": model_id, "tree": tree_index, "root": None, "values": []}
    root = tree_info.get("tree_structure")
    if not isinstance(root, dict):
        return {"model_id": model_id, "tree": tree_index, "root": None, "values": []}
    feature_names = [str(name) for name in dump.get("feature_names", []) if name is not None]
    feature_config = store.read_json(store.artifact_path(model_id, "feature_config"), [])
    category_map = categorical_feature_map(dump, feature_config if isinstance(feature_config, list) else [])
    values: list[float] = []
    normalised = normalise_tree_node(
        root,
        tree_index=tree_index,
        feature_names=feature_names,
        category_map=category_map,
        values=values,
        is_root=True,
    )
    return {"model_id": model_id, "tree": tree_index, "root": normalised, "values": values}


def tree_summary_from_table(store: GbmModelStore, model_id: str) -> list[dict[str, Any]]:
    path = store.artifact_path(model_id, "tree_table")
    if not path.exists():
        return []
    con = duckdb.connect(database=":memory:")
    try:
        try:
            records = con.execute(
                f"""
SELECT tree_index, split_feature, split_gain
FROM read_parquet({sql_literal(str(path))})
ORDER BY tree_index, node_depth, node_index
"""
            ).fetchall()
        except duckdb.Error:
            return []
    finally:
        con.close()
    grouped: dict[int, dict[str, Any]] = {}
    for tree_index, split_feature, split_gain in records:
        if tree_index is None:
            continue
        tree = int(tree_index)
        item = grouped.setdefault(tree, {"tree": tree, "features": [], "gain": 0.0})
        feature = clean_feature_name(split_feature)
        if feature and feature not in item["features"]:
            item["features"].append(feature)
        gain_value = finite_float(split_gain)
        if gain_value is not None:
            item["gain"] += gain_value
    return [summary_row(item["tree"], item["features"], item["gain"]) for item in sorted(grouped.values(), key=lambda row: row["tree"])]


def tree_summary_from_dump(store: GbmModelStore, model_id: str) -> list[dict[str, Any]]:
    dump = store.read_json(store.artifact_path(model_id, "tree_dump"), {})
    if not isinstance(dump, dict):
        return []
    feature_names = [str(name) for name in dump.get("feature_names", []) if name is not None]
    rows: list[dict[str, Any]] = []
    for index, tree in enumerate(dump.get("tree_info", []) or []):
        if not isinstance(tree, dict):
            continue
        tree_index = int(tree.get("tree_index", index) or 0)
        features: list[str] = []
        gain = 0.0
        for node in iter_split_nodes(tree.get("tree_structure")):
            feature = feature_name(node.get("split_feature"), feature_names)
            if feature and feature not in features:
                features.append(feature)
            split_gain = node.get("split_gain")
            gain_value = finite_float(split_gain)
            if gain_value is not None:
                gain += gain_value
        rows.append(summary_row(tree_index, features, gain))
    return sorted(rows, key=lambda row: row["tree"])


def normalise_tree_node(
    node: dict[str, Any],
    *,
    tree_index: int,
    feature_names: list[str],
    category_map: dict[str, list[str]],
    values: list[float],
    is_root: bool = False,
) -> dict[str, Any]:
    if "leaf_index" in node:
        value = finite_float(node.get("leaf_value"))
        if value is not None:
            values.append(value)
        leaf_index = int(node.get("leaf_index") or 0)
        return {
            "id": f"{tree_index}-L{leaf_index}",
            "type": "leaf",
            "leaf_index": leaf_index,
            "cover": int(node.get("leaf_count") or 0),
            "value": value,
            "label": node_label("Leaf", node.get("leaf_count"), None, value, is_root=False),
            "tooltip": node_tooltip("Leaf", node.get("leaf_count"), None, value),
            "children": [],
        }

    feature = feature_name(node.get("split_feature"), feature_names)
    value = finite_float(node.get("internal_value"))
    gain = finite_float(node.get("split_gain"))
    if value is not None:
        values.append(value)
    decision_type = str(node.get("decision_type") or "<=").strip() or "<="
    threshold = split_threshold_label(node.get("threshold"), feature, decision_type, category_map)
    default_left = bool(node.get("default_left"))
    left_child = node.get("left_child") if isinstance(node.get("left_child"), dict) else None
    right_child = node.get("right_child") if isinstance(node.get("right_child"), dict) else None
    children: list[dict[str, Any]] = []
    if left_child:
        child = normalise_tree_node(
            left_child,
            tree_index=tree_index,
            feature_names=feature_names,
            category_map=category_map,
            values=values,
        )
        child["edge_label"] = branch_label(decision_type, threshold["label"], left=True)
        child["edge_tooltip"] = branch_label(decision_type, threshold["full"], left=True)
        child["default_branch"] = default_left
        children.append(child)
    if right_child:
        child = normalise_tree_node(
            right_child,
            tree_index=tree_index,
            feature_names=feature_names,
            category_map=category_map,
            values=values,
        )
        child["edge_label"] = "else"
        child["edge_tooltip"] = f"else {branch_label(decision_type, threshold['full'], left=True)}"
        child["default_branch"] = not default_left
        children.append(child)

    split_index = int(node.get("split_index") or 0)
    label = node_label(feature, node.get("internal_count"), gain, value, is_root=is_root, tree_index=tree_index)
    return {
        "id": f"{tree_index}-S{split_index}",
        "type": "split",
        "split_index": split_index,
        "feature": feature,
        "cover": int(node.get("internal_count") or 0),
        "gain": gain,
        "value": value,
        "decision_type": decision_type,
        "threshold": threshold["label"],
        "threshold_full": threshold["full"],
        "default_child": "left" if default_left else "right",
        "label": label,
        "tooltip": node_tooltip(feature, node.get("internal_count"), gain, value, tree_index=tree_index if is_root else None),
        "children": children,
    }


def selected_tree_info(dump: dict[str, Any], tree_index: int) -> dict[str, Any] | None:
    for index, tree in enumerate(dump.get("tree_info", []) or []):
        if isinstance(tree, dict) and int(tree.get("tree_index", index) or 0) == tree_index:
            return tree
    return None


def categorical_feature_map(dump: dict[str, Any], feature_config: list[dict[str, Any]] | None = None) -> dict[str, list[str]]:
    feature_infos = dump.get("feature_infos") if isinstance(dump.get("feature_infos"), dict) else {}
    feature_names = [str(name) for name in dump.get("feature_names", []) if name is not None]
    categorical_names = [
        name for name in feature_names
        if isinstance(feature_infos.get(name), dict) and feature_infos[name].get("values")
    ]
    configured_categoricals = {
        str(item.get("name"))
        for item in (feature_config or [])
        if isinstance(item, dict) and "categorical" in str(item.get("kind") or "").lower()
    }
    categorical_names.extend(
        name for name in feature_names
        if name in configured_categoricals and name not in categorical_names
    )
    categories = dump.get("pandas_categorical") if isinstance(dump.get("pandas_categorical"), list) else []
    result: dict[str, list[str]] = {}
    for name, values in zip(categorical_names, categories):
        if isinstance(values, list):
            result[name] = [str(value) for value in values]
    return result


def split_threshold_label(raw_threshold: Any, feature: str, decision_type: str, category_map: dict[str, list[str]]) -> dict[str, str]:
    text = threshold_text(raw_threshold)
    if decision_type == "==" and feature in category_map:
        decoded = decode_category_threshold(text, category_map[feature])
        if decoded:
            text = decoded
    elif decision_type == "==":
        text = text.replace("||", " / ")
    compact = compact_label(text)
    return {"label": compact, "full": text}


def decode_category_threshold(text: str, categories: list[str]) -> str:
    labels: list[str] = []
    for part in text.split("||"):
        try:
            index = int(float(part))
        except ValueError:
            continue
        if 0 <= index < len(categories):
            labels.append(categories[index])
        else:
            labels.append(str(index))
    return " / ".join(labels)


def branch_label(decision_type: str, threshold: str, *, left: bool) -> str:
    if not left:
        return "else"
    if decision_type == "==":
        return f"== {threshold}"
    return f"{decision_type} {threshold}"


def threshold_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number) < 1e-30:
            return "0"
        return format_number(number)
    text = str(value)
    return "0" if "e-35" in text else text


def compact_label(text: str, max_length: int = MAX_SPLIT_LABEL_LENGTH) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_length:
        return clean
    return f"{clean[:max_length - 3]}..."


def node_label(title: str, cover: Any, gain: float | None, value: float | None, *, is_root: bool = False, tree_index: int | None = None) -> list[str]:
    lines = [str(title or "Feature")]
    if is_root and tree_index is not None:
        lines.insert(0, f"Tree {tree_index}")
    lines.append(f"Cover: {format_count(cover)}")
    if gain is not None:
        lines.append(f"Gain: {format_number(gain)}")
    lines.append(f"Value: {format_number(value)}")
    return lines


def node_tooltip(title: str, cover: Any, gain: float | None, value: float | None, *, tree_index: int | None = None) -> str:
    return "\n".join(node_label(title, cover, gain, value, is_root=tree_index is not None, tree_index=tree_index))


def iter_split_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(node, dict):
        return
    if "split_index" in node:
        yield node
        yield from iter_split_nodes(node.get("left_child"))
        yield from iter_split_nodes(node.get("right_child"))


def feature_name(value: Any, feature_names: list[str]) -> str:
    if isinstance(value, int) and 0 <= value < len(feature_names):
        return feature_names[value]
    if isinstance(value, float) and value.is_integer() and 0 <= int(value) < len(feature_names):
        return feature_names[int(value)]
    return clean_feature_name(value) or "Feature"


def clean_feature_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.upper() == "NA" or text.lower() == "nan":
        return ""
    return text


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_count(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "0"
    return f"{int(round(number)):,}"


def format_number(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "0"
    magnitude = abs(number)
    if number.is_integer():
        return f"{int(number):,}"
    if magnitude >= 1000:
        return f"{number:,.0f}"
    if magnitude >= 10:
        return f"{number:,.1f}"
    return f"{number:,.4f}".rstrip("0").rstrip(".")


def summary_row(tree: int, features: list[str], gain: float) -> dict[str, Any]:
    return {
        "tree": int(tree),
        "dim": len(features),
        "features": " x ".join(features),
        "gain": round(float(gain or 0.0)),
    }


__all__ = ["tree_detail", "tree_summary", "tree_summary_from_dump", "tree_summary_from_table"]
