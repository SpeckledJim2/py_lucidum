from __future__ import annotations

import math
from typing import Any

import duckdb

from py_lucidum.core import quote_ident, sql_literal

from .store import GbmModelStore


MAX_SPLIT_LABEL_LENGTH = 160
MAX_CATEGORICAL_SPLIT_LABEL_CATEGORIES = 12
MAX_CATEGORICAL_SPLIT_LABEL_PREVIEW = 3
TREE_DETAIL_COLUMNS = [
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


def tree_summary(store: GbmModelStore, model_id: str) -> dict[str, Any]:
    store.manifest(model_id)
    rows = tree_summary_from_table(store, model_id)
    return {"model_id": model_id, "trees": rows}


def ebm_gain_summary(store: GbmModelStore, model_id: str) -> dict[str, Any]:
    manifest = store.manifest(model_id)
    if str(manifest.get("training_mode") or "").strip().lower() != "ebm":
        return {"model_id": model_id, "rows": []}
    rows = ebm_gain_summary_from_table(store, model_id, manifest.get("best_iteration"))
    return {"model_id": model_id, "rows": rows}


def tree_detail(store: GbmModelStore, model_id: str, tree_index: int) -> dict[str, Any]:
    store.manifest(model_id)
    rows = tree_rows_from_table(store, model_id, tree_index)
    if not rows:
        return {"model_id": model_id, "tree": tree_index, "root": None, "values": []}
    values: list[float] = []
    normalised = normalise_tree_rows(rows, tree_index=tree_index, values=values)
    if not normalised:
        return {"model_id": model_id, "tree": tree_index, "root": None, "values": []}
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


def ebm_gain_summary_from_table(store: GbmModelStore, model_id: str, best_iteration: Any = None) -> list[dict[str, Any]]:
    path = store.artifact_path(model_id, "tree_table")
    if not path.exists():
        return []
    con = duckdb.connect(database=":memory:")
    try:
        try:
            columns = parquet_columns(con, path)
            if not {"tree_index", "split_feature", "split_gain"}.issubset(columns):
                return []
            best_iteration_value = count_int(best_iteration)
            where_sql = "WHERE tree_index IS NOT NULL"
            params: list[Any] = []
            if best_iteration_value > 0:
                where_sql += " AND tree_index < ?"
                params.append(best_iteration_value)
            records = con.execute(
                f"""
SELECT tree_index, split_feature, split_gain
FROM read_parquet({sql_literal(str(path))})
{where_sql}
ORDER BY tree_index
""",
                params,
            ).fetchall()
        except duckdb.Error:
            return []
    finally:
        con.close()

    trees: dict[int, dict[str, Any]] = {}
    for tree_index, split_feature, split_gain in records:
        if tree_index is None:
            continue
        tree = int(tree_index)
        item = trees.setdefault(tree, {"features": set(), "gain": 0.0})
        feature = clean_feature_name(split_feature)
        if feature:
            item["features"].add(feature)
        gain_value = finite_float(split_gain)
        if gain_value is not None:
            item["gain"] += gain_value

    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in trees.values():
        features = tuple(sorted(item["features"], key=lambda value: (value.lower(), value)))
        if not features:
            continue
        row = grouped.setdefault(features, {"features": features, "trees": 0, "gain": 0.0})
        row["trees"] += 1
        row["gain"] += item["gain"]

    total_gain = sum(float(row["gain"] or 0.0) for row in grouped.values())
    rows = [
        ebm_gain_summary_row(row["features"], row["trees"], row["gain"], total_gain)
        for row in grouped.values()
    ]
    return sorted(rows, key=lambda row: (-float(row["gain"] or 0.0), str(row["tree_features"]).lower()))


def tree_rows_from_table(store: GbmModelStore, model_id: str, tree_index: int) -> list[dict[str, Any]]:
    path = store.artifact_path(model_id, "tree_table")
    if not path.exists():
        return []
    con = duckdb.connect(database=":memory:")
    try:
        try:
            columns = parquet_columns(con, path)
            if "tree_index" not in columns or "node_index" not in columns:
                return []
            select_sql = ", ".join(select_expression(name, columns) for name in TREE_DETAIL_COLUMNS)
            order_columns = [name for name in ("node_depth", "node_index") if name in columns]
            order_sql = ", ".join(quote_ident(name) for name in order_columns) or "node_index"
            records = con.execute(
                f"""
SELECT {select_sql}
FROM read_parquet({sql_literal(str(path))})
WHERE tree_index = ?
ORDER BY {order_sql}
""",
                [int(tree_index)],
            ).fetchall()
        except duckdb.Error:
            return []
    finally:
        con.close()
    return [dict(zip(TREE_DETAIL_COLUMNS, record)) for record in records]


def parquet_columns(con: duckdb.DuckDBPyConnection, path: Any) -> set[str]:
    records = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_literal(str(path))})").fetchall()
    return {str(record[0]) for record in records if record and record[0] is not None}


def select_expression(name: str, columns: set[str]) -> str:
    if name in columns:
        return quote_ident(name)
    return f"NULL AS {quote_ident(name)}"


def normalise_tree_rows(rows: list[dict[str, Any]], *, tree_index: int, values: list[float]) -> dict[str, Any] | None:
    rows_by_id = {clean_node_ref(row.get("node_index")): row for row in rows if clean_node_ref(row.get("node_index"))}
    if not rows_by_id:
        return None
    root = next((row for row in rows if not clean_node_ref(row.get("parent_index"))), None)
    if root is None:
        root = min(rows, key=lambda row: (finite_float(row.get("node_depth")) or 0.0, clean_node_ref(row.get("node_index"))))
    total_cover = count_int(root.get("count"))
    return normalise_table_node(
        root,
        rows_by_id=rows_by_id,
        tree_index=tree_index,
        values=values,
        total_cover=total_cover,
        is_root=True,
        seen=set(),
    )


def normalise_table_node(
    row: dict[str, Any],
    *,
    rows_by_id: dict[str, dict[str, Any]],
    tree_index: int,
    values: list[float],
    total_cover: int,
    seen: set[str],
    is_root: bool = False,
) -> dict[str, Any] | None:
    node_id = clean_node_ref(row.get("node_index"))
    if not node_id or node_id in seen:
        return None
    seen.add(node_id)
    feature = clean_feature_name(row.get("split_feature"))
    value = finite_float(row.get("value"))
    if value is not None:
        values.append(value)
    cover = count_int(row.get("count"))

    if not feature:
        leaf_index = node_index_number(node_id)
        return {
            "id": node_id,
            "type": "leaf",
            "leaf_index": leaf_index,
            "cover": cover,
            "value": value,
            "label": node_label(
                "Leaf",
                cover,
                None,
                value,
                total_cover=total_cover,
                is_root=is_root,
                tree_index=tree_index if is_root else None,
            ),
            "tooltip": node_tooltip(
                "Leaf",
                cover,
                None,
                value,
                total_cover=total_cover,
                tree_index=tree_index if is_root else None,
            ),
            "children": [],
        }

    gain = finite_float(row.get("split_gain"))
    decision_type = str(row.get("decision_type") or "<=").strip() or "<="
    threshold = split_threshold_label(row.get("threshold"), decision_type, row.get("threshold_label"))
    default_left = str(row.get("missing_direction") or "").strip().lower() == "left"
    children: list[dict[str, Any]] = []
    left_child = rows_by_id.get(clean_node_ref(row.get("left_child")))
    right_child = rows_by_id.get(clean_node_ref(row.get("right_child")))
    if left_child:
        child = normalise_table_node(
            left_child,
            rows_by_id=rows_by_id,
            tree_index=tree_index,
            values=values,
            total_cover=total_cover,
            seen=seen,
        )
        if child:
            child["edge_label"] = branch_label(decision_type, threshold["label"], left=True)
            child["edge_tooltip"] = branch_label(decision_type, threshold["full"], left=True)
            child["default_branch"] = default_left
            children.append(child)
    if right_child:
        child = normalise_table_node(
            right_child,
            rows_by_id=rows_by_id,
            tree_index=tree_index,
            values=values,
            total_cover=total_cover,
            seen=seen,
        )
        if child:
            child["edge_label"] = "else"
            child["edge_tooltip"] = f"else {branch_label(decision_type, threshold['full'], left=True)}"
            child["default_branch"] = not default_left
            children.append(child)

    label = node_label(feature, cover, gain, value, total_cover=total_cover, is_root=is_root, tree_index=tree_index)
    return {
        "id": node_id,
        "type": "split",
        "split_index": node_index_number(node_id),
        "feature": feature,
        "cover": cover,
        "gain": gain,
        "value": value,
        "decision_type": decision_type,
        "threshold": threshold["label"],
        "threshold_full": threshold["full"],
        "default_child": "left" if default_left else "right",
        "label": label,
        "tooltip": node_tooltip(feature, cover, gain, value, total_cover=total_cover, tree_index=tree_index if is_root else None),
        "children": children,
    }


def split_threshold_label(raw_threshold: Any, decision_type: str, decoded_label: Any = None) -> dict[str, str]:
    decoded_text = clean_label(decoded_label)
    text = decoded_text or threshold_text(raw_threshold)
    if decision_type == "==":
        if not decoded_text:
            text = text.replace("||", " / ")
        return categorical_threshold_label(text)
    compact = compact_label(text)
    return {"label": compact, "full": text}


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
    if "e-35" in text:
        return "0"
    number = finite_float(text)
    return format_number(number) if number is not None else text


def categorical_threshold_label(text: str) -> dict[str, str]:
    full = " ".join(str(text or "").split())
    categories = split_categorical_label(full)
    if len(categories) > MAX_CATEGORICAL_SPLIT_LABEL_CATEGORIES:
        preview = " / ".join(categories[:MAX_CATEGORICAL_SPLIT_LABEL_PREVIEW])
        return {"label": f"{len(categories)} categories in split: {preview}, ...", "full": full}
    return {"label": full, "full": full}


def split_categorical_label(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(" / ") if part.strip()]


def compact_label(text: str, max_length: int = MAX_SPLIT_LABEL_LENGTH) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_length:
        return clean
    return f"{clean[:max_length - 3]}..."


def node_label(
    title: str,
    cover: Any,
    gain: float | None,
    value: float | None,
    *,
    total_cover: Any = None,
    is_root: bool = False,
    tree_index: int | None = None,
) -> list[str]:
    lines = [str(title or "Feature")]
    if is_root and tree_index is not None:
        lines.insert(0, f"Tree {tree_index}")
    lines.append(f"Cover: {format_cover(cover, total_cover)}")
    if gain is not None:
        lines.append(f"Gain: {format_number(gain)}")
    lines.append(f"Value: {format_number(value)}")
    return lines


def node_tooltip(
    title: str,
    cover: Any,
    gain: float | None,
    value: float | None,
    *,
    total_cover: Any = None,
    tree_index: int | None = None,
) -> str:
    return "\n".join(
        node_label(title, cover, gain, value, total_cover=total_cover, is_root=tree_index is not None, tree_index=tree_index)
    )


def clean_feature_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.upper() == "NA" or text.lower() == "nan":
        return ""
    return text


def clean_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def clean_node_ref(value: Any) -> str:
    return clean_label(value)


def node_index_number(node_id: Any) -> int:
    text = clean_node_ref(node_id)
    for marker in ("-S", "-L"):
        if marker in text:
            try:
                return int(text.rsplit(marker, 1)[1])
            except ValueError:
                return 0
    return 0


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def count_int(value: Any) -> int:
    number = finite_float(value)
    return int(round(number)) if number is not None else 0


def format_count(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "0"
    return f"{int(round(number)):,}"


def format_cover(value: Any, total: Any) -> str:
    count = format_count(value)
    percentage = cover_percentage(value, total)
    return f"{count} ({percentage})" if percentage else count


def cover_percentage(value: Any, total: Any) -> str:
    cover = finite_float(value)
    total_cover = finite_float(total)
    if cover is None or total_cover is None or total_cover <= 0:
        return ""
    return f"{cover / total_cover * 100:.1f}%"


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


def ebm_gain_summary_row(features: tuple[str, ...], trees: int, gain: float, total_gain: float) -> dict[str, Any]:
    gain_value = float(gain or 0.0)
    percent = gain_value / total_gain * 100 if total_gain > 0 else 0.0
    return {
        "tree_features": " x ".join(features),
        "dim": len(features),
        "trees": int(trees),
        "gain": round(gain_value, 3),
        "gain_percent": percent,
    }


__all__ = [
    "ebm_gain_summary",
    "ebm_gain_summary_from_table",
    "tree_detail",
    "tree_summary",
    "tree_summary_from_table",
]
