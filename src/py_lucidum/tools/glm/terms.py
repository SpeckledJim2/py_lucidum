from __future__ import annotations

import re
from typing import Any


def column_tokens(expression: str, columns: list[str]) -> list[str]:
    found: list[str] = []
    for column in sorted(columns, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
        if re.search(pattern, expression):
            found.append(column)
    return sorted(set(found))


def term_groups(estimator: Any, offset_terms: list[str], source_columns: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    spec = getattr(estimator, "X_model_spec_", None)
    source_set = set(source_columns)
    if spec is not None:
        term_variables = getattr(spec, "term_variables", {}) or {}
        term_indices = getattr(spec, "term_indices", {}) or {}
        for term in getattr(spec, "terms", []) or []:
            indices = list(term_indices.get(term, []) or [])
            if not indices:
                continue
            variables = tuple(sorted(str(name) for name in (term_variables.get(term, set()) or set()) if str(name) in source_set))
            if not variables:
                continue
            entry = groups.setdefault(variables, {"variables": list(variables), "term_indices": [], "offset_terms": []})
            entry["term_indices"].extend(indices)
    for expression in offset_terms:
        variables = tuple(column_tokens(expression, source_columns))
        entry = groups.setdefault(variables, {"variables": list(variables), "term_indices": [], "offset_terms": []})
        entry["offset_terms"].append(expression)
    return groups


def model_matrix(estimator: Any, frame: Any, context: dict[str, Any]) -> Any:
    matrix = estimator.X_model_spec_.get_model_matrix(frame, context=context)
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "to_numpy"):
        return matrix.to_numpy()
    return matrix


__all__ = ["column_tokens", "model_matrix", "term_groups"]
