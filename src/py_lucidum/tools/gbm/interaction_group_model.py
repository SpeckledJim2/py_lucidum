"""Extract isolated LightGBM interaction-group models without importing LightGBM."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any


_TREE_LINE_RE = re.compile(r"Tree=(\d+)")
_PARAMETER_LINE_RE = re.compile(r"\[([^:\]]+):(?: (.*))?\]")
_PER_FEATURE_PARAMETERS = (
    "monotone_constraints",
    "feature_contri",
    "cegb_penalty_feature_lazy",
    "cegb_penalty_feature_coupled",
    "max_bin_by_feature",
)


class NoInteractionGroupTreesError(ValueError):
    """Raised when a valid saved interaction group did not produce any trees."""


def interaction_group_model_filename(grouping: str, used_filenames: Iterable[str] = ()) -> str:
    """Return a readable, deterministic filename for one interaction-group model."""

    name = str(grouping or "").strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "group"
    used = {str(filename).casefold() for filename in used_filenames}
    needs_hash = len(stem) > 80
    if needs_hash:
        stem = stem[:80].rstrip("._-") or "group"
    candidate = f"model_{stem}.txt"
    if candidate.casefold() in used:
        needs_hash = True
    if needs_hash:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        stem = stem[:71].rstrip("._-") or "group"
        candidate = f"model_{stem}_{digest}.txt"
    return candidate


def extract_lightgbm_interaction_group(
    source_model: str | PathLike[str],
    group_features: Iterable[str],
    output_model: str | PathLike[str],
    *,
    shap_centered: bool = True,
) -> Path:
    """Write the isolated trees for one saved LightGBM interaction group.

    The source must be a LightGBM text model whose saved
    ``interaction_constraints`` contain exactly ``group_features`` as one
    non-overlapping group. Trees that use the group are retained, feature
    indexes are compacted, and feature-dependent model metadata is subset so
    that LightGBM can load the result and predict from only those columns.

    By default, each retained tree is shifted by its count-weighted expected
    value. The extracted model's raw scores then equal the sum of the source
    model's SHAP values for ``group_features``. Set ``shap_centered=False`` to
    preserve the retained trees' original leaf outputs instead.

    This transformation parses the text format directly and deliberately does
    not import LightGBM, so it can be used without starting Lucidum or loading
    its optional modelling dependencies.
    """

    source_path = Path(source_model)
    output_path = Path(output_model)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("The extracted LightGBM model must not overwrite its source model")

    requested_features = _normalise_requested_features(group_features)
    model_text = source_path.read_text(encoding="utf-8")
    extracted_text = _extract_model_text(
        model_text,
        requested_features,
        shap_centered=shap_centered,
    )
    _atomic_write_text(output_path, extracted_text)
    return output_path


def _normalise_requested_features(group_features: Iterable[str]) -> tuple[str, ...]:
    if isinstance(group_features, (str, bytes)):
        raise TypeError("group_features must be an iterable of feature names, not one string")
    try:
        features = tuple(str(value).strip() for value in group_features)
    except TypeError as exc:
        raise TypeError("group_features must be an iterable of feature names") from exc
    if not features or any(not feature for feature in features):
        raise ValueError("Choose at least one nonblank LightGBM group feature")
    duplicates = sorted({feature for feature in features if features.count(feature) > 1})
    if duplicates:
        raise ValueError(f"Duplicate LightGBM group features: {', '.join(duplicates)}")
    return features


def _extract_model_text(
    model_text: str,
    requested_features: tuple[str, ...],
    *,
    shap_centered: bool,
) -> str:
    text = model_text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "tree":
        raise ValueError("The source is not a LightGBM text model")

    end_tree_positions = [index for index, line in enumerate(lines) if line.rstrip("\n") == "end of trees"]
    if len(end_tree_positions) != 1:
        raise ValueError("The LightGBM model must contain one 'end of trees' marker")
    end_trees = end_tree_positions[0]
    tree_positions = [
        index
        for index, line in enumerate(lines[:end_trees])
        if _TREE_LINE_RE.fullmatch(line.rstrip("\n"))
    ]
    if not tree_positions:
        raise ValueError("The LightGBM model does not contain any trees")

    header = lines[: tree_positions[0]]
    tail = lines[end_trees:]
    version = _header_value(header, "version")
    if version != "v4":
        raise ValueError(f"Unsupported LightGBM text-model version: {version}")
    feature_names = _header_values(header, "feature_names")
    feature_infos = _header_values(header, "feature_infos")
    if not feature_names or len(feature_infos) != len(feature_names):
        raise ValueError("The LightGBM feature_names and feature_infos metadata are inconsistent")
    max_feature_idx = _header_integer(header, "max_feature_idx")
    if max_feature_idx != len(feature_names) - 1:
        raise ValueError("The LightGBM max_feature_idx metadata is inconsistent with feature_names")

    unknown = [feature for feature in requested_features if feature not in feature_names]
    if unknown:
        raise ValueError(f"Unknown LightGBM group features: {', '.join(unknown)}")
    requested_set = set(requested_features)
    selected_indexes = tuple(index for index, name in enumerate(feature_names) if name in requested_set)
    selected_names = tuple(feature_names[index] for index in selected_indexes)
    selected_index_set = set(selected_indexes)
    index_map = {old_index: new_index for new_index, old_index in enumerate(selected_indexes)}

    parameter_positions = _parameter_positions(tail)
    interaction_value = _required_parameter_value(tail, parameter_positions, "interaction_constraints")
    interaction_groups = _parse_interaction_constraints(interaction_value, len(feature_names))
    _validate_requested_constraint(selected_index_set, selected_names, interaction_groups)

    linear_tree = _parameter_value(tail, parameter_positions, "linear_tree")
    if linear_tree and linear_tree.strip().lower() not in {"0", "false"}:
        raise ValueError("LightGBM linear-tree models are not supported for interaction-group extraction")

    tree_ends = [*tree_positions[1:], end_trees]
    tree_blocks = ["".join(lines[start:end]) for start, end in zip(tree_positions, tree_ends)]
    _validate_tree_metadata(header, tree_blocks)
    tree_feature_indexes = [
        _tree_feature_indexes(block, len(feature_names)) for block in tree_blocks
    ]
    selected_flags: list[bool] = []
    for tree_number, indexes in enumerate(tree_feature_indexes):
        if not indexes.intersection(selected_index_set):
            selected_flags.append(False)
            continue
        outside = indexes - selected_index_set
        if outside:
            outside_names = ", ".join(feature_names[index] for index in sorted(outside))
            raise ValueError(
                f"Tree {tree_number} mixes the requested interaction group with: {outside_names}"
            )
        selected_flags.append(True)

    num_tree_per_iteration = _header_integer(header, "num_tree_per_iteration")
    if num_tree_per_iteration < 1 or len(tree_blocks) % num_tree_per_iteration:
        raise ValueError("The LightGBM tree count is inconsistent with num_tree_per_iteration")
    selected_tree_indexes = _selected_complete_iterations(selected_flags, num_tree_per_iteration)
    if not selected_tree_indexes:
        raise NoInteractionGroupTreesError(
            "The requested interaction group has no trees in the LightGBM model: "
            f"{', '.join(selected_names)}"
        )

    transformed_trees = [
        _transform_tree_block(
            tree_blocks[old_index],
            new_index,
            index_map,
            shap_centered=shap_centered,
        )
        for new_index, old_index in enumerate(selected_tree_indexes)
    ]
    retained_iterations = len(transformed_trees) // num_tree_per_iteration

    categorical_indexes = _categorical_feature_indexes(
        _parameter_value(tail, parameter_positions, "categorical_feature"),
        feature_names,
        feature_infos,
    )
    _transform_header(header, selected_indexes, transformed_trees)
    _transform_tail(
        tail,
        parameter_positions,
        feature_names=feature_names,
        selected_indexes=selected_indexes,
        selected_names=set(selected_names),
        categorical_indexes=categorical_indexes,
        retained_iterations=retained_iterations,
    )
    return "".join(header) + "".join(transformed_trees) + "".join(tail)


def _header_positions(lines: list[str], key: str) -> list[int]:
    prefix = f"{key}="
    return [index for index, line in enumerate(lines) if line.startswith(prefix)]


def _header_value(lines: list[str], key: str) -> str:
    positions = _header_positions(lines, key)
    if len(positions) != 1:
        raise ValueError(f"The LightGBM model must contain one {key} header")
    return lines[positions[0]].rstrip("\n").split("=", 1)[1]


def _header_values(lines: list[str], key: str) -> list[str]:
    value = _header_value(lines, key).strip()
    return value.split() if value else []


def _header_integer(lines: list[str], key: str) -> int:
    try:
        return int(_header_value(lines, key))
    except ValueError as exc:
        raise ValueError(f"The LightGBM {key} header must be an integer") from exc


def _replace_header(lines: list[str], key: str, value: str) -> None:
    positions = _header_positions(lines, key)
    if len(positions) != 1:
        raise ValueError(f"The LightGBM model must contain one {key} header")
    lines[positions[0]] = f"{key}={value}\n"


def _parameter_positions(lines: list[str]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _PARAMETER_LINE_RE.fullmatch(line.rstrip("\n"))
        if not match:
            continue
        name = match.group(1)
        if name in positions:
            raise ValueError(f"Duplicate LightGBM parameter metadata: {name}")
        positions[name] = index
    return positions


def _parameter_value(lines: list[str], positions: dict[str, int], name: str) -> str | None:
    position = positions.get(name)
    if position is None:
        return None
    match = _PARAMETER_LINE_RE.fullmatch(lines[position].rstrip("\n"))
    if match is None:
        raise ValueError(f"Invalid LightGBM parameter metadata: {name}")
    return match.group(2) or ""


def _required_parameter_value(lines: list[str], positions: dict[str, int], name: str) -> str:
    value = _parameter_value(lines, positions, name)
    if value is None:
        raise ValueError(f"The LightGBM model does not contain saved {name} metadata")
    return value


def _replace_parameter(lines: list[str], positions: dict[str, int], name: str, value: str) -> None:
    position = positions.get(name)
    if position is not None:
        lines[position] = f"[{name}: {value}]\n"


def _parse_interaction_constraints(value: str, feature_count: int) -> list[set[int]]:
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return []
    raw_groups = re.findall(r"\[([^\[\]]*)\]", compact)
    if not raw_groups or ",".join(f"[{group}]" for group in raw_groups) != compact:
        raise ValueError("The saved LightGBM interaction_constraints metadata is invalid")
    groups: list[set[int]] = []
    for raw_group in raw_groups:
        tokens = raw_group.split(",")
        if any(token == "" for token in tokens):
            raise ValueError("Saved LightGBM interaction constraint groups contain an empty index")
        try:
            indexes = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("Saved LightGBM interaction constraints must use feature indexes") from exc
        if not indexes or len(indexes) != len(set(indexes)):
            raise ValueError("Saved LightGBM interaction constraint groups must be nonempty and unique")
        if min(indexes) < 0 or max(indexes) >= feature_count:
            raise ValueError("Saved LightGBM interaction constraints contain an invalid feature index")
        groups.append(set(indexes))
    return groups


def _validate_requested_constraint(
    selected_indexes: set[int],
    selected_names: tuple[str, ...],
    interaction_groups: list[set[int]],
) -> None:
    if selected_indexes not in interaction_groups:
        raise ValueError(
            "Requested features do not exactly match a saved LightGBM interaction constraint group: "
            f"{', '.join(selected_names)}"
        )
    for group in interaction_groups:
        if group != selected_indexes and group.intersection(selected_indexes):
            raise ValueError(
                "The requested LightGBM interaction constraint overlaps another saved constraint "
                "and is not isolated"
            )


def _validate_tree_metadata(header: list[str], tree_blocks: list[str]) -> None:
    tree_ids: list[int] = []
    for block in tree_blocks:
        first_line = block.splitlines()[0] if block else ""
        match = _TREE_LINE_RE.fullmatch(first_line)
        if match is None:
            raise ValueError("Invalid LightGBM tree block")
        tree_ids.append(int(match.group(1)))
    if tree_ids != list(range(len(tree_blocks))):
        raise ValueError("LightGBM Tree indexes must be contiguous and zero-based")

    try:
        saved_sizes = [int(value) for value in _header_value(header, "tree_sizes").split()]
    except ValueError as exc:
        raise ValueError("The LightGBM tree_sizes header must contain integers") from exc
    actual_sizes = [len(block.encode("utf-8")) for block in tree_blocks]
    if saved_sizes != actual_sizes:
        raise ValueError("The LightGBM tree_sizes metadata does not match its tree blocks")


def _tree_feature_indexes(block: str, feature_count: int) -> set[int]:
    split_lines = [line for line in block.splitlines() if line.startswith("split_feature=")]
    if len(split_lines) != 1:
        raise ValueError("Each LightGBM tree must contain one split_feature row")
    raw_values = split_lines[0].split("=", 1)[1].strip()
    try:
        indexes = {int(value) for value in raw_values.split()} if raw_values else set()
    except ValueError as exc:
        raise ValueError("A LightGBM tree contains an invalid split_feature index") from exc
    if indexes and (min(indexes) < 0 or max(indexes) >= feature_count):
        raise ValueError("A LightGBM tree contains an out-of-range split_feature index")
    if any(line.startswith("leaf_features=") for line in block.splitlines()):
        raise ValueError("LightGBM linear-tree models are not supported for interaction-group extraction")
    return indexes


def _selected_complete_iterations(selected_flags: list[bool], trees_per_iteration: int) -> list[int]:
    selected: list[int] = []
    for start in range(0, len(selected_flags), trees_per_iteration):
        batch = selected_flags[start : start + trees_per_iteration]
        if any(batch) and not all(batch):
            iteration = start // trees_per_iteration
            raise ValueError(
                "The requested interaction group occupies only part of LightGBM iteration "
                f"{iteration}; a valid {trees_per_iteration}-tree iteration cannot be extracted"
            )
        if all(batch):
            selected.extend(range(start, start + trees_per_iteration))
    return selected


def _transform_tree_block(
    block: str,
    tree_index: int,
    index_map: dict[int, int],
    *,
    shap_centered: bool,
) -> str:
    lines = block.splitlines(keepends=True)
    lines[0] = f"Tree={tree_index}\n"
    for index, line in enumerate(lines):
        if not line.startswith("split_feature="):
            continue
        raw_values = line.rstrip("\n").split("=", 1)[1].strip()
        old_indexes = [int(value) for value in raw_values.split()] if raw_values else []
        try:
            new_indexes = [index_map[value] for value in old_indexes]
        except KeyError as exc:
            raise ValueError("A retained LightGBM tree uses a feature outside the requested group") from exc
        lines[index] = "split_feature=" + " ".join(str(value) for value in new_indexes) + "\n"
    if shap_centered:
        _center_tree_values(lines)
    return "".join(lines)


def _center_tree_values(lines: list[str]) -> None:
    leaf_values = _tree_numeric_values(lines, "leaf_value")
    leaf_counts = _tree_numeric_values(lines, "leaf_count")
    if len(leaf_values) != len(leaf_counts) or not leaf_values:
        raise ValueError("A LightGBM tree has inconsistent leaf_value and leaf_count metadata")
    total_count = math.fsum(leaf_counts)
    if not math.isfinite(total_count) or total_count <= 0:
        raise ValueError("A LightGBM tree must have a positive total leaf_count")
    expected_value = math.fsum(
        value * count for value, count in zip(leaf_values, leaf_counts)
    ) / total_count
    _shift_tree_numeric_values(lines, "leaf_value", expected_value)
    _shift_tree_numeric_values(lines, "internal_value", expected_value)


def _tree_numeric_values(lines: list[str], key: str) -> list[float]:
    prefix = f"{key}="
    positions = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(positions) != 1:
        raise ValueError(f"Each LightGBM tree must contain one {key} row")
    raw_values = lines[positions[0]].rstrip("\n").split("=", 1)[1].strip()
    try:
        values = [float(value) for value in raw_values.split()] if raw_values else []
    except ValueError as exc:
        raise ValueError(f"A LightGBM tree contains an invalid {key} value") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"A LightGBM tree contains a non-finite {key} value")
    return values


def _shift_tree_numeric_values(lines: list[str], key: str, shift: float) -> None:
    values = _tree_numeric_values(lines, key)
    position = next(index for index, line in enumerate(lines) if line.startswith(f"{key}="))
    lines[position] = f"{key}=" + " ".join(format(value - shift, ".17g") for value in values) + "\n"


def _transform_header(
    header: list[str], selected_indexes: tuple[int, ...], transformed_trees: list[str]
) -> None:
    feature_names = _header_values(header, "feature_names")
    feature_infos = _header_values(header, "feature_infos")
    _replace_header(header, "max_feature_idx", str(len(selected_indexes) - 1))
    _replace_header(
        header,
        "feature_names",
        " ".join(feature_names[index] for index in selected_indexes),
    )
    _replace_header(
        header,
        "feature_infos",
        " ".join(feature_infos[index] for index in selected_indexes),
    )
    monotone_positions = _header_positions(header, "monotone_constraints")
    if monotone_positions:
        values = _header_values(header, "monotone_constraints")
        if len(values) != len(feature_names):
            raise ValueError("The LightGBM monotone_constraints header has the wrong feature count")
        _replace_header(
            header,
            "monotone_constraints",
            " ".join(values[index] for index in selected_indexes),
        )
    _replace_header(
        header,
        "tree_sizes",
        " ".join(str(len(block.encode("utf-8"))) for block in transformed_trees),
    )


def _transform_tail(
    tail: list[str],
    parameter_positions: dict[str, int],
    *,
    feature_names: list[str],
    selected_indexes: tuple[int, ...],
    selected_names: set[str],
    categorical_indexes: tuple[int, ...],
    retained_iterations: int,
) -> None:
    importance_positions = [
        index for index, line in enumerate(tail) if line.rstrip("\n") == "feature_importances:"
    ]
    parameters_markers = [index for index, line in enumerate(tail) if line.rstrip("\n") == "parameters:"]
    end_parameter_markers = [
        index for index, line in enumerate(tail) if line.rstrip("\n") == "end of parameters"
    ]
    if (
        len(importance_positions) != 1
        or len(parameters_markers) != 1
        or len(end_parameter_markers) != 1
    ):
        raise ValueError(
            "The LightGBM model must contain feature_importances and parameters sections"
        )
    importance_start = importance_positions[0]
    parameters_start = parameters_markers[0]
    if importance_start >= parameters_start or parameters_start >= end_parameter_markers[0]:
        raise ValueError("The LightGBM feature_importances section is invalid")
    for index in range(importance_start + 1, parameters_start):
        value = tail[index].rstrip("\n")
        if not value:
            continue
        if "=" not in value:
            raise ValueError("The LightGBM feature_importances section contains an invalid row")
        feature = value.split("=", 1)[0]
        if feature not in feature_names:
            raise ValueError(f"Unknown feature in LightGBM feature_importances: {feature}")
        if feature not in selected_names:
            tail[index] = ""

    selected_index_set = set(selected_indexes)
    index_map = {old_index: new_index for new_index, old_index in enumerate(selected_indexes)}
    compact_constraint = "[" + ",".join(str(index_map[index]) for index in selected_indexes) + "]"
    _replace_parameter(tail, parameter_positions, "interaction_constraints", compact_constraint)
    _replace_parameter(tail, parameter_positions, "num_iterations", str(retained_iterations))
    for name in _PER_FEATURE_PARAMETERS:
        value = _parameter_value(tail, parameter_positions, name)
        if value is None or not value.strip():
            continue
        tokens = [token.strip() for token in value.split(",")]
        if len(tokens) != len(feature_names):
            raise ValueError(f"The LightGBM {name} metadata has the wrong feature count")
        _replace_parameter(
            tail,
            parameter_positions,
            name,
            ",".join(tokens[index] for index in selected_indexes),
        )

    selected_categorical = [
        index_map[index] for index in categorical_indexes if index in selected_index_set
    ]
    _replace_parameter(
        tail,
        parameter_positions,
        "categorical_feature",
        ",".join(str(index) for index in selected_categorical),
    )
    _transform_pandas_categorical(
        tail,
        categorical_indexes=categorical_indexes,
        selected_indexes=selected_index_set,
    )


def _categorical_feature_indexes(
    parameter_value: str | None,
    feature_names: list[str],
    feature_infos: list[str],
) -> tuple[int, ...]:
    if parameter_value and parameter_value.strip():
        value = parameter_value.strip()
        if value.startswith("name:"):
            names = [name.strip() for name in value[5:].split(",") if name.strip()]
            unknown = [name for name in names if name not in feature_names]
            if unknown:
                raise ValueError(
                    f"Unknown categorical features in LightGBM metadata: {', '.join(unknown)}"
                )
            return tuple(index for index, name in enumerate(feature_names) if name in set(names))
        try:
            indexes = tuple(int(token.strip()) for token in value.split(",") if token.strip())
        except ValueError as exc:
            raise ValueError("The LightGBM categorical_feature metadata is invalid") from exc
        if indexes and (min(indexes) < 0 or max(indexes) >= len(feature_names)):
            raise ValueError("The LightGBM categorical_feature metadata has an invalid feature index")
        if len(indexes) != len(set(indexes)):
            raise ValueError("The LightGBM categorical_feature metadata contains duplicates")
        return tuple(sorted(indexes))
    return tuple(
        index
        for index, info in enumerate(feature_infos)
        if info != "none" and not (info.startswith("[") and info.endswith("]"))
    )


def _transform_pandas_categorical(
    tail: list[str],
    *,
    categorical_indexes: tuple[int, ...],
    selected_indexes: set[int],
) -> None:
    positions = [index for index, line in enumerate(tail) if line.startswith("pandas_categorical:")]
    if not positions:
        return
    if len(positions) != 1:
        raise ValueError("The LightGBM model contains duplicate pandas_categorical metadata")
    position = positions[0]
    raw_value = tail[position].rstrip("\n").split(":", 1)[1]
    try:
        metadata: Any = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("The LightGBM pandas_categorical metadata is invalid") from exc
    if metadata is None:
        return
    if not isinstance(metadata, list):
        raise ValueError("The LightGBM pandas_categorical metadata must be a list")
    if not metadata:
        return
    if len(metadata) != len(categorical_indexes):
        raise ValueError(
            "The LightGBM pandas_categorical metadata does not match its categorical features"
        )
    selected_metadata = [
        values
        for feature_index, values in zip(categorical_indexes, metadata)
        if feature_index in selected_indexes
    ]
    tail[position] = "pandas_categorical:" + json.dumps(selected_metadata) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


__all__ = [
    "NoInteractionGroupTreesError",
    "extract_lightgbm_interaction_group",
    "interaction_group_model_filename",
]
