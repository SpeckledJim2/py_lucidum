"""Data-dependent formula checks, with no modelling imports at module load time."""
from __future__ import annotations

import re
from typing import Any


def _has_spline(term: str) -> bool:
    return re.search(r"\b(?:ns|bs|cs)\s*\(", term) is not None


def formula_syntax_message(exc: SyntaxError) -> str:
    expression = str(exc.text or "").strip()
    location = f" in `{expression}`" if expression else ""
    return (
        f"The GLM formula has invalid syntax{location}: {exc.msg}. "
        "Check parentheses and commas. Use == to compare values inside ifelse(...); "
        "a single = is for named arguments such as df=4."
    )


def issue_message(issue: dict[str, Any]) -> str:
    terms = ", ".join(f"`{term}`" for term in issue["terms"])
    related = ", ".join(f"`{term}`" for term in issue["related_terms"])
    return f"{terms}: {issue['reason']}" + (f" Related terms: {related}." if related else "") + f" {issue['suggestion']}"


def blocking_message(report: dict[str, Any]) -> str:
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    prefix = "Training did not save a model: "
    if report["rank"] is not None and report["rank"] < report["columns"]:
        prefix += f"the GLM design matrix is rank-deficient (rank {report['rank']} of {report['columns']}). "
    elif any(issue["code"] == "ill_conditioned" for issue in errors):
        condition = report["condition_number"]
        prefix += "the GLM design matrix is too ill-conditioned for an unpenalized fit "
        prefix += f"(condition number {condition:.3g}). " if condition is not None else "(infinite condition number). "
    return prefix + "GLM check found problems; fitting has not started. " + " ".join(map(issue_message, errors))


def check_terms(
    formula: str,
    frame: Any,
    context: dict[str, Any],
    *,
    estimator: Any,
    training_scope: str,
    penalized: bool,
    condition_limit: float,
    internal_intercept_column: str = "",
    block_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    """Check exactly the supplied fit rows using Glum's formula encoding settings.

    QR reduction uses all rows without retaining a dense rows-by-columns matrix.
    Its compact R preserves the scaled design's singular values and column
    relationships. No Gram matrix is formed (which would square the condition
    number).
    """
    import numpy as np
    import tabmat as tm
    from formulaic.errors import FormulaicError
    from formulaic.parser import DefaultFormulaParser
    from scipy.linalg import qr, svdvals

    issues: list[dict[str, Any]] = []
    report = {
        "ok": True, "training_scope": training_scope, "rows": len(frame),
        "columns": 0, "rank": None, "condition_number": None, "issues": issues,
    }
    severity = "warning" if penalized else "error"

    def add(code: str, terms: list[str], columns: list[str], reason: str, suggestion: str,
            related: list[str] | None = None, level: str | None = None) -> None:
        issues.append({
            "code": code, "severity": level or severity,
            "terms": list(dict.fromkeys(terms)), "columns": columns,
            "related_terms": list(dict.fromkeys(related or [])),
            "reason": reason, "suggestion": suggestion,
        })

    def materialise(formula_terms: Any) -> Any:
        return tm.from_formula(
            formula_terms, frame,
            include_intercept=False, ensure_full_rank=estimator.drop_first,
            categorical_format=estimator.categorical_format,
            cat_missing_method=estimator.cat_missing_method,
            interaction_separator=estimator.interaction_separator,
            add_column_for_intercept=True, context=context,
        )
    materialisation_errors = (FormulaicError, ValueError, TypeError, KeyError, AttributeError)
    parsed_terms = None
    try:
        parsed = DefaultFormulaParser(include_intercept=estimator.fit_intercept).get_terms(formula)
        parsed_terms = getattr(parsed, "rhs", parsed)
        matrix = materialise(parsed_terms)
    except materialisation_errors as exc:
        detail = str(exc)
        match = re.search(r"Unable to evaluate factor `(.*)`\.", detail)
        term = match.group(1) if match else "Formula"
        # Encoding failures can occur after factor evaluation, without Formulaic's
        # named-factor wrapper. Isolate the failing term only on this error path.
        if not match and parsed_terms is not None:
            for candidate in parsed_terms:
                try:
                    materialise([candidate])
                except materialisation_errors:
                    term = str(candidate)
                    break
        reason = f"The expression could not be evaluated: {exc.__cause__ or exc}"
        suggestion = "Correct the named expression or its input values."
        if _has_spline(term) and "distinct knots" in detail:
            reason = "This spline cannot be constructed because its input has too few distinct values on the fitting rows."
            suggestion = "Change the spline input or reduce df; consider C(feature) for a feature with only a few values."
        elif isinstance(exc.__cause__, NameError):
            reason = f"`{term}` refers to a name that is not available in the dataset or formula context."
            suggestion = "Check the feature name and spelling."
        elif isinstance(exc, AttributeError):
            reason = "The formula encoder cannot turn this expression's output into predictor columns."
            suggestion = "Review the transform and the type of values it returns."
            if "is_numeric" in detail and re.search(r"\bC\s*\(", term):
                reason = "This categorical transform returns an array that the formula encoder cannot currently encode."
                suggestion = "For a binary condition, use ifelse(...) directly without C(...); otherwise create a categorical dataset column first."
        add("evaluation", [term], [], reason, suggestion, level="error")
        report["ok"] = False
        return report

    names = ["(Intercept)" if name in {"Intercept", internal_intercept_column} else str(name)
             for name in matrix.model_spec.column_names]
    terms = list(names)
    for term, indices in matrix.model_spec.term_indices.items():
        label = "(Intercept)" if str(term) in {"1", internal_intercept_column, f"`{internal_intercept_column}`"} else str(term)
        for index in indices:
            terms[index] = label
        if not indices:
            add("empty_term", [label], [], "This term produces no predictor columns on the fitting rows.",
                "Remove the term if it is not needed, or review its levels and fitting scope.", level="warning")
    n, p = matrix.shape
    report["columns"] = p
    report["intercept_only"] = names == ["(Intercept)"]
    if not p:
        add("empty_matrix", ["Formula"], [], "The formula produces no predictor columns.",
            "Include an intercept or a predictor that varies.", level="error")
        report["ok"] = False
        return report

    block_rows = max(1, block_bytes // (8 * p))
    first = None
    constant = np.ones(p, dtype=bool)
    finite = np.ones(p, dtype=bool)
    maxima = np.zeros(p)
    for start in range(0, n, block_rows):
        values = np.asarray(matrix[start:start + block_rows, :].toarray(), dtype=float)
        if first is None:
            first = values[0].copy()
        constant &= (values == first).all(axis=0)
        finite &= np.isfinite(values).all(axis=0)
        maxima = np.maximum(maxima, np.max(np.abs(values), axis=0))
    if not finite.all():
        for index in np.flatnonzero(~finite):
            suggestion = "Handle missing values or correct the transform's domain."
            if re.search(r"\bpoly\s*\(", terms[index]):
                suggestion = "Check that the polynomial input varies and reduce its degree; it needs enough distinct input values."
            add("non_finite", [terms[index]], [names[index]],
                "This column contains missing or infinite evaluated values on the fitting rows.",
                suggestion, level="error")
        report["ok"] = False
        return report

    handled: set[int] = set()
    constants = list(np.flatnonzero(constant))
    nonzero_constants = [index for index in constants if first[index] != 0]
    anchor = next((index for index in nonzero_constants if names[index] == "(Intercept)"),
                  next(iter(nonzero_constants), None))
    for index in constants:
        if first[index] == 0:
            add("zero", [terms[index]], [names[index]], f"This column is zero on all {n:,} fitting rows.",
                "Use fewer spline degrees of freedom or change the spline input; it must vary on the fitting rows."
                if _has_spline(terms[index]) else "Remove the term or change its condition or fitting scope.")
            handled.add(index)
        elif index != anchor:
            add("constant", [terms[index]], [names[index]],
                f"This column is constant ({first[index]:.6g}) on all {n:,} fitting rows and duplicates a constant contribution.",
                "Remove or change this term; remove the intercept only if intended."
                if names[anchor] == "(Intercept)" else "Remove or change one of these constant terms.", [terms[anchor]])
            handled.add(index)

    # Scale before QR: finite source values can still overflow a column norm
    # (for example, 50,000 values around 1e307). The first pass above also lets
    # us reject invalid values before invoking any numerical decomposition.
    scales = np.where(maxima > 0, maxima, 1)
    reduced = np.empty((0, p), dtype=float)
    for start in range(0, n, block_rows):
        values = np.asarray(matrix[start:start + block_rows, :].toarray(), dtype=float)
        values = values / scales
        reduced = qr(np.vstack((reduced, values)), mode="r", check_finite=False, overwrite_a=True)[0][:p].copy()
    # Restore relative source-column scales for the existing unscaled condition
    # policy. A single global divisor preserves condition without overflowing.
    relative_scales = maxima / max(float(maxima.max()), np.finfo(float).smallest_subnormal)
    singular = svdvals(reduced * relative_scales, check_finite=False)
    condition = float(singular[0] / singular[-1]) if n >= p and singular[-1] > 0 else float("inf")
    report["condition_number"] = condition if np.isfinite(condition) else None
    norms = np.linalg.norm(reduced, axis=0)
    normalised = reduced / np.where(norms > 0, norms, 1)
    scaled_singular = svdvals(normalised, check_finite=False)
    tolerance = float(np.finfo(float).eps * max(n, p) * scaled_singular[0])
    rank = int((scaled_singular > tolerance).sum())
    report.update(rank=rank, rank_tolerance=tolerance)
    candidates = [index for index in range(p) if index not in handled]
    identified = len(handled)
    if rank < len(candidates):
        _, pivots = qr(normalised[:, candidates], mode="r", pivoting=True, check_finite=False)
        independent = [candidates[index] for index in pivots[:rank]]
        for pivot in pivots[rank:]:
            index = candidates[pivot]
            coefficients = np.linalg.lstsq(normalised[:, independent], normalised[:, index],
                                           rcond=np.finfo(float).eps * max(n, p))[0]
            # Prune insignificant contributors only if the smaller relationship
            # still explains the column to the rank tolerance.
            selected = np.flatnonzero(np.abs(coefficients) > tolerance)
            related_indices = [independent[i] for i in selected]
            residual = normalised[:, index] - normalised[:, related_indices] @ coefficients[selected]
            if np.linalg.norm(residual) > tolerance:
                related_indices = independent
                residual = normalised[:, index] - normalised[:, independent] @ coefficients
            if np.linalg.norm(residual) > tolerance:
                continue
            code = "dependent"
            reason = "This column is numerically dependent on the related terms on the fitting rows."
            if len(related_indices) == 1:
                other = related_indices[0]
                duplicate = True
                for start in range(0, n, block_rows):
                    pair = matrix[start:start + block_rows, :].toarray()[:, [index, other]]
                    if not np.array_equal(pair[:, 0], pair[:, 1]):
                        duplicate = False
                        break
                if duplicate:
                    code = "duplicate"
                    reason = f"This column duplicates `{names[other]}` on all {n:,} fitting rows."
                else:
                    code = "proportional"
                    reason = "This column is numerically proportional to the related column on the fitting rows."
            suggestion = "Remove or change one of the related terms."
            if len(related_indices) == 1 and names[related_indices[0]] == "(Intercept)":
                suggestion += " For a nearly constant predictor, consider centering it by subtracting a suitable reference value."
            related_splines = [terms[other] for other in [index, *related_indices] if _has_spline(terms[other])]
            if any(re.search(r"\b(?:ns|cs)\s*\(", term) and "constraints" not in term for term in related_splines):
                suggestion += ' For an unconstrained natural or cyclic spline, consider constraints="center" or an appropriate no-intercept formula.'
            elif related_splines:
                suggestion += " Check for overlap between spline bases and other predictors, or reduce spline df."
            add(code, [terms[index]], [names[index]], reason, suggestion,
                [terms[other] for other in related_indices])
            identified += 1

    if identified < p - rank:
        add("dependent", list(dict.fromkeys(terms)), names,
            f"These terms jointly produce a rank-deficient matrix (rank {rank} of {p}).",
            "Simplify the related terms or use regularisation.")
    if rank == p and (not np.isfinite(condition) or condition > condition_limit):
        add("ill_conditioned", list(dict.fromkeys(terms)), [],
            "The matrix is too ill-conditioned for reliable unregularised fitting; predictor scales or near-dependencies may be responsible.",
            "Rescale predictors, simplify near-dependent terms, or use ridge/auto regularisation.")
    report["ok"] = not any(issue["severity"] == "error" for issue in issues)
    return report
