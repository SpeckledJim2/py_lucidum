from __future__ import annotations

# Compatibility re-exports for code that imported the original monolithic query module.
from .core import (
    ColumnInfo,
    Dataset as CoreDataset,
    infer_kind,
    is_numeric_kind,
    json_number,
    parse_positive_float,
    quote_ident,
    sql_literal,
    suggested_band_width,
)
from .core.schema import nice_band_steps
from .tools.line_bar.query import __all__ as _line_bar_exports
from .tools.line_bar.query import *  # noqa: F403


class Dataset(CoreDataset):
    def chart(self, request):
        from .tools.line_bar.query import chart

        return chart(self, request)


__all__ = [
    "ColumnInfo",
    "Dataset",
    "infer_kind",
    "is_numeric_kind",
    "json_number",
    "nice_band_steps",
    "parse_positive_float",
    "quote_ident",
    "sql_literal",
    "suggested_band_width",
] + list(_line_bar_exports)
