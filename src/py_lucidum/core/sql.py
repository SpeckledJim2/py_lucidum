from __future__ import annotations


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
