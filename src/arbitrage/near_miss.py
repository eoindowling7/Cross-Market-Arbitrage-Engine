"""Small helpers for ranked paper-trading near-miss diagnostics."""
from __future__ import annotations


def push_top(rows: list[dict], row: dict | None, *, key: str, limit: int = 10) -> None:
    if not row:
        return
    rows.append(dict(row))
    rows.sort(key=lambda x: float(x.get(key, float('-inf'))), reverse=True)
    del rows[max(1, int(limit)):]
