"""V13.0 Presentation Final paper-engine entry point.

This wrapper performs one read-only refresh of the Kalshi open-market cache
before entering the heavily regression-tested V8-derived execution/allocator
implementation.  The refresh closes a coverage hole present in older builds:
live event metadata could contain more open markets than the stale parquet used
as the matcher's left-hand universe.

Paper-only: this module never places or changes an order.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from src.api.markets import get_open_markets
from src.arbitrage.paper_engine_v8 import main as _engine_main


def _default_cache_path() -> Path:
    """Prefer an explicit persistent cache; auto-use mounted Colab Drive."""
    explicit = os.getenv("KALSHI_OPEN_MARKETS_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive / "kalshi-market-cache" / "open_markets.parquet"
    return Path("data/processed/open_markets.parquet")


CACHE_PATH = _default_cache_path()


def refresh_kalshi_open_market_cache(*, fetcher=get_open_markets, cache_path: Path = CACHE_PATH) -> tuple[bool, int]:
    """Atomically refresh the public Kalshi open-market universe.

    A failed refresh never replaces a known-good cache.  If a cache exists we
    continue conservatively and print its age; if no cache exists, execution
    stops because running against an empty/unknown Kalshi universe would make
    discovery diagnostics misleading.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    try:
        markets = fetcher(max_pages=None, limit=1000)
        if not isinstance(markets, pd.DataFrame) or markets.empty or "ticker" not in markets.columns:
            raise RuntimeError("Kalshi refresh returned no usable open markets")
        # Deduplicate defensively before writing the scanner's canonical input.
        markets = markets.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
        markets.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, cache_path)
        print(f"V13 Kalshi cache refresh: {len(markets)} current open markets -> {cache_path}")
        return True, int(len(markets))
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        if not cache_path.exists():
            raise RuntimeError(f"Kalshi cache refresh failed and no fallback cache exists: {exc}") from exc
        try:
            age_hours = max(0.0, (time.time() - cache_path.stat().st_mtime) / 3600.0)
            rows = int(len(pd.read_parquet(cache_path, columns=["ticker"])))
        except Exception:
            age_hours = float("nan")
            rows = -1
        age_text = f"{age_hours:.1f}h" if age_hours == age_hours else "unknown"
        print(
            "V13 Kalshi cache refresh warning: "
            f"{type(exc).__name__}: {exc} | using fallback rows={rows} age={age_text}"
        )
        return False, rows


def main():
    refresh_kalshi_open_market_cache()
    _engine_main()


if __name__ == "__main__":
    main()
