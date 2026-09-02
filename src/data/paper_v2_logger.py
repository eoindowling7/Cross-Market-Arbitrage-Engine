import csv
import json
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path("data/paper_v2_opportunities.csv")


def log_opportunities(rows, path=LOG_PATH):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    fieldnames = [
        "timestamp_utc", "ticker", "subject", "topic", "strategy",
        "quantity", "capital", "gross_profit", "kalshi_fee", "poly_fee",
        "net_profit", "net_per_contract", "return_on_capital",
        "kalshi_price", "poly_avg_price", "poly_worst_price",
        "queue_ahead", "poly_volume24hr", "poly_liquidity",
        "activity_score", "execution_score", "quote_skew_seconds",
        "kalshi_title", "poly_question",
    ]

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            payload = {**row, "timestamp_utc": now}
            writer.writerow(payload)
