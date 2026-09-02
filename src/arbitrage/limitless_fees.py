"""Conservative implementation of Limitless' published CLOB taker buy curve.

The published table gives discrete points rather than an analytic formula.
For prices between table points this module uses the *higher* neighbouring
fee, so paper profitability is not improved by optimistic interpolation.
Buy fees are paid in outcome tokens, therefore a gross fill of ``q`` shares
leaves ``q * (1-fee_rate)`` usable shares.
"""
from __future__ import annotations

# (upper price bound, conservative buy fee rate)
# Official published schedule as of 2026-08-26.
_BUY_STEPS = [
    (0.50, 0.0300),
    (0.55, 0.0252),
    (0.60, 0.0213),
    (0.65, 0.0180),
    (0.70, 0.0151),
    (0.75, 0.0126),
    (0.80, 0.0105),
    (0.85, 0.0085),
    (0.90, 0.0068),
    (0.95, 0.0053),
    (0.99, 0.0042),
    (0.999, 0.0040),
    (1.0000001, 0.0040),
]


def limitless_buy_fee_rate(price: float) -> float:
    p = max(0.0, min(1.0, float(price)))
    # Conservative step rule: for an in-between price use the fee at the lower
    # probability interval, which is >= the linearly interpolated published fee.
    previous_rate = 0.0300
    previous_bound = 0.0
    for bound, rate in _BUY_STEPS:
        if p <= bound + 1e-12:
            return max(previous_rate, rate) if p > previous_bound + 1e-12 else rate
        previous_bound, previous_rate = bound, rate
    return 0.0040


def net_tokens_after_buy_fee(gross_tokens: float, price: float) -> float:
    return float(gross_tokens) * (1.0 - limitless_buy_fee_rate(price))
