from src.arbitrage.execution_utils import consume_asks, choose_maker_price
from src.arbitrage.exact_fees import kalshi_fee


def test_depth_consumption():
    fill = consume_asks(
        [
            {"price": 0.30, "size": 2},
            {"price": 0.31, "size": 3},
        ],
        4,
    )
    assert fill.fully_filled
    assert fill.quantity == 4
    assert abs(fill.cost - 1.22) < 1e-9
    assert abs(fill.average_price - 0.305) < 1e-9
    assert abs(fill.worst_price - 0.31) < 1e-9


def test_dynamic_maker_price():
    assert choose_maker_price(0.18, 0.20, 0.01, True) == 0.19
    assert choose_maker_price(0.18, 0.19, 0.01, True) == 0.18


def test_quadratic_maker_fee_free():
    fee = kalshi_fee(0.5, 10, "quadratic", 1, maker=True)
    assert fee["cash_fee_upper"] == 0.0


if __name__ == "__main__":
    test_depth_consumption()
    test_dynamic_maker_price()
    test_quadratic_maker_fee_free()
    print("paper_engine_v2 offline tests passed")
