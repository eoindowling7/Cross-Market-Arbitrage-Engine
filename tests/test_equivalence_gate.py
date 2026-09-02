from src.arbitrage.equivalence_matcher import deterministic_gate


def test_exact_count_mismatch_is_rejected():
    result = deterministic_gate(
        kalshi_title="Will Democrats win exactly 3 seats?",
        polymarket_question="Will Democrats win exactly 4 seats?",
        ksig={},
        psig={},
    )
    assert result.status == "REJECT"
    assert "exact_count_mismatch" in result.reason


def test_explicit_jurisdiction_mismatch_is_rejected():
    result = deterministic_gate(
        kalshi_title="Will Candidate A win?",
        polymarket_question="Will Candidate A win?",
        ksig={"jurisdiction_region": "US-AZ"},
        psig={"jurisdiction_region": "US-NV"},
    )
    assert result.status == "REJECT"
    assert "jurisdiction_region_mismatch" in result.reason


def test_standard_yes_no_market_can_pass_gate():
    result = deterministic_gate(
        kalshi_title="Will Candidate A win?",
        polymarket_question="Will Candidate A win?",
        ksig={},
        psig={},
        polymarket_market={"outcomes": ["Yes", "No"]},
    )
    assert result.status == "PASS"
