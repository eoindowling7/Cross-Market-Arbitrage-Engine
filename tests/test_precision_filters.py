import production_veto_v7 as v7
import production_veto_v8_precision as v8
import production_veto_v8_1_precision as v81


def test_v7_dependency_chain_loads():
    assert v7.norm("Tour de France") == "tour de france"


def test_v8_exact_title_is_positive_proof():
    row = {
        "kalshi_title": "Will Ajax win the Eredivisie?",
        "poly_question": "Will Ajax win the Eredivisie?",
    }
    ksig = {"subject": "ajax", "event_identity": "eredivisie"}
    psig = {"subject": "ajax", "event_identity": "eredivisie"}
    result = v8.pair_decision(row, ksig, psig, {})
    assert result["decision"] == "PASS"
    assert result["reason"] == "PROOF_EXACT_TITLE"


def test_v81_preserves_exact_title_pass():
    row = {
        "kalshi_title": "Will Ajax win the Eredivisie?",
        "poly_question": "Will Ajax win the Eredivisie?",
    }
    ksig = {"subject": "ajax", "event_identity": "eredivisie"}
    psig = {"subject": "ajax", "event_identity": "eredivisie"}
    result = v81.pair_decision(row, ksig, psig, {})
    assert result["decision"] == "PASS"
