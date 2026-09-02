from src.arbitrage import execution_recovery_v28 as r


def _base_candidate():
    return {
        "ticker": "KXTEST-26",
        "kalshi_title": "Will Team Alpha win the 2026 Example Cup?",
        "kalshi_signature": {"domain":"sports","proposition":"winner","subject":"team alpha","competition":"example cup","year":"2026","office_scope":None,"jurisdiction_region":None,"jurisdiction_district":None},
        "polymarket_signature": {"domain":"sports","proposition":"winner","subject":"team alpha","competition":"example cup","year":"2026","office_scope":None,"jurisdiction_region":None,"jurisdiction_district":None},
        "equivalence_certificate": {"resolution_lane":"LOW_BASIS","basis_risk_reserve_per_contract":0.03},
    }


def test_missing_field_is_not_explicit_contradiction():
    ks={"office_scope":"house","jurisdiction_region":"CA"}
    ps={"office_scope":None,"jurisdiction_region":"CA"}
    assert r._explicit_structural_contradictions(ks, ps) == []


def test_observed_mismatch_is_contradiction():
    ks={"office_scope":"house","jurisdiction_region":"CA"}
    ps={"office_scope":"senate","jurisdiction_region":"AZ"}
    reasons=r._explicit_structural_contradictions(ks, ps)
    assert any("office/chamber mismatch" in x for x in reasons)
    assert any("jurisdiction mismatch" in x for x in reasons)


def test_empty_max_classified_as_transient():
    reason, transient = r.classify_exception(ValueError("max() iterable argument is empty"))
    assert reason == "empty_iterable_guarded"
    assert transient is True


def test_late_hydration_can_recover_compatible_rules(monkeypatch):
    c=_base_candidate()
    pm={"id":"123","question":"Will Team Alpha win the 2026 Example Cup?","description":"Official tournament result. Official tournament statistics determine the winner."}
    monkeypatch.setattr(r, "get_market_details", lambda ticker: {
        "title": c["kalshi_title"],
        "rules_primary":"Resolves Yes if Team Alpha wins the 2026 Example Cup according to official tournament statistics."
    })
    monkeypatch.setattr(r, "get_market_by_id", lambda pid: pm)
    state, reason = r.late_hydrate_candidate(c, pm, {}, retries=0)
    assert state == "SAFE"
    assert c["equivalence_certificate"]["resolution_rule_status"] == "LOW_BASIS"
    assert c["equivalence_certificate"]["v28_late_rule_hydrated"] is True


def test_late_hydration_preserves_explicit_contradiction(monkeypatch):
    c=_base_candidate()
    c["kalshi_signature"]["year"]="2026"
    c["polymarket_signature"]["year"]="2027"
    pm={"id":"123","question":"Will Team Alpha win the 2027 Example Cup?"}
    state, reason = r.late_hydrate_candidate(c, pm, {}, retries=0)
    assert state == "CONTRADICTION"
    assert "year mismatch" in reason
