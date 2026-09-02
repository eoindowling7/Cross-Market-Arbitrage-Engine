"""Strict short-horizon Limitless × Polymarket paper arbitrage (V8.5).

The lane only accepts threshold-style crypto/stock contracts when asset,
direction, threshold, deadline and explicitly named oracle/source agree.
V8.5 also fixes Limitless discovery pagination (25-row API cap), enriches
candidate markets with exact market metadata when needed, and prices the
published Limitless CLOB taker-buy fee curve conservatively per book level.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.api.limitless_client import (
    get_active_markets as get_limitless_markets,
    get_market_details as get_limitless_market_details,
    get_orderbook as get_limitless_orderbook,
)
from src.api.polymarket_client import parse_token_ids, get_orderbooks
from src.arbitrage.limitless_fees import limitless_buy_fee_rate
from src.arbitrage.polymarket_fees import get_polymarket_fee_rate, polymarket_taker_fee


@dataclass
class LimitlessPolyConfig:
    enabled: bool = True
    max_settlement_days: float = 14.0
    max_pairs_per_scan: int = 75
    max_capital_fraction_per_trade: float = 0.08
    min_profit_dollars: float = 0.05
    min_return_on_capital: float = 0.003
    safety_buffer_per_contract: float = 0.0015
    deadline_tolerance_seconds: float = 180.0
    max_qty: int = 200
    limitless_discovery_limit: int = 500
    enrich_limitless_candidates: int = 120


ASSET_ALIASES = {
    "bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL", "dogecoin": "DOGE", "doge": "DOGE", "xrp": "XRP",
    "chainlink": "LINK", "link": "LINK", "hyperliquid": "HYPE", "hype": "HYPE",
    "tesla": "TSLA", "tsla": "TSLA", "nvidia": "NVDA", "nvda": "NVDA",
    "apple": "AAPL", "aapl": "AAPL", "amazon": "AMZN", "amzn": "AMZN",
    "meta": "META", "microsoft": "MSFT", "msft": "MSFT",
}
SOURCES = ("pyth", "binance", "coinbase", "chainlink", "kraken", "okx", "cf benchmarks", "cme cf")


def _norm(text):
    return " ".join(str(text or "").lower().replace(",", "").split())


def _combined_text(m):
    metadata=m.get("metadata") if isinstance(m.get("metadata"),dict) else {}
    oracle=m.get("oracle") if isinstance(m.get("oracle"),dict) else {}
    fields=(
        m.get("title"),m.get("question"),m.get("proxyTitle"),m.get("description"),
        m.get("resolutionSource"),m.get("priceOracleId"),m.get("price_oracle_id"),
        metadata.get("priceOracleId"),metadata.get("price_oracle_id"),metadata.get("oracle"),
        metadata.get("chainlinkDataStream"),oracle,
    )
    return " ".join(str(x or "") for x in fields)


def _asset(text):
    t = _norm(text).replace("$", " ")
    for alias, canonical in ASSET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return canonical
    return None


def _threshold(text):
    t = str(text or "").replace(",", "")
    m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmM])?", t)
    if not m:
        return None
    value = float(m.group(1)); mult=(m.group(2) or "").lower()
    if mult == "k": value *= 1_000
    if mult == "m": value *= 1_000_000
    return value


def _operator(text):
    t = f" {_norm(text)} "
    if any(x in t for x in (" above ", " over ", " greater than ", " higher than ", " exceed ", " close above ")):
        return "above"
    if any(x in t for x in (" below ", " under ", " less than ", " lower than ", " dip to ", " close below ")):
        return "below"
    return None


def _source(m):
    t=_norm(_combined_text(m))
    hits=[]
    for s in SOURCES:
        if s in t:
            canonical="cf_benchmarks" if s in ("cf benchmarks","cme cf") else s
            if canonical not in hits: hits.append(canonical)
    return hits[0] if len(hits)==1 else None


def _parse_ts(raw):
    if raw is None: return None
    try:
        x=float(raw)
        return x/1000.0 if x>1e11 else x
    except Exception: pass
    try:
        s=str(raw)
        if len(s)==10: s += "T23:59:59+00:00"
        dt=datetime.fromisoformat(s.replace("Z","+00:00"))
        if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception: return None


def _poly_end(m):
    for key in ("endDate","endDateIso","deadline","expirationDate","expirationTimestamp"):
        ts=_parse_ts(m.get(key))
        if ts is not None: return ts
    return None


def _limitless_end(m):
    # Current Limitless docs call this deadline; retain legacy fields too.
    for key in ("deadline","deadlineAt","endAt","expirationTimestamp","expirationDate","endDate"):
        ts=_parse_ts(m.get(key))
        if ts is not None: return ts
    nav=m.get("navigation") if isinstance(m.get("navigation"),dict) else {}
    ts=_parse_ts(nav.get("deadline"))
    return ts


def _trade_type(m):
    return str(m.get("tradeType") or m.get("trade_type") or "").lower()


def _signature(m, venue):
    title=m.get("title") or m.get("question") or m.get("proxyTitle") or ""
    asset=_asset(title); op=_operator(title); threshold=_threshold(title); source=_source(m)
    end=_limitless_end(m) if venue=="limitless" else _poly_end(m)
    if None in (asset,op,threshold,source,end):
        return None
    return asset,op,round(float(threshold),8),source,float(end)


def _pre_signature(m, venue):
    """Signature without source, used to decide which Limitless rows merit detail fetches."""
    title=m.get("title") or m.get("question") or m.get("proxyTitle") or ""
    asset=_asset(title); op=_operator(title); threshold=_threshold(title)
    end=_limitless_end(m) if venue=="limitless" else _poly_end(m)
    if None in (asset,op,threshold,end): return None
    return asset,op,round(float(threshold),8),float(end)


def _enrich_limitless_rows(rows, poly_markets, cfg):
    """Fetch exact details only for rows that could plausibly match a Poly signature."""
    now=time.time(); targets=[]
    poly_pre=[]
    for p in poly_markets:
        ps=_pre_signature(p,"poly")
        if ps and 0 < ps[3]-now <= cfg.max_settlement_days*86400: poly_pre.append(ps)
    if not poly_pre: return rows,0,0

    enriched=[]; attempted=0; succeeded=0
    for lm in rows:
        current=lm
        ls=_pre_signature(lm,"limitless")
        sig=_signature(lm,"limitless")
        plausible=False
        if ls and 0 < ls[3]-now <= cfg.max_settlement_days*86400:
            plausible=any(ls[:3]==ps[:3] and abs(ls[3]-ps[3])<=cfg.deadline_tolerance_seconds for ps in poly_pre)
        if plausible and sig is None and attempted < cfg.enrich_limitless_candidates:
            slug=lm.get("slug")
            if slug:
                attempted += 1
                try:
                    detail=get_limitless_market_details(str(slug))
                    if isinstance(detail,dict):
                        current={**lm,**detail}; succeeded += 1
                except Exception:
                    pass
        enriched.append(current)
    return enriched,attempted,succeeded


def find_exact_pairs(poly_markets, limitless_markets, cfg: LimitlessPolyConfig):
    now=time.time(); out=[]; poly=[]
    for p in poly_markets:
        sig=_signature(p,"poly")
        if sig and 0 < sig[4]-now <= cfg.max_settlement_days*86400: poly.append((p,sig))
    for lm in limitless_markets:
        if _trade_type(lm)!="clob": continue
        ls=_signature(lm,"limitless")
        if not ls or not (0 < ls[4]-now <= cfg.max_settlement_days*86400): continue
        for p,ps in poly:
            if ls[:4]==ps[:4] and abs(ls[4]-ps[4])<=cfg.deadline_tolerance_seconds:
                out.append((lm,p,ls))
                if len(out)>=cfg.max_pairs_per_scan: return out
    return out


def _pair_diagnostics(poly_markets, limitless_markets, cfg):
    now=time.time(); stats={
        "poly_markets_input":len(poly_markets),"limitless_markets":len(limitless_markets),
        "poly_signatures":0,"limitless_clob":0,"limitless_presignatures":0,"limitless_signatures":0,
        "asset_operator_threshold_matches":0,"source_matches":0,"deadline_matches":0,
        "exact_pairs":0,"book_pairs":0,"raw_positive":0,"positive_after_limitless_fee":0,
        "positive_after_fees":0,"qualified":0,"best_near_miss":None,
        "detail_fetch_attempts":0,"detail_fetch_successes":0,"lumy_markets":0,
    }
    poly=[]
    for pm in poly_markets:
        ps=_signature(pm,"poly")
        if ps and 0 < ps[4]-now <= cfg.max_settlement_days*86400:
            stats["poly_signatures"]+=1; poly.append((pm,ps))
    pairs=[]
    for lm in limitless_markets:
        if _trade_type(lm)!="clob": continue
        stats["limitless_clob"]+=1
        pre=_pre_signature(lm,"limitless")
        if pre and 0 < pre[3]-now <= cfg.max_settlement_days*86400: stats["limitless_presignatures"]+=1
        ls=_signature(lm,"limitless")
        if not ls or not (0 < ls[4]-now <= cfg.max_settlement_days*86400): continue
        stats["limitless_signatures"]+=1
        for pm,ps in poly:
            if ls[:3]!=ps[:3]: continue
            stats["asset_operator_threshold_matches"]+=1
            if ls[3]!=ps[3]: continue
            stats["source_matches"]+=1
            if abs(ls[4]-ps[4])>cfg.deadline_tolerance_seconds: continue
            stats["deadline_matches"]+=1; pairs.append((lm,pm,ls))
            if len(pairs)>=cfg.max_pairs_per_scan: break
        if len(pairs)>=cfg.max_pairs_per_scan: break
    stats["exact_pairs"]=len(pairs)
    return pairs,stats


def _levels(rows, transform=None):
    out=[]
    for x in rows or []:
        try:
            p=float(x["price"]); q=float(x["size"])
            if transform: p=transform(p)
            if 0<p<1 and q>0: out.append((p,q))
        except Exception: pass
    return sorted(out)


def _cost(levels, qty):
    rem=float(qty); total=0.0
    for p,s in levels:
        take=min(rem,s); total+=take*p; rem-=take
        if rem<=1e-9: break
    return None if rem>1e-9 else total


def _limitless_buy_cost(levels, net_qty):
    """Collateral needed to receive ``net_qty`` tokens after per-level buy fees."""
    rem=float(net_qty); total=0.0; gross=0.0; weighted_fee=0.0
    for p,size in levels:
        r=limitless_buy_fee_rate(p); net_per_gross=max(1e-9,1-r)
        net_capacity=size*net_per_gross
        take_net=min(rem,net_capacity)
        take_gross=take_net/net_per_gross
        total += take_gross*p; gross += take_gross; weighted_fee += take_gross*r
        rem -= take_net
        if rem<=1e-9: break
    if rem>1e-9: return None
    avg_fee=weighted_fee/max(gross,1e-9)
    return total,gross,avg_fee


def _evaluate_books(lm,pm,sig,cfg,capital_cap, diagnostics_only=False):
    try:
        lb=get_limitless_orderbook(lm["slug"]); ids=parse_token_ids(pm); pb=get_orderbooks(ids)
    except Exception:
        return [],None
    l_yes=_levels(lb.get("asks")); l_no=_levels(lb.get("bids"),lambda p:1-p)
    p_yes=_levels((pb.get(str(ids[0])) or {}).get("asks")); p_no=_levels((pb.get(str(ids[1])) or {}).get("asks"))
    if not all((l_yes,l_no,p_yes,p_no)): return [],None
    out=[]; best_near=None; fee_rate=get_polymarket_fee_rate(pm)
    for name,ll,pl in (("L_YES+P_NO",l_yes,p_no),("L_NO+P_YES",l_no,p_yes)):
        best=None
        for q in range(1,cfg.max_qty+1):
            raw_lc=_cost(ll,q); lfill=_limitless_buy_cost(ll,q); pc=_cost(pl,q)
            if raw_lc is None or lfill is None or pc is None: break
            lc,gross_q,lfee_rate=lfill
            pavg=pc/q; pfee=polymarket_taker_fee(pavg,q,fee_rate)
            raw=q-raw_lc-pc
            after_lfee=q-lc-pc
            after_fees=after_lfee-pfee
            cap=lc+pc+pfee; net=after_fees-cfg.safety_buffer_per_contract*q
            roc=net/max(cap,1e-9)
            near={"limitless_slug":lm.get("slug"),"poly_question":pm.get("question"),"strategy":name,
                  "raw_per_contract":raw/q,"after_limitless_fee_per_contract":after_lfee/q,
                  "poly_fee_per_contract":pfee/q,"limitless_avg_buy_fee":lfee_rate,
                  "net_after_buffer":net/q}
            if best_near is None or near["net_after_buffer"]>best_near["net_after_buffer"]: best_near=near
            if cap>capital_cap: break
            if net>=cfg.min_profit_dollars and roc>=cfg.min_return_on_capital:
                best={"venue":"limitless+polymarket","strategy":name,"quantity":q,"capital":cap,
                      "net_profit":net,"return_on_capital":roc,"limitless_slug":lm.get("slug"),
                      "poly_id":pm.get("id"),"poly_question":pm.get("question"),"asset":sig[0],
                      "operator":sig[1],"threshold":sig[2],"source":sig[3],"settlement_ts":sig[4],
                      "settlement_days":max((sig[4]-time.time())/86400,1/24),"poly_fee":pfee,
                      "limitless_avg_buy_fee":lfee_rate,"limitless_gross_tokens":gross_q}
        if best: out.append(best)
    return out,best_near


def scan_limitless_polymarket(poly_markets, available_cash, bankroll, cfg: LimitlessPolyConfig, *, return_diagnostics=False):
    empty={"poly_markets_input":len(poly_markets),"limitless_markets":0,"poly_signatures":0,"limitless_clob":0,
           "limitless_presignatures":0,"limitless_signatures":0,"asset_operator_threshold_matches":0,
           "source_matches":0,"deadline_matches":0,"exact_pairs":0,"book_pairs":0,"raw_positive":0,
           "positive_after_limitless_fee":0,"positive_after_fees":0,"qualified":0,"best_near_miss":None,
           "detail_fetch_attempts":0,"detail_fetch_successes":0,"lumy_markets":0}
    if not cfg.enabled or available_cash<=0:
        return ([],empty) if return_diagnostics else ([],{"limitless_markets":0,"exact_pairs":0})
    try:
        # Short-horizon oracle-driven (Lumy) markets are the priority. The
        # official API supports automationType=lumy, so reserve discovery
        # capacity for them rather than hoping they appear in a generic page.
        lumy_target=min(cfg.limitless_discovery_limit,350)
        lumy=get_limitless_markets(limit=lumy_target,trade_type="clob",automation_type="lumy",include_next_market=True)
        generic=get_limitless_markets(limit=cfg.limitless_discovery_limit,trade_type="clob")
        lm=[]; seen=set()
        for row in [*lumy,*generic]:
            key=str(row.get("slug") or row.get("id") or "")
            if not key or key in seen: continue
            seen.add(key); lm.append(row)
            if len(lm)>=cfg.limitless_discovery_limit: break
    except Exception as exc:
        empty["error"]=f"{type(exc).__name__}: {exc}"
        return ([],empty) if return_diagnostics else ([],{"limitless_markets":0,"exact_pairs":0,"error":empty["error"]})

    lm,attempts,successes=_enrich_limitless_rows(lm,poly_markets,cfg)
    pairs,stats=_pair_diagnostics(poly_markets,lm,cfg)
    stats["lumy_markets"]=len(lumy)
    stats["detail_fetch_attempts"]=attempts; stats["detail_fetch_successes"]=successes
    cap=min(available_cash,bankroll*cfg.max_capital_fraction_per_trade)
    out=[]; best_near=None
    for a,b,sig in pairs:
        ops,near=_evaluate_books(a,b,sig,cfg,cap)
        if near:
            stats["book_pairs"]+=1
            if near["raw_per_contract"]>0: stats["raw_positive"]+=1
            if near["after_limitless_fee_per_contract"]>0: stats["positive_after_limitless_fee"]+=1
            if near["after_limitless_fee_per_contract"]-near["poly_fee_per_contract"]>0: stats["positive_after_fees"]+=1
            if best_near is None or near["net_after_buffer"]>best_near["net_after_buffer"]: best_near=near
        out.extend(ops)
    stats["best_near_miss"]=best_near; stats["qualified"]=len(out)
    for x in out: x["annualized_return"]=x["return_on_capital"]*365/max(x["settlement_days"],1/24)
    out=sorted(out,key=lambda x:(x["annualized_return"],x["net_profit"]),reverse=True)
    if return_diagnostics: return out,stats
    return out,{"limitless_markets":stats["limitless_markets"],"exact_pairs":stats["exact_pairs"],**({"error":stats["error"]} if "error" in stats else {})}
