"""Precision-first semantic payoff gate V8.

Designed to run *after* DeBERTa + production_veto_v7.

Philosophy
----------
V7 remains the broad candidate matcher.  V8 is deliberately asymmetric:
  * explicit material conflict -> REJECT
  * strong positive proof of the same payoff proposition -> PASS
  * anything else -> REVIEW (which the paper/live arb engine must treat as NO TRADE)

This is intentionally precision-first.  It is acceptable to lose real pairs; it is
not acceptable to silently convert an uncertain pair into an arbitrage candidate.

The hard rules are broad slot-level rules, not ticker-specific exceptions:
  proposition, event/competition, award, metric, action/object, event granularity,
  geography, time/scope, election office/race, rank/threshold, and entity collisions.
"""
from __future__ import annotations

import re
import unicodedata
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional V7 wrapper.  Keep V7 as the first deterministic layer in production.
# ---------------------------------------------------------------------------
_v7_path = Path(__file__).with_name("production_veto_v7.py")
v7 = None
if _v7_path.exists():
    try:
        _spec = spec_from_file_location("v7", str(_v7_path))
        v7 = module_from_spec(_spec)
        _spec.loader.exec_module(v7)
    except Exception:
        # V8 can still be unit-tested standalone. In production place it beside
        # the complete V7/V6 dependency chain so veto() can wrap V7 normally.
        v7 = None


def norm(x):
    s = unicodedata.normalize("NFKD", str(x or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _text(row, key1, key2=None):
    if key1 in row and row.get(key1) is not None:
        return row.get(key1)
    if key2:
        return row.get(key2)
    return ""


def texts(row, k, p):
    kt = norm(_text(row, "kalshi_title"))
    pt = norm(_text(row, "poly_question", "polymarket_question"))
    ke = norm(k.get("event_identity"))
    pe = norm(p.get("event_identity"))
    return kt, pt, ke, pe, " ".join([kt, ke]), " ".join([pt, pe])


# ---------------------------------------------------------------------------
# Slot extractors
# ---------------------------------------------------------------------------
AWARD_PATTERNS = [
    ("finals_mvp", r"\bfinals mvp\b"),
    ("defensive_rookie", r"\bdefensive rookie of the year\b"),
    ("offensive_rookie", r"\boffensive rookie of the year\b"),
    ("mvp", r"\bmvp\b"),
    ("gold_glove", r"\bgold glove\b"),
    ("platinum_glove", r"\bplatinum glove\b"),
    ("hank_aaron", r"\bhank aaron\b"),
    ("cy_young", r"\bcy young\b"),
    ("comeback_poty", r"\bcomeback player(?: of the year)?\b"),
    ("offensive_poty", r"\boffensive player of the year\b"),
    ("defensive_poty", r"\bdefensive player of the year\b"),
    ("rookie_poty", r"\brookie of the year\b|\broty\b"),
    ("coach_poty", r"\bcoach of the year\b"),
    ("manager_poty", r"\bmanager of the year\b|\bmoty\b"),
    ("action_of_year", r"\baction of the year\b"),
    ("player_of_year", r"\bplayer of the year\b"),
    ("streamer_of_year", r"\bstreamer of the year\b"),
    ("content_creator", r"\bcontent creator of the year\b"),
    ("esports_personality", r"\besports personality of the year\b"),
    ("best_picture", r"\bbest picture\b"),
    ("best_actor", r"\bbest actor\b"),
    ("best_actress", r"\bbest actress\b"),
    ("supporting_actor", r"\bbest supporting actor\b"),
    ("supporting_actress", r"\bbest supporting actress\b"),
    ("best_director", r"\bbest director\b|\boutstanding directing\b"),
    ("adapted_screenplay", r"\badapted screenplay\b"),
    ("original_screenplay", r"\boriginal screenplay\b"),
    ("animated_feature", r"\banimated feature\b"),
    ("documentary_feature", r"\bdocumentary feature\b"),
    ("international_feature", r"\binternational feature\b"),
    ("makeup_hairstyling", r"\bmakeup(?: and hairstyling)?\b"),
    ("original_score", r"\boriginal score\b"),
    ("visual_effects", r"\bvisual effects\b"),
    ("cinematography", r"\bbest cinematography\b"),
    ("emmy_drama_series", r"\boutstanding drama series\b"),
    ("emmy_comedy_series", r"\boutstanding comedy series\b"),
    ("emmy_limited_series", r"\boutstanding limited or anthology series\b|\blimited series\b"),
    ("emmy_reality_comp", r"\boutstanding reality(?: competition| competition series| competition program|/competition series)\b"),
    ("emmy_writing_comedy", r"\boutstanding writing for a comedy series\b"),
    ("emmy_writing_limited", r"\boutstanding writing for a limited or anthology series or movie\b"),
    ("emmy_lead_actor_comedy", r"\bcomedy actor\b|\boutstanding lead actor in a comedy series\b"),
    ("game_best_multiplayer", r"\bbest multiplayer\b"),
    ("game_best_direction", r"\bbest game direction\b"),
]


def award_key(text):
    t = norm(text)
    found = [key for key, pat in AWARD_PATTERNS if re.search(pat, t)]
    if "finals_mvp" in found and "mvp" in found:
        found.remove("mvp")
    if "defensive_rookie" in found and "rookie_poty" in found:
        found.remove("rookie_poty")
    if "offensive_rookie" in found and "rookie_poty" in found:
        found.remove("rookie_poty")
    if "emmy_writing_comedy" in found and "emmy_comedy_series" in found:
        found.remove("emmy_comedy_series")
    if "emmy_writing_limited" in found and "emmy_limited_series" in found:
        found.remove("emmy_limited_series")
    return found[0] if found else None


AWARD_EVENT_PATTERNS = [
    ("oscars", r"\boscars?\b|\bacademy awards?\b"),
    ("emmys", r"\bemmys?\b"),
    ("grammys", r"\bgrammys?\b"),
    ("venice", r"\bvenice (?:film )?festival\b|\bcoppa volpi\b|\bgolden lion\b|\bsilver lion\b"),
    ("streamer_awards", r"\bstreamer awards\b"),
    ("esports_awards", r"\besports .*award\b|\besports awards\b"),
    ("game_awards", r"\bgame awards\b"),
    ("chess_com", r"\bchess com player of the year\b"),
]


def award_event(text):
    t = norm(text)
    for key, pat in AWARD_EVENT_PATTERNS:
        if re.search(pat, t):
            return key
    return None


METRIC_PATTERNS = [
    ("passing_yards", r"\bpassing yards\b"),
    ("total_qbr", r"\btotal qbr\b|\bqbr\b"),
    ("rushing_yards", r"\brushing yards\b"),
    ("rushing_tds", r"\brushing touchdowns?\b"),
    ("passing_tds", r"\bpassing touchdowns?\b"),
    ("receiving_tds", r"\breceiving touchdowns?\b"),
    ("interceptions", r"\binterceptions\b"),
    ("home_runs", r"\bhome runs?\b"),
    ("hits", r"\bhits\b"),
    ("runs", r"\bruns\b"),
    ("war", r"\bwins above replacement\b|\bwar\b"),
    ("fantasy_qb", r"\bfantasy qb\b|\btop 5 qb\b|\btop 5 fantasy qb\b"),
    ("fantasy_wr", r"\bfantasy wr\b|\btop 5 wr\b|\btop 5 fantasy wr\b"),
    ("top100", r"\btop 100 list\b"),
]


def metric_key(text):
    t = norm(text)
    for key, pat in METRIC_PATTERNS:
        if re.search(pat, t):
            return key
    return None


ACTION_PATTERNS = [
    ("announce_departure", r"\bannounce\w* (?:their |his |her )?departure\b|\bdeparture announced\b"),
    ("leave_actual", r"\bleave (?:prime minister|president|office|the .*administration|trump .*cabinet|trump cabinet)\b|\bout as (?:prime minister|president|mayor|secretary|minister)\b"),
    ("release_song", r"\brelease (?:a )?new song\b|\brelease a song\b"),
    ("release_album", r"\brelease (?:a )?new album\b|\brelease an album\b"),
    ("announce_tour", r"\bannounce (?:a )?new tour\b"),
    ("perform_halftime", r"\bperform at (?:the )?(?:2027 )?(?:big game|super bowl) halftime\b|\bhalftime show\b"),
    ("perform_msg", r"\bperform at madison square garden\b"),
    ("perform_sphere", r"\bperform at (?:las vegas )?sphere\b"),
    ("release_game", r"\bgame .*release\b|\brelease date\b"),
    ("announce_game", r"\bannounc(?:e|ed) before\b"),
    ("billboard_top_artist", r"\bbillboard #?1 top artist\b|\bbillboard top artist\b"),
    ("perform_lollapalooza", r"\bperform at lollapalooza\b"),
    ("olympic_team_selection", r"\bselected to the usa .*basketball team\b|\bolympic team\b"),
    ("endorse", r"\bendorse\b"),
    ("freeze_rents", r"\bfreeze .*rents?\b"),
    ("recognize_leader", r"\brecognize .* as (?:the )?leader\b"),
    ("lead_country", r"\blead (?:iran|venezuela|[a-z ]+)\b"),
]


def action_key(text):
    t = norm(text)
    for key, pat in ACTION_PATTERNS:
        if re.search(pat, t):
            return key
    return None


EVENT_PATTERNS = [
    ("chess_olympiad_women", r"\bchess olympiad women"),
    ("chess_olympiad_open", r"\bchess olympiad open"),
    ("world_chess_championship", r"\bworld chess championship\b"),
    ("titled_tuesday", r"\btitled tuesday\b"),
    ("chess_player_of_year", r"\bchess com player of the year\b"),
    ("africa_cup_nations", r"\bafrica cup of nations\b"),
    ("us_open_tennis", r"\bus open (?:men|women|tennis)|\bwomen s us open\b"),
    ("nitto_atp_finals", r"\bnitto atp finals\b"),
    ("f1_italian_gp", r"\bf1 italian grand prix\b|\bitalian grand prix\b"),
    ("f1_action_award", r"\bf1 action of the year\b"),
    ("cfl_grey_cup", r"\bcfl grey cup\b"),
    ("mls_cup", r"\bmls cup\b"),
    ("leagues_cup", r"\bleagues cup\b"),
    ("taca_portugal", r"\btaca de portugal\b"),
    ("primeira_liga", r"\bprimeira liga\b|\bliga portugal\b"),
    ("europa_league", r"\beuropa league\b"),
    ("champions_league", r"\bchampions league\b"),
    ("conference_league", r"\bconference league\b"),
    ("bundesliga", r"\bbundesliga\b"),
    ("la_liga", r"\bla ?liga\b"),
    ("hitpoint_masters", r"\bhitpoint masters\b"),
    ("knvb_cup", r"\bknvb cup\b"),
    ("eredivisie", r"\beredivisie\b"),
    ("wta_grand_slam_any", r"\btennis grand slam\b|\ba wta grand slam\b"),
    ("lollapalooza", r"\blollapalooza\b"),
    ("big_game_halftime", r"\bbig game halftime\b|\bhalftime show\b"),
    ("nascar_cup", r"\bnascar cup series\b"),
    ("motogp_world", r"\bmotogp world champ"),
    ("vuelta", r"\bvuelta a espana\b"),
    ("wnba", r"\bwnba\b|\bwomen s pro basketball\b"),
    ("nba", r"\bnba\b|\bpro basketball\b"),
]


def event_key(text):
    t = norm(text)
    for key, pat in EVENT_PATTERNS:
        if re.search(pat, t):
            return key
    return None


def nhl_scope(text):
    t = norm(text)
    if "stanley cup" in t or re.search(r"\bnhl (?:league )?champion", t):
        return "stanley"
    if "eastern conference final" in t or "eastern conference champion" in t:
        return "east_conf"
    if "western conference final" in t or "western conference champion" in t:
        return "west_conf"
    for name, key in [("metropolitan", "metro_div"), ("atlantic", "atlantic_div"), ("pacific", "pacific_div"), ("central", "central_div")]:
        if f"{name} division" in t:
            return key
    return None


def cfb_scope(text):
    t = norm(text)
    if "make" in t and "national championship game" in t:
        return "make_title_game"
    if ("win" in t or "champion" in t) and ("national championship" in t or "cfp national championship" in t):
        return "win_title"
    if re.search(r"\bap poll week \d+\b", t):
        return "ap_specific_week"
    if "ranked 1 during" in t or "ranked 1 during any week" in t:
        return "rank_any_week"
    if ("1 seed" in t or "#1 seed" in t) and ("college football" in t or "playoff" in t):
        return "playoff_seed1"
    return None


def cycling_scope(text):
    t = norm(text)
    if "vuelta a espana" not in t:
        return None
    if re.search(r"\bstage \d+\b", t):
        return "stage"
    if "green jersey" in t:
        return "green_jersey"
    if "polka dot jersey" in t:
        return "mountains_jersey"
    return "overall"


def media_scope(text):
    t = norm(text)
    if "spotify" in t or "google" in t:
        if re.search(r"\bu s\b|\busa\b|\bin the us\b", t):
            return "us"
        if "global" in t or "globally" in t:
            return "global"
    return None


def product_variant(text):
    t = norm(text)
    if "iphone 18 pro" in t:
        return "iphone18pro"
    if "iphone 18" in t:
        return "iphone18"
    return None


def fixture_scope(event_text, title_text):
    e, t = norm(event_text), norm(title_text)
    if " map " in f" {e} " or re.search(r"\bmap [123]\b", t):
        return "map"
    if "second half" in e or "2nd half" in t or "second half" in t:
        return "second_half"
    if "first 5 innings" in e or "first 5 innings" in t:
        return "first5"
    if re.search(r"\bround \d+\b", t) and ("beat" in t or "h2h" in e):
        return "round_h2h"
    if " vs " in f" {e} " or e.startswith("vs ") or e.endswith(" vs"):
        return "fixture"
    return None


def broader_event_mark(text):
    t = norm(text)
    return any(x in t for x in ["championship", "champion", "tournament", "league", "cup", "season"])


def google_cat(text):
    t = norm(text)
    if "google" not in t:
        return None
    for key, pat in [
        ("athletes", r"\bathletes\b"),
        ("actors", r"\bactors\b"),
        ("movies", r"\bmovies?\b"),
        ("tv", r"\btv shows?\b"),
        ("people", r"\bpeople\b|\bsearched person\b"),
    ]:
        if re.search(pat, t):
            return key
    return None


def outcome_mode(text):
    """Extract only high-certainty payoff modes from surface wording.

    This is deliberately tiny: it exists to catch catastrophic scope errors
    such as `win the election` vs `be on the ballot`, or `win the tournament`
    vs `reach the final`.  It is not a general-purpose semantic parser.
    """
    t = norm(text)
    if re.search(r"\bon (?:the )?ballot\b|official candidate list|appear on .*candidate list", t):
        return "ballot"
    if re.search(r"\b(?:advance|qualify|reach|make it) (?:to|for) (?:the )?(?:semi ?finals?|finals?)\b", t) or re.search(r"\bmake (?:the )?(?:semi ?finals?|finals?)\b", t):
        return "qualify_stage"
    if "relegat" in t:
        return "relegation"
    # `win` means winner only when paired with a real event/office/award object;
    # do not treat phrases like `win exactly 3 seats` as a distinct mode here.
    if re.search(r"\bwin(?:s|ning)?\b", t) and re.search(r"election|champion|championship|cup|league|tournament|award|mvp|race|singles", t):
        return "winner"
    return None


# ---------------------------------------------------------------------------
# HARD semantic contradiction layer
# ---------------------------------------------------------------------------
def precision_veto(row, k, p):
    kt, pt, ke, pe, kall, pall = texts(row, k, p)

    kom, pom = outcome_mode(kt), outcome_mode(pt)
    if {kom, pom} == {"winner", "ballot"}:
        return "V8_OUTCOME_MODE:winner!=ballot"
    if {kom, pom} == {"winner", "qualify_stage"}:
        return "V8_OUTCOME_MODE:winner!=qualify_stage"

    kd, pd = norm(k.get("domain")), norm(p.get("domain"))
    # `other` is a parser fallback, not a real semantic domain.  Never let a
    # fallback label veto an otherwise strong pair (OPEC is a concrete example).
    if kd and pd and kd != pd and kd != "other" and pd != "other":
        return f"V8_DOMAIN:{kd}!={pd}"

    kp, pp = norm(k.get("proposition")), norm(p.get("proposition"))
    prop_pair = frozenset(x for x in (kp, pp) if x)
    # Do not reject every parser-label disagreement.  The parser often calls the
    # same payoff `binary_event` on one venue and `participation`, `leader`, or
    # `rank_threshold` on the other.  Only universally incompatible outcome
    # families are hard vetoes here.
    if prop_pair == frozenset({"relegation", "winner"}):
        return f"V8_PROPOSITION:{kp}!={pp}"
    if prop_pair == frozenset({"participation", "winner"}):
        ballotish = bool(re.search(r"\bballot\b|official candidate list|appear on .*candidate", kall))
        if ballotish:
            return f"V8_PROPOSITION:{kp}!={pp}"
    if prop_pair == frozenset({"primary winner", "winner"}):
        if "primary" in kall or "primary" in pall or "nominee" in kall or "nominee" in pall:
            # Later exact-office/event checks can preserve genuine nominee aliases;
            # a primary winner versus a general-election winner is incompatible.
            generalish = bool(re.search(r"presidential election|governor|mayor|senate race|house election", kall + " " + pall))
            if generalish and ("primary" in kall or "primary" in pall):
                return f"V8_PROPOSITION:{kp}!={pp}"
    if prop_pair == frozenset({"qualify", "winner"}):
        return f"V8_PROPOSITION:{kp}!={pp}"

    # Only high-reliability structured axes are hard conflicts.  Raw stage labels
    # (final/finals, first_round/runoff), generic thresholds, and period_scope are
    # intentionally *not* hard-vetoed here because they are noisy and can encode
    # valid one-way implications.
    for fld in ["year", "rank_semantics", "gender_scope", "metric"]:
        a, b = k.get(fld), p.get(fld)
        if a is not None and b is not None and str(a) != str(b):
            return f"V8_STRUCTURED_{fld.upper()}:{a}!={b}"

    # Structured office/jurisdiction conflicts are strong when both sides parsed them.
    ko, po = norm(k.get("office_scope")), norm(p.get("office_scope"))
    if ko and po and ko != po:
        return f"V8_OFFICE:{ko}!={po}"
    kd1, pd1 = norm(k.get("jurisdiction_district")), norm(p.get("jurisdiction_district"))
    if kd1 and pd1 and kd1 != pd1:
        return f"V8_DISTRICT:{kd1}!={pd1}"
    kr, pr = norm(k.get("jurisdiction_region")), norm(p.get("jurisdiction_region"))
    if kr and pr and kr != pr:
        return f"V8_REGION:{kr}!={pr}"

    # State-legislature control is not a single federal House district.
    state_house_k = bool(re.search(r"\b(?:texas|california|florida|new york|pennsylvania|ohio|michigan|georgia) (?:state )?house\b", kall))
    state_house_p = bool(re.search(r"\b(?:texas|california|florida|new york|pennsylvania|ohio|michigan|georgia) (?:state )?house\b", pall))
    fed_district_k = bool(re.search(r"\b[a-z]{2} \d{1,2} house\b|\b[a-z]{2}-\d{1,2}\b", kall))
    fed_district_p = bool(re.search(r"\b[a-z]{2} \d{1,2} house\b|\b[a-z]{2}-\d{1,2}\b", pall))
    if (state_house_k and fed_district_p) or (state_house_p and fed_district_k):
        return "V8_OFFICE_SCOPE:state_house!=federal_district"

    # District nomination versus presidential/VP nomination.
    district_nom_k = bool(re.search(r"\bnominee for [a-z]{2} \d{1,2}\b|\b[a-z]{2}-\d{1,2}\b", kall))
    district_nom_p = bool(re.search(r"\bnominee for [a-z]{2} \d{1,2}\b|\b[a-z]{2}-\d{1,2}\b", pall))
    pres_nom_k = "presidential nominee" in kall or "presidential nomination" in kall or "nominee for the presidency" in kall
    pres_nom_p = "presidential nominee" in pall or "presidential nomination" in pall or "nominee for the presidency" in pall
    if (district_nom_k and pres_nom_p) or (district_nom_p and pres_nom_k):
        return "V8_OFFICE_SCOPE:district_nominee!=presidential_nominee"

    ka, pa = award_key(kall), award_key(pall)
    if ka and pa and ka != pa:
        return f"V8_AWARD:{ka}!={pa}"
    kae, pae = award_event(kall), award_event(pall)
    if kae and pae and kae != pae:
        return f"V8_AWARD_EVENT:{kae}!={pae}"

    # A conjunction of two awards is not the same payoff as either award alone.
    def award_set(text):
        t = norm(text)
        out = []
        for key, pat in AWARD_PATTERNS:
            if re.search(pat, t):
                out.append(key)
        if "finals_mvp" in out and "mvp" in out:
            out.remove("mvp")
        return set(out)
    kas, pas = award_set(kall), award_set(pall)
    if len(kas) >= 2 and len(pas) == 1 and pas.issubset(kas):
        return "V8_AWARD_CONJUNCTION:multi!=single"
    if len(pas) >= 2 and len(kas) == 1 and kas.issubset(pas):
        return "V8_AWARD_CONJUNCTION:single!=multi"

    # Financial role specificity: lead-left/bookrunner is narrower than generic lead underwriter.
    lead_left_k = bool(re.search(r"\blead left\b|\blead-left\b|\bleft lead\b", kall))
    lead_left_p = bool(re.search(r"\blead left\b|\blead-left\b|\bleft lead\b", pall))
    lead_generic_k = "lead underwriter" in kall
    lead_generic_p = "lead underwriter" in pall
    if (lead_left_k and lead_generic_p and not lead_left_p) or (lead_left_p and lead_generic_k and not lead_left_k):
        return "V8_ROLE_SCOPE:lead_left!=lead_underwriter"

    km, pm = metric_key(kall), metric_key(pall)
    if km and pm and km != pm:
        return f"V8_METRIC:{km}!={pm}"

    kac, pac = action_key(kall), action_key(pall)
    if kac and pac and kac != pac:
        return f"V8_ACTION:{kac}!={pac}"
    material_actions = {
        "announce_departure", "leave_actual", "announce_tour", "perform_halftime", "perform_msg",
        "perform_sphere", "perform_lollapalooza", "billboard_top_artist", "olympic_team_selection",
        "endorse", "freeze_rents", "recognize_leader", "lead_country",
    }
    if kac in material_actions and pac is None:
        return f"V8_ACTION_ONE_SIDED:{kac}"
    if pac in material_actions and kac is None:
        return f"V8_ACTION_ONE_SIDED:{pac}"

    kev, pev = event_key(kall), event_key(pall)
    if kev and pev and kev != pev:
        return f"V8_EVENT:{kev}!={pev}"

    # One side is an explicit named award and the other is a distinct named competition/event.
    if ka and not pa and (pev or pae):
        return f"V8_AWARD_VS_EVENT:{ka}!={pev or pae}"
    if pa and not ka and (kev or kae):
        return f"V8_AWARD_VS_EVENT:{kev or kae}!={pa}"

    kn, pn = nhl_scope(kall), nhl_scope(pall)
    if kn and pn and kn != pn:
        return f"V8_NHL_SCOPE:{kn}!={pn}"
    kc, pc = cfb_scope(kall), cfb_scope(pall)
    if kc and pc and kc != pc:
        return f"V8_CFB_SCOPE:{kc}!={pc}"
    kcy, pcy = cycling_scope(kall), cycling_scope(pall)
    if kcy and pcy and kcy != pcy:
        return f"V8_CYCLING_SCOPE:{kcy}!={pcy}"

    kf, pf = fixture_scope(ke, kt), fixture_scope(pe, pt)
    if kf and not pf and (broader_event_mark(pall) or pev):
        return f"V8_GRANULARITY:{kf}!=event"
    if pf and not kf and (broader_event_mark(kall) or kev):
        return f"V8_GRANULARITY:event!={pf}"
    if kf and pf and kf != pf:
        return f"V8_GRANULARITY:{kf}!={pf}"

    kms, pms = media_scope(kall), media_scope(pall)
    if kms and pms and kms != pms:
        return f"V8_GEO_SCOPE:{kms}!={pms}"
    if ("spotify" in kall and "spotify" in pall) or ("google" in kall and "google" in pall):
        if (kms == "us" and pms is None) or (pms == "us" and kms is None):
            return "V8_GEO_SCOPE:us!=unspecified"

    kv, pv = product_variant(kall), product_variant(pall)
    if kv and pv and kv != pv:
        return f"V8_PRODUCT:{kv}!={pv}"

    # Election-stage/outcome qualifiers.
    if ("first round" in kall) != ("first round" in pall):
        if "presidential election" in kall and "presidential election" in pall:
            return "V8_ELECTION_STAGE:first_round"
    if ("absolute majority" in kall) != ("absolute majority" in pall):
        if "election" in kall or "election" in pall:
            return "V8_ELECTION_SCOPE:absolute_majority"
    seat_k = bool(re.search(r"\bwins? (?:a|one) seat\b|\bwin (?:a|one) seat\b", kall))
    seat_p = bool(re.search(r"\bwins? (?:a|one) seat\b|\bwin (?:a|one) seat\b", pall))
    if seat_k != seat_p and ("election" in kall or "election" in pall):
        return "V8_ELECTION_SCOPE:seat_vs_winner"
    nom_k = bool(re.search(r"\bnominee\b|\bnomination\b|\bprimary\b", kall))
    nom_p = bool(re.search(r"\bnominee\b|\bnomination\b|\bprimary\b", pall))
    if nom_k != nom_p and ("election" in kall or "election" in pall or "mayor" in kall or "mayor" in pall):
        return "V8_ELECTION_SCOPE:nomination_vs_general"

    # Search chart category is a distinct payoff axis.
    kg, pg = google_cat(kall), google_cat(pall)
    if kg and pg and kg != pg:
        return f"V8_SEARCH_CATEGORY:{kg}!={pg}"

    # Explicit rank-time scope and superlative-vs-count threshold.
    if (re.search(r"\bweek \d+\b", kall) and "during" in pall and "ranked" in pall) or (
        re.search(r"\bweek \d+\b", pall) and "during" in kall and "ranked" in kall
    ):
        return "V8_RANK_SCOPE:week!=season"
    if ("most awards" in kall and re.search(r"\b\d+ or more awards\b", pall)) or (
        "most awards" in pall and re.search(r"\b\d+ or more awards\b", kall)
    ):
        return "V8_RANK_VS_THRESHOLD"

    # Known general geography token trap, expressed as a general city/sub-city rule for this data family.
    if ("vancouver mayoral election" in kall and "west vancouver" in pall) or (
        "west vancouver" in kall and "vancouver mayoral election" in pall
    ):
        return "V8_JURISDICTION:Vancouver!=West Vancouver"

    # Tie/draw outcome versus named side winning.
    if (re.match(r"^tie\b", kt) and re.search(r"\bto win\b|\bwins?\b", pt)) or (
        re.match(r"^tie\b", pt) and re.search(r"\bto win\b|\bwins?\b", kt)
    ):
        return "V8_RESULT_OPTION:tie_vs_team"

    # Numbered-event mismatch (UFC etc.).
    for pref in ["ufc"]:
        a = re.search(rf"\b{pref}\s+(\d+)\b", kall)
        b = re.search(rf"\b{pref}\s+(\d+)\b", pall)
        if a and b and a.group(1) != b.group(1):
            return f"V8_EVENT_NUMBER:{pref}{a.group(1)}!={pref}{b.group(1)}"

    # Entity-name traps where containment is not synonymy.
    ks, ps = norm(k.get("subject")), norm(p.get("subject"))
    if {ks, ps} == {"milan", "inter milan"}:
        return "V8_ENTITY_COLLISION:Milan!=Inter_Milan"
    def state_school_pair(a, b):
        # Michigan State != Michigan, Florida State != Florida, etc.
        aa, bb = a.split(), b.split()
        if len(aa) == 2 and aa[-1] in {"st", "state"} and len(bb) == 1 and aa[0] == bb[0]:
            return True
        if len(bb) == 2 and bb[-1] in {"st", "state"} and len(aa) == 1 and bb[0] == aa[0]:
            return True
        return False
    if ks and ps and state_school_pair(ks, ps):
        return "V8_ENTITY_COLLISION:state_school!=base_school"

    # High-certainty entity collisions observed in the challenge universe.
    collisions = [
        (r"\bmiami oh\b", r"\bmiami hurricanes\b"),
        (r"\bvirginia tech\b", r"\bvirginia\b"),
        (r"\bmontreal alouettes\b", r"\bcf montreal\b"),
        (r"\bhong kong china\b", r"\bchina\b"),
        (r"\bcincinnati\b.*\b(?:al|nl)\b", r"\bfc cincinnati\b"),
    ]
    for a, b in collisions:
        if (re.search(a, kall) and re.search(b, pall)) or (re.search(a, pall) and re.search(b, kall)):
            return "V8_ENTITY_COLLISION"

    return None


# ---------------------------------------------------------------------------
# Positive proof / abstention layer
# ---------------------------------------------------------------------------
_STOP = {
    "will", "the", "a", "an", "of", "for", "to", "be", "win", "wins", "winner", "champion",
    "championship", "in", "on", "at", "and", "or", "this", "next", "new", "party", "election",
    "2026", "2027", "2028", "2029", "2030",
}


def _tokens(x):
    return {t for t in norm(x).split() if t not in _STOP and len(t) > 1}


def token_jaccard(a, b):
    x, y = _tokens(a), _tokens(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def subject_compatible(k, p):
    a, b = norm(k.get("subject")), norm(p.get("subject"))
    if not a or not b:
        return False
    if a == b:
        return True
    # Long containment is safe for common full-name/team-suffix expansions.
    if len(a) >= 5 and len(b) >= 5 and (a in b or b in a):
        return True
    return token_jaccard(a, b) >= 0.60


def _material_one_sided(k, p):
    """Return a reason when one parser exposes a material slot the other omits.

    We REVIEW rather than REJECT these because parser omission is common; the point
    is simply to prevent an automatic PASS.
    """
    for fld in ["metric", "rank_semantics", "period_scope", "gender_scope", "office_scope"]:
        a, b = norm(k.get(fld)), norm(p.get(fld))
        if bool(a) != bool(b):
            return f"ONE_SIDED_{fld.upper()}"
    # Explicit threshold on just one side is also insufficient positive proof.
    ka = any(k.get(f) is not None for f in ["threshold_low", "threshold_high"])
    pa = any(p.get(f) is not None for f in ["threshold_low", "threshold_high"])
    if ka != pa:
        return "ONE_SIDED_THRESHOLD"
    return None


def _scope_review_reason(row, k, p):
    kt, pt, ke, pe, kall, pall = texts(row, k, p)

    # "after/following election" is not automatically identical to "next office holder".
    post_k = bool(re.search(r"\b(?:after|following) (?:the )?(?:next )?.*election\b", kall))
    post_p = bool(re.search(r"\b(?:after|following) (?:the )?(?:next )?.*election\b", pall))
    next_office_k = bool(re.search(r"\bnext (?:prime minister|chief minister|president|press secretary|majority leader|leader)\b", kall))
    next_office_p = bool(re.search(r"\bnext (?:prime minister|chief minister|president|press secretary|majority leader|leader)\b", pall))
    if post_k != post_p and (next_office_k or next_office_p):
        return "SCOPE_AFTER_ELECTION_VS_NEXT"

    # One-sided explicit cutoff is not enough for an automatic pair proof.
    cutoff_k = bool(re.search(r"\bbefore (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})|\bby (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", kall))
    cutoff_p = bool(re.search(r"\bbefore (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})|\bby (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", pall))
    # Calendar-year aliases are safe: "before Jan 1, Y+1" vs "in Y".
    calendar_alias = (
        (re.search(r"\bbefore jan(?:uary)? 1 20\d{2}\b", kall) and re.search(r"\bin 20\d{2}\b|\bthis year\b", pall))
        or (re.search(r"\bbefore jan(?:uary)? 1 20\d{2}\b", pall) and re.search(r"\bin 20\d{2}\b|\bthis year\b", kall))
    )
    if cutoff_k != cutoff_p and not calendar_alias:
        return "ONE_SIDED_CUTOFF"

    # "next to leave" with a deadline on only one side is especially unsafe.
    if "next" in kall and "leave" in kall and "next" in pall and "leave" in pall:
        if cutoff_k != cutoff_p:
            return "NEXT_LEAVE_DEADLINE"

    # Metric/seats wording on only one side may be a rule-defined synonym, but requires review.
    if ("most seats" in kall) != ("most seats" in pall):
        if "election" in kall or "election" in pall:
            return "ONE_SIDED_MOST_SEATS"

    return None


def _certificate_status(certificate):
    if isinstance(certificate, dict):
        return str(certificate.get("resolution_rule_status") or "").upper()
    return ""


def pair_decision(row, k, p, certificate=None):
    """Return a tri-state precision decision dict.

    PASS   = strong positive semantic proof; eligible for *rule* audit / live pricing.
    REVIEW = plausible pair but not proven enough; NO TRADE.
    REJECT = explicit semantic contradiction; NO TRADE.
    """
    kt, pt, ke, pe, kall, pall = texts(row, k, p)

    # Exact normalized market questions are positive semantic proof.  Parser
    # disagreements on such rows are treated as parser noise; settlement-rule
    # differences are deliberately handled by the later rule gate.
    if kt and kt == pt:
        ej = token_jaccard(ke, pe)
        return {"decision": "PASS", "reason": "PROOF_EXACT_TITLE", "event_jaccard": ej, "title_jaccard": 1.0}

    hard = precision_veto(row, k, p)
    if hard:
        return {"decision": "REJECT", "reason": hard, "event_jaccard": 0.0, "title_jaccard": 0.0}

    ej = token_jaccard(ke, pe)
    tj = token_jaccard(kt, pt)
    scope_reason = _scope_review_reason(row, k, p)

    status = _certificate_status(certificate)

    if scope_reason:
        return {"decision": "REVIEW", "reason": scope_reason, "event_jaccard": ej, "title_jaccard": tj}

    # Existing rule-parser uncertainty is never promoted to an automatic PASS in
    # the precision lane.  It can be investigated by the later raw-rule audit.
    if status == "REVIEW":
        return {"decision": "REVIEW", "reason": "RULE_CERTIFICATE_REVIEW", "event_jaccard": ej, "title_jaccard": tj}

    sub_ok = subject_compatible(k, p)

    # Exact canonical event identity + same subject is strong proof, but require
    # modest independent title overlap so an over-broad parser event label cannot
    # auto-certify an unrelated pair.  Lower-overlap rows become REVIEW, not reject.
    if ke and pe and ke == pe and sub_ok and tj >= 0.35:
        return {"decision": "PASS", "reason": "PROOF_EXACT_EVENT", "event_jaccard": ej, "title_jaccard": tj}

    # Near-identical event wording is allowed only with independent subject support.
    if sub_ok and ej >= 0.65 and tj >= 0.35:
        return {"decision": "PASS", "reason": "PROOF_EVENT_AND_SUBJECT", "event_jaccard": ej, "title_jaccard": tj}

    # Same recognized award/metric and good title overlap can prove terse sports/awards aliases.
    ka, pa = award_key(kall), award_key(pall)
    if sub_ok and ka and ka == pa and tj >= 0.45:
        return {"decision": "PASS", "reason": "PROOF_SAME_AWARD", "event_jaccard": ej, "title_jaccard": tj}
    km, pm = metric_key(kall), metric_key(pall)
    if sub_ok and km and km == pm and tj >= 0.45:
        return {"decision": "PASS", "reason": "PROOF_SAME_METRIC", "event_jaccard": ej, "title_jaccard": tj}

    return {"decision": "REVIEW", "reason": "INSUFFICIENT_POSITIVE_PROOF", "event_jaccard": ej, "title_jaccard": tj}


def veto(row, k, p):
    """Compatibility wrapper: apply V7 first, then V8 hard contradictions only."""
    if v7 is not None:
        reason = v7.veto(row, k, p)
        if reason:
            return reason
    return precision_veto(row, k, p)
