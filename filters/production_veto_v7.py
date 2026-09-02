"""Production equivalence veto V7.

Successor to V6. DeBERTa remains the primary matcher.

V7 fixes one regression introduced by V6 (cycling Grand Prix events were no
longer mistaken for F1, but that exposed true cycling-event mismatches), and
suppresses only a handful of demonstrated false entity vetoes from the full
11,025-universe side-effect audit.
"""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import re, csv, json, collections

_spec = spec_from_file_location('v6', str(Path(__file__).with_name('production_veto_v6.py')))
v6 = module_from_spec(_spec); _spec.loader.exec_module(v6)
v5 = v6.v5
v4 = v6.v4
base = v6.base
norm = v6.norm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def same_year(a,b):
    ya = re.search(r'\b(20\d{2})\b', str(a))
    yb = re.search(r'\b(20\d{2})\b', str(b))
    return not (ya and yb) or ya.group(1) == yb.group(1)


def cycling_event(text):
    t = norm(text)
    if 'vuelta a espana' in t:
        return 'vuelta_a_espana'
    if 'grand prix cycliste de quebec' in t:
        return 'gp_cycliste_quebec'
    if 'grand prix cycliste de montreal' in t:
        return 'gp_cycliste_montreal'
    if 'tour de france' in t:
        return 'tour_de_france'
    if 'giro d italia' in t or 'giro ditalia' in t:
        return 'giro_d_italia'
    return None


def heisman_attribute(text):
    t = norm(text)
    if 'senior' in t or 'graduate' in t:
        return 'senior_or_graduate'
    if 'sophomore' in t:
        return 'sophomore'
    if 'junior' in t:
        return 'junior'
    if 'freshman' in t:
        return 'freshman'
    if re.search(r'\bsec\b|southeastern conference', t):
        return 'sec'
    if 'big ten' in t or 'big 10' in t:
        return 'big_ten'
    if 'big 12' in t:
        return 'big_12'
    if re.search(r'\bacc\b|atlantic coast conference', t):
        return 'acc'
    if 'conference usa' in t:
        return 'conference_usa'
    return None


def mlb_axis(text):
    t = norm(text)
    for div in ['al east','al central','al west','nl east','nl central','nl west']:
        if div in t and ('division winner' in t or 'title' in t):
            return div.replace(' ','_')
    if 'american league championship series' in t or 'pro baseball american league championship' in t:
        return 'al_championship'
    if 'national league championship series' in t or 'pro baseball national league championship' in t:
        return 'nl_championship'
    if 'world series' in t or 'pro baseball championship' in t:
        return 'world_series'
    return None

MLB_ALIAS = {
    'cincinnati': 'cincinnati_reds', 'cincinnati reds': 'cincinnati_reds',
    'kansas city': 'kansas_city_royals', 'kansas city royals': 'kansas_city_royals',
    'cleveland': 'cleveland_guardians', 'cleveland guardians': 'cleveland_guardians',
    'arizona': 'arizona_diamondbacks', 'arizona diamondbacks': 'arizona_diamondbacks',
    'miami': 'miami_marlins', 'miami marlins': 'miami_marlins',
    'atlanta': 'atlanta_braves', 'atlanta braves': 'atlanta_braves',
    'toronto': 'toronto_blue_jays', 'toronto blue': 'toronto_blue_jays', 'toronto blue jays': 'toronto_blue_jays',
    'boston': 'boston_red_sox', 'boston red': 'boston_red_sox', 'boston red sox': 'boston_red_sox',
    'chicago ws': 'chicago_white_sox', 'chicago white': 'chicago_white_sox', 'chicago white sox': 'chicago_white_sox',
    'as': 'athletics', 'athletics': 'athletics',
}


def mlb_entity_from_title(text):
    t = norm(text)
    # Kalshi division-winner template.
    m = re.match(r'^will (.+?) be the 20\d{2} (?:al|nl) (?:east|central|west) division winner$', t)
    if m:
        return MLB_ALIAS.get(m.group(1), m.group(1))
    # Generic win championship/title template.
    m = re.match(r'^will (.+?) win the 20\d{2} ', t)
    if m:
        x = m.group(1)
        x = re.sub(r'^the ', '', x)
        return MLB_ALIAS.get(x, x)
    return None


def baseball_wins_leader_name(text):
    t = norm(text)
    m = re.match(r'^will (.+?) lead pro baseball in wins for the 20\d{2} regular season$', t)
    if m:
        return m.group(1)
    m = re.match(r'^will (.+?) lead mlb in pitcher wins for the 20\d{2} season$', t)
    if m:
        return m.group(1)
    return None


def primary_nominee_parts(text):
    t = norm(text)
    # Kalshi: candidate be the Democratic/Republican nominee for Governor in STATE
    m = re.match(r'^will (.+?) be the (democratic|republican) nominee for governor in (.+?)\??$', t)
    if m:
        return (m.group(1), m.group(2), base.state_name(t), 'governor')
    # Polymarket: candidate win the 2026 STATE Governor Democratic/Republican primary election
    m = re.match(r'^will (.+?) win the 20\d{2} (.+?) governor (democratic|republican) primary election\??$', t)
    if m:
        return (m.group(1), m.group(3), base.state_name(t), 'governor')
    return None


def same_primary_nominee_market(a,b):
    x=primary_nominee_parts(a); y=primary_nominee_parts(b)
    if not x or not y:
        return False
    na,pa,sa,oa=x; nb,pb,sb,ob=y
    return pa==pb and oa==ob and sa and sa==sb and base.entity_equiv(na,nb,False) is True and same_year(a,b)


def bernie_endorse_target(text):
    t=norm(text)
    if 'bernie' not in t or 'endorse' not in t:
        return None
    # Long Kalshi form.
    m=re.search(r'bernie(?: sanders)? endorse (.+?) in the 20\d{2} united states senate election in ([a-z ]+?) before ',t)
    if m:
        return (m.group(1), m.group(2).strip())
    # Short PM form, e.g. James Talarico for TX-Sen.
    m=re.search(r'bernie(?: sanders)? endorse (.+?) for ([a-z]{2}) sen by ',t)
    if m:
        code=m.group(2)
        states={'tx':'texas','ne':'nebraska'}
        return (m.group(1), states.get(code,code))
    return None


def same_bernie_endorse_market(a,b):
    x=bernie_endorse_target(a); y=bernie_endorse_target(b)
    if not x or not y:
        return False
    name1,state1=x; name2,state2=y
    if state1 != state2 or base.entity_equiv(name1,name2,False) is not True:
        return False
    c1=base.cutoff(a); c2=base.cutoff(b)
    return v6.cutoff_equivalent_v6(c1,c2) is True


def explicit_party_general_equiv(a,b):
    t1=norm(a); t2=norm(b)
    # Both titles must explicitly be party-level outcomes, avoiding candidate-vs-party markets.
    def party_explicit(t):
        if 'democrat' in t or 'democratic' in t:
            return 'D'
        if 'republican' in t:
            return 'R'
        return None
    p1=party_explicit(t1); p2=party_explicit(t2)
    if not p1 or p1!=p2:
        return False
    if not (('party candidate' in t1 or 'party candidate' in t2) and ('win' in t1 and 'win' in t2)):
        return False
    s1=base.state_name(t1); s2=base.state_name(t2)
    if s1 and s2 and s1!=s2:
        return False
    office1='senate' if 'senate' in t1 else ('governor' if 'governor' in t1 else None)
    office2='senate' if 'senate' in t2 else ('governor' if 'governor' in t2 else None)
    return bool(office1 and office1==office2 and same_year(a,b))


# ---------------------------------------------------------------------------
# Main veto
# ---------------------------------------------------------------------------

def veto(row,k,p):
    kt=row['kalshi_title']; pt=row['polymarket_question']
    t1=norm(kt); t2=norm(pt)

    # V6 correctly reclassified "Grand Prix Cycliste" as cycling rather than F1.
    # But different named cycling races/classifications are still different markets.
    ce1=cycling_event(kt); ce2=cycling_event(pt)
    if ce1 and ce2 and ce1 != ce2:
        return f'SPORT_CYCLING_EVENT:{ce1}!={ce2}'

    reason=v6.veto(row,k,p)
    if not reason:
        return None

    # -----------------------------------------------------------------------
    # Suppress only demonstrated false entity vetoes.
    # -----------------------------------------------------------------------

    if reason.startswith('SPORT_ENTITY:'):
        # Same Heisman demographic/conference attribute, phrased differently.
        if 'heisman trophy' in t1 and 'heisman trophy' in t2:
            h1=heisman_attribute(kt); h2=heisman_attribute(pt)
            if h1 and h1==h2 and same_year(kt,pt):
                return None

        # Known representation aliases from full-universe audit.
        alias_pairs=[
            ('m gladbach','borussia monchengladbach'),
            ('winnipeg jets','winnipeg jets'),
            ('sporting jax','sporting jacksonville'),
            ('sporting jax','sporting club jacksonville'),
            ('umass','massachusetts minutemen'),
        ]
        # Use title text rather than parsed subject because the bug is in subject extraction.
        for a,b in alias_pairs:
            if ((a in t1 and b in t2) or (b in t1 and a in t2)) and same_year(kt,pt):
                return None

        # One audited MLB metadata typo: the Yankees rookie is officially Spencer
        # Jones (no Jr.); both titles are the same AL Rookie of the Year market.
        if (('spencer jones jr' in t1 and 'spencer jones' in t2) or
            ('spencer jones jr' in t2 and 'spencer jones' in t1)) and \
           ('rookie of the year' in t1 or 'roty' in t1) and \
           ('rookie of the year' in t2 or 'roty' in t2) and same_year(kt,pt):
            return None

        # Same MLB pitcher-wins leader, with Pro Baseball vs MLB wording.
        n1=baseball_wins_leader_name(kt); n2=baseball_wins_leader_name(pt)
        if n1 and n2 and base.entity_equiv(n1,n2,False) is True and same_year(kt,pt):
            return None

        # City/abbreviation versus MLB team name is safe only when the competition
        # axis itself is exactly the same on both sides.
        ax1=mlb_axis(kt); ax2=mlb_axis(pt)
        if ax1 and ax1==ax2:
            e1=mlb_entity_from_title(kt); e2=mlb_entity_from_title(pt)
            if e1 and e2 and e1==e2 and same_year(kt,pt):
                return None

    if reason.startswith('ENT_ENTITY:'):
        # Edition number / work-title suffixes observed to refer to the same nominee.
        if ('83rd annual golden globes' in t1 and 'the golden globes' in t2) or ('83rd annual golden globes' in t2 and 'the golden globes' in t1):
            return None
        if ('68th annual grammy awards' in t1 and 'the grammys' in t2) or ('68th annual grammy awards' in t2 and 'the grammys' in t1):
            return None
        for person in ['constance zimmer','sarah pidgeon']:
            if person in t1 and person in t2:
                return None

    if reason.startswith('POL_ENTITY:'):
        # Daniel J. Sullivan / Daniel J. Sullivan Jr. is the same 2026 Alaska
        # Senate candidate; the suffix is used inconsistently across feeds.
        if 'alaska senate' in t1 and 'alaska senate' in t2 and same_year(kt,pt):
            if (('daniel j sullivan' in t1 and 'daniel j sullivan jr' in t2) or
                ('daniel j sullivan jr' in t1 and 'daniel j sullivan' in t2)):
                return None

        # Winning a party primary and being that party's nominee for the same
        # gubernatorial election are the same candidate-level outcome.
        if same_primary_nominee_market(kt,pt):
            return None

        # "anyone" and "any presidential candidate" are representation variants
        # for the same first-round-outcome question.
        if 'win outright in the first round' in t1 and 'win outright in the first round' in t2:
            if (('anyone' in t1 and 'any presidential candidate' in t2) or
                ('anyone' in t2 and 'any presidential candidate' in t1)):
                return None

        # Same Bernie endorsement with calendar-equivalent cutoff wording.
        if same_bernie_endorse_market(kt,pt):
            return None

        # Party wins race == that party's candidate wins race, only when both
        # titles are explicitly party-level, same office/state/year.
        if explicit_party_general_equiv(kt,pt):
            return None

    return reason


if __name__=='__main__':
    import sys
    path=sys.argv[1] if len(sys.argv)>1 else 'matcher_review_4119.csv'
    out=[]
    with open(path,encoding='utf8') as f:
        for i,r in enumerate(csv.DictReader(f)):
            k=json.loads(r['ksig']); p=json.loads(r['psig']); reason=veto(r,k,p)
            out.append((i,bool(reason),reason))
    print('N',len(out),'veto',sum(x[1] for x in out),'pass',sum(not x[1] for x in out))
    print(collections.Counter((x[2] or 'PASS').split(':')[0] for x in out).most_common(100))
