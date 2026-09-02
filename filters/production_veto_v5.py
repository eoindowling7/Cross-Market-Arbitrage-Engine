"""Production equivalence veto V5.

Successor to V4. DeBERTa remains the primary matcher; this file only adds or
repairs high-confidence contradiction vetoes found during independent review
of the 497 V4-vs-working-reference disagreements.

V5 changes are intentionally narrow:
- repair proven alias/parser over-rejections (MAC and fixture-name aliases)
- add a small number of repeatable semantic contradiction families that V4 missed
- avoid converting the veto layer into a second matcher
"""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import re, json, csv, collections
from datetime import datetime, timezone

_spec = spec_from_file_location('v4', str(Path(__file__).with_name('production_veto_v4.py')))
v4 = module_from_spec(_spec); _spec.loader.exec_module(v4)
base = v4.base
norm = v4.norm

# ---------------------------------------------------------------------------
# PROTECTIONS / BUG FIXES discovered in independent disagreement review
# ---------------------------------------------------------------------------
# These are genuine representation aliases observed on same-fixture pairs.
# Mutating the V3 alias table is deliberate: V4 ultimately delegates entity
# comparison to V3's canonicalizer.
base.TEAM_ALIASES.update({
    'shenzhen xinpengcheng': 'shenzhen peng city',
    'shenzhen xinpengcheng fc': 'shenzhen peng city',
    'al ettifaq saudi': 'al ittifaq',
    'al ettifaq saudi club': 'al ittifaq',
    'al ettifaq': 'al ittifaq',
    'sparta praha': 'sparta prague',
    'ac sparta praha': 'sparta prague',
    'liverpool montevideo': 'liverpool m',
    'el masry': 'al masry',
    'el masry sc': 'al masry',
    'bod glimt': 'bodoe glimt',
    'fk bod glimt': 'bodoe glimt',
    'rc deportivo a coruna': 'deportivo de la coruna',
})

# V4's old substring order read "Mid-American Conference" as AAC because
# "American Conference" appeared before MAC. Replace with boundary-aware logic.
def conf_name_v5(text):
    t = norm(text)
    patterns = [
        ('big_ten', [r'\bbig ten\b', r'\bbig 10\b']),
        ('big_12', [r'\bbig 12\b']),
        ('sec', [r'\bsoutheastern conference\b', r'\bsec\b']),
        ('acc', [r'\batlantic coast conference\b', r'\bacc\b']),
        ('mac', [r'\bmid american conference\b', r'\bmac\b']),
        ('aac', [r'\bamerican athletic conference\b', r'\bamerican conference\b', r'\baac\b']),
        ('cusa', [r'\bconference usa\b', r'\bc usa\b', r'\bcusa\b']),
        ('sun_belt', [r'\bsun belt\b']),
        ('mwc', [r'\bmountain west\b', r'\bmwc\b']),
        ('pac12', [r'\bpac 12\b']),
        ('southern', [r'\bsouthern conference\b']),
    ]
    for key, pats in patterns:
        if any(re.search(p, t) for p in pats):
            return key
    return None

v4.conf_name = conf_name_v5

# ---------------------------------------------------------------------------
# Helpers for new narrow contradiction families
# ---------------------------------------------------------------------------
MONTHNUM = {m.upper(): i for i, m in enumerate([
    'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
], 1)}

def ticker_calendar_date(ticker):
    """Read Kalshi's YYMONDD date immediately after the first dash when present."""
    m = re.search(r'-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})', str(ticker).upper())
    if not m:
        return None
    yy, mon, dd = int(m.group(1)), MONTHNUM[m.group(2)], int(m.group(3))
    try:
        return datetime(2000 + yy, mon, dd, tzinfo=timezone.utc).date()
    except ValueError:
        return None

def iso_date_in_text(text):
    m = re.search(r'\b(20\d{2})-(\d{2})-(\d{2})\b', str(text))
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc).date()
    except ValueError:
        return None

def clean_subject(sig):
    s = norm(sig.get('subject') or '')
    s = re.sub(r'\b(?:wins?|winning)\b.*$', '', s).strip()
    s = re.sub(r'\bfirst\s*5\s*innings?\b.*$', '', s).strip()
    return s

def same_person_without_suffix(a, b):
    a = norm(a); b = norm(b)
    aa = set(a.split()) - {'jr','sr','ii','iii','iv'}
    bb = set(b.split()) - {'jr','sr','ii','iii','iv'}
    return bool(aa) and aa == bb

def explicit_year(text):
    m = re.search(r'\b(20\d{2})\b', str(text))
    return int(m.group(1)) if m else None

def state_name(text):
    return base.state_name(text)

# ---------------------------------------------------------------------------
# New high-certainty vetoes
# ---------------------------------------------------------------------------
def sports_v5_extra(row, k, p):
    kt = row['kalshi_title']; pt = row['polymarket_question']
    t1 = norm(kt); t2 = norm(pt); pref = str(row['kalshi_ticker']).split('-')[0].upper()

    # 1) First-five winner on opposite sides of the same baseball fixture.
    if ('F5' in pref or 'first 5 innings' in t1 or 'first 5 innings' in t2 or 'after 5 innings' in t2):
        a = clean_subject(k); b = clean_subject(p)
        if a and b and base.entity_equiv(a, b, True) is False:
            return f'SPORT_FIRST5_ENTITY:{base.canon_entity(a,True)}!={base.canon_entity(b,True)}'

    # 2) Match rows carrying materially different calendar dates. Allow a one-day
    # tolerance for timezone/local-date representation; reject larger gaps only.
    if any(x in pref for x in ['GAME','2H','1H','F5','1Q']):
        dk = ticker_calendar_date(row['kalshi_ticker']); dp = iso_date_in_text(pt)
        if dk and dp and abs((dk-dp).days) >= 2:
            return f'SPORT_FIXTURE_DATE:{dk}!={dp}'

    # 3) First-quarter/period outcome versus a season/national championship.
    if (('1st quarter' in t1 or 'first quarter' in t1) and re.search(r'national championship|cfp championship', t2)) or \
       (('1st quarter' in t2 or 'first quarter' in t2) and re.search(r'national championship|cfp championship', t1)):
        return 'SPORT_PERIOD_VS_SEASON_TITLE'

    # 4) College conference tournament title is not the NCAA national title.
    conf_tourn1 = ('conference tournament champion' in t1 or re.search(r'\b(?:acc|sec|big ten|big 12|pac 12|mid american|american athletic|conference usa|missouri valley|ohio valley|southwestern athletic|southland)\b.*\btournament champions?\b', t1))
    conf_tourn2 = ('conference tournament champion' in t2 or re.search(r'\b(?:acc|sec|big ten|big 12|pac 12|mid american|american athletic|conference usa|missouri valley|ohio valley|southwestern athletic|southland)\b.*\btournament champions?\b', t2))
    nat1 = bool(re.search(r'ncaa (?:mens|men s) basketball national championship|college basketball national championship', t1))
    nat2 = bool(re.search(r'ncaa (?:mens|men s) basketball national championship|college basketball national championship', t2))
    if (conf_tourn1 and nat2) or (conf_tourn2 and nat1):
        return 'SPORT_CBB_CONFERENCE_VS_NATIONAL'

    # 5) Same statistic, but conference leader vs national Division-I leader.
    conf_leader1 = bool(re.search(r'\b(?:big ten|acc|sec|big 12|pac 12|conference)\b.*(?:passing|rushing) yards leader|most (?:passing|rushing) yards in the (?:big ten|acc|sec|big 12|pac 12)', t1))
    conf_leader2 = bool(re.search(r'\b(?:big ten|acc|sec|big 12|pac 12|conference)\b.*(?:passing|rushing) yards leader|most (?:passing|rushing) yards in the (?:big ten|acc|sec|big 12|pac 12)', t2))
    nat_leader1 = 'd1 passing yards leader' in t1 or 'd1 rushing yards leader' in t1
    nat_leader2 = 'd1 passing yards leader' in t2 or 'd1 rushing yards leader' in t2
    if (conf_leader1 and nat_leader2) or (conf_leader2 and nat_leader1):
        return 'SPORT_CONFERENCE_STAT_VS_NATIONAL_STAT'

    # 6) Match winner versus player/team top-batter prop in cricket.
    if ('cricket' in t1 or 'cricket' in t2 or 't20' in t1+t2):
        if ('team top batter' in t1) != ('team top batter' in t2):
            if ('wins' in t1 or 'winner' in t1 or 'wins' in t2 or 'winner' in t2):
                return 'SPORT_MATCH_WINNER_VS_TOP_BATTER'

    # 7) MLB division winner versus league championship series winner.
    div1 = bool(re.search(r'\b(?:al|nl) (?:east|central|west) division winner\b', t1))
    div2 = bool(re.search(r'\b(?:al|nl) (?:east|central|west) division winner\b', t2))
    lcs1 = 'american league championship series' in t1 or 'national league championship series' in t1
    lcs2 = 'american league championship series' in t2 or 'national league championship series' in t2
    if (div1 and lcs2) or (div2 and lcs1):
        return 'SPORT_MLB_DIVISION_VS_LCS'

    # 8) National Rugby League versus NFL/AFC/NFC championship.
    rugby1 = 'national rugby league' in t1; rugby2 = 'national rugby league' in t2
    nfl1 = bool(re.search(r'\bnfl\b|\bafc championship\b|\bnfc championship\b|pro football', t1))
    nfl2 = bool(re.search(r'\bnfl\b|\bafc championship\b|\bnfc championship\b|pro football', t2))
    if (rugby1 and nfl2) or (rugby2 and nfl1):
        return 'SPORT_RUGBY_VS_AMERICAN_FOOTBALL'

    # 9) Halftime performer/headliner is not the football champion.
    show1 = 'halftime show' in t1 and ('perform' in t1 or 'headline' in t1)
    show2 = 'halftime show' in t2 and ('perform' in t2 or 'headline' in t2)
    champ1 = bool(re.search(r'nfl league championship|pro football champion|afc championship|nfc championship', t1))
    champ2 = bool(re.search(r'nfl league championship|pro football champion|afc championship|nfc championship', t2))
    if (show1 and champ2) or (show2 and champ1):
        return 'SPORT_HALFTIME_ROLE_VS_CHAMPION'

    # 10) Soccer transfer polarity: leave versus stay.
    if 'manchester united' in t1+t2 or 'transfer' in norm((k.get('context') or '')+' '+(p.get('context') or '')):
        leave1 = 'leave ' in t1; leave2 = 'leave ' in t2
        stay1 = 'stay at ' in t1 or 'stay with ' in t1; stay2 = 'stay at ' in t2 or 'stay with ' in t2
        if (leave1 and stay2) or (leave2 and stay1):
            return 'SPORT_TRANSFER_LEAVE_VS_STAY'

    # 11) College State-name distinction (Florida St != Florida, Michigan St != Michigan, etc.)
    if any(x in t1+t2 for x in ['college football','college basketball','ncaa','cfp']):
        a = norm(base.sports_outcome_entity(kt,k) or k.get('subject') or '')
        b = norm(base.sports_outcome_entity(pt,p) or p.get('subject') or '')
        if a and b:
            A=set(a.split()); B=set(b.split())
            # st/state is identity-bearing when the other side lacks it.
            astate = bool(A & {'st','state'}); bstate = bool(B & {'st','state'})
            abase=A-{'st','state'}-base.SPORT_NICKS; bbase=B-{'st','state'}-base.SPORT_NICKS
            if astate != bstate and abase and bbase and (abase <= bbase or bbase <= abase):
                return 'SPORT_COLLEGE_STATE_ENTITY'

    # 12) Generic Ballon d'Or outcome rows still carry the selected outcome in sig.subject.
    if ('ballon d or' in t1 or 'ballon dor' in t1) and ('ballon d or' in t2 or 'ballon dor' in t2):
        a=norm(k.get('subject') or ''); b=norm(p.get('subject') or '')
        if a and b and base.entity_equiv(a,b,False) is False:
            return f'SPORT_AWARD_ENTITY:{base.canon_entity(a)}!={base.canon_entity(b)}'

    return None


def politics_v5_extra(row, k, p):
    kt=row['kalshi_title']; pt=row['polymarket_question']; t1=norm(kt); t2=norm(pt)

    # Minister of Defense versus Prime Minister / other explicit political office.
    def office_extra(t):
        if 'minister of defense' in t or 'defence minister' in t or 'defense minister' in t: return 'defense_minister'
        if 'prime minister' in t: return 'prime_minister'
        return None
    o1=office_extra(t1);o2=office_extra(t2)
    if o1 and o2 and o1!=o2:
        return f'POL_OFFICE_EXTRA:{o1}!={o2}'

    # Impeachment with an explicit early cutoff versus "before term ends".
    if 'impeach' in t1 and 'impeach' in t2 and (('term end' in t1) or ('term end' in t2) or ('term ends' in t1) or ('term ends' in t2)):
        d1=k.get('resolution_rule_deadline_ts'); d2=p.get('resolution_rule_deadline_ts')
        if d1 and d2 and abs(float(d1)-float(d2)) > 86400*2:
            return 'POL_IMPEACH_HORIZON_MISMATCH'

    # Same-state governor market whose Kalshi settlement horizon is years beyond
    # the explicitly named PM election year (caught NH 2028-vs-2026 family).
    if ('governor' in t1 or 'governorship' in t1) and ('governor' in t2 or 'gubernatorial' in t2):
        s1=state_name(t1); s2=state_name(t2); y=explicit_year(pt)
        if s1 and s2 and s1==s2 and y and k.get('end_ts'):
            ky=datetime.fromtimestamp(float(k['end_ts']), timezone.utc).year
            if ky >= y+2:
                return f'POL_ELECTION_HORIZON:{ky}!={y}'

    # Speaker/action direction: "Warsh says Trump" != "Trump praises Warsh".
    if (' say ' in (' '+t1+' ') and ' praise ' in (' '+t2+' ')) or (' say ' in (' '+t2+' ') and ' praise ' in (' '+t1+' ')):
        return 'POL_SPEAKER_ACTION_MISMATCH'

    return None


def misc_v5_extra(row,k,p):
    kt=row['kalshi_title']; pt=row['polymarket_question']; t1=norm(kt);t2=norm(pt)

    # Netflix operational metric versus NFLX equity price.
    app1='netflix app downloads' in t1; app2='netflix app downloads' in t2
    stock1='netflix nflx' in t1 and ('finish week' in t1 or '$' in str(kt)); stock2='netflix nflx' in t2 and ('finish week' in t2 or '$' in str(pt))
    if (app1 and stock2) or (app2 and stock1):
        return 'MISC_NETFLIX_APP_VS_STOCK'

    # Current-dollar/nominal GDP is not equivalent to unspecified GDP growth.
    gdp1=('gdp' in t1 or 'gross domestic product' in t1); gdp2=('gdp' in t2 or 'gross domestic product' in t2)
    if gdp1 and gdp2:
        nominal1='current dollar' in t1 or 'nominal' in t1
        nominal2='current dollar' in t2 or 'nominal' in t2
        real1='real gdp' in t1 or 'real gross domestic product' in t1
        real2='real gdp' in t2 or 'real gross domestic product' in t2
        if nominal1!=nominal2 or real1!=real2:
            return 'MISC_GDP_MEASURE_V5'

    # OpenAI/Anthropic IPO race versus one-company deadline. V4 missed spaced "Open AI".
    both1=('openai' in t1 or 'open ai' in t1) and 'anthropic' in t1
    both2=('openai' in t2 or 'open ai' in t2) and 'anthropic' in t2
    race1=both1 and 'ipo first' in t1; race2=both2 and 'ipo first' in t2
    deadline1='ipo by' in t1; deadline2='ipo by' in t2
    if (race1 and deadline2) or (race2 and deadline1):
        return 'MISC_IPO_RACE_VS_DEADLINE_V5'

    # Obvious sports proposition versus equity-price proposition.
    sports_terms=lambda t: bool(re.search(r'1st half|first half|1st quarter|first quarter|wins game|football|basketball|baseball',t))
    stock_terms=lambda raw,t: ('finish week' in t and ('$' in str(raw) or re.search(r'\([A-Z]{1,5}\)',str(raw))))
    if (sports_terms(t1) and stock_terms(pt,t2)) or (sports_terms(t2) and stock_terms(kt,t1)):
        return 'MISC_SPORT_VS_STOCK'

    return None


def veto(row,k,p):
    # Run repaired V4 first. Alias/confidence protections above affect this call.
    reason=v4.veto(row,k,p)
    if reason:
        return reason

    if k.get('domain')=='sports' or p.get('domain')=='sports' or any(x in str(row['kalshi_ticker']).upper() for x in ['MLB','NCAA','NFL','SOCCER','NRL']):
        reason=sports_v5_extra(row,k,p)
        if reason:return reason

    # The politics extras are individually narrow (office, impeachment horizon,
    # election horizon, speaker/action direction), so call them unconditionally. This
    # avoids missing political-language markets whose upstream domain was 'other'.
    reason=politics_v5_extra(row,k,p)
    if reason:return reason

    reason=misc_v5_extra(row,k,p)
    if reason:return reason
    return None


if __name__=='__main__':
    path = __import__('sys').argv[1] if len(__import__('sys').argv)>1 else '/mnt/data/matcher_review_4119.csv'
    out=[]
    with open(path,encoding='utf8') as f:
        for i,r in enumerate(csv.DictReader(f)):
            k=json.loads(r['ksig']);p=json.loads(r['psig']);reason=veto(r,k,p);out.append((i,bool(reason),reason))
    print('N',len(out),'veto',sum(x[1] for x in out),'pass',sum(not x[1] for x in out))
    print(collections.Counter((x[2] or 'PASS').split(':')[0] for x in out).most_common(100))
