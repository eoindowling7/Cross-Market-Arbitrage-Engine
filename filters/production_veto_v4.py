"""Production equivalence veto V4.

DeBERTa remains the primary equivalence matcher.  This module adds only
high-confidence semantic contradiction vetoes discovered in the live audit.
It intentionally does not try to re-match arbitrary markets from scratch.
"""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import re, json, csv, collections

_spec=spec_from_file_location('base_v3', str(Path(__file__).with_name('production_veto_v3.py')))
base=module_from_spec(_spec); _spec.loader.exec_module(base)
norm=base.norm
entity_equiv=base.entity_equiv
canon_entity=base.canon_entity
cutoff=base.cutoff
cutoff_equivalent=base.cutoff_equivalent
parse_date_like=base.parse_date_like


def sigtext(title,sig):
    return norm(' '.join(str(x or '') for x in [
        title, sig.get('context'), sig.get('event_identity'), sig.get('competition'),
        sig.get('sports_market_type'), sig.get('stage'), sig.get('metric')
    ]))

# ------------------------------------------------------------------
# SPORTS: explicit semantic axes
# ------------------------------------------------------------------
def sport_discipline(title,sig):
    t=sigtext(title,sig)
    if 'motogp' in t: return 'motogp'
    if re.search(r'\bformula 1\b|\bf1\b|grand prix|constructors champion',t): return 'formula1'
    if 'lacrosse' in t: return 'lacrosse'
    if re.search(r'\bnba\b|\bwnba\b|basketball|march madness',t): return 'basketball'
    if re.search(r'\bnfl\b|\bcfb\b|ncaa football|college football|pro football|quarterback|super bowl',t): return 'american_football'
    if re.search(r'\bmlb\b|baseball|world series|silver slugger|cy young',t): return 'baseball'
    if re.search(r'\bnhl\b|ice hockey|hockey',t): return 'hockey'
    if re.search(r'\bufc\b|\bmma\b|ko or tko|knockout',t): return 'mma'
    if re.search(r'\bpga\b|golf|masters tournament|ryder cup|solheim cup',t): return 'golf'
    if re.search(r'cycling|vuelta a espana|tour de france',t): return 'cycling'
    if re.search(r'\bfide\b|titled tuesday|chess',t): return 'chess'
    if re.search(r'\batp\b|\bwta\b|tennis',t): return 'tennis'
    # explicit soccer ecosystem; do not classify bare word football as soccer.
    if re.search(r'uefa|fifa|premier league|\bepl\b|la liga|bundesliga|serie a|ligue 1|mls|nwsl|efl |fa cup|europa league|conference league|copa |liga mx|usl |eredivisie|superliga|soccer',t):
        return 'soccer'
    return None

CONF_NAMES={
 'big ten':'big_ten','big 10':'big_ten','big 12':'big_12','sec':'sec','acc':'acc','aac':'aac',
 'american conference':'aac','conference usa':'cusa','c usa':'cusa','cusa':'cusa','sun belt':'sun_belt',
 'mountain west':'mwc','mwc':'mwc','pac 12':'pac12','pac-12':'pac12','mac':'mac','southern conference':'southern',
}

def conf_name(t):
    t=norm(t)
    for x,k in CONF_NAMES.items():
        if x in t:return k
    return None

def sport_season_axis(title,sig):
    t=sigtext(title,sig); d=sport_discipline(title,sig)
    out=set()
    if d=='basketball':
        if 'nba cup' in t: out.add('nba_cup')
        if 'nba eastern conference' in t: out.add('nba_east')
        if 'nba western conference' in t: out.add('nba_west')
        if ('nba finals' in t or 'pro basketball finals' in t or 'pro basketball champion' in t) and 'women' not in t and 'wnba' not in t: out.add('nba_title')
        if 'wnba finals' in t or 'womens pro basketball championship' in t or 'women s pro basketball championship' in t: out.add('wnba_title')
        if ('college basketball national championship' in t or "ncaa men s basketball national championship" in t or 'march madness' in t): out.add('cbb_men_national')
        if "ncaa women s basketball national championship" in t or 'womens college basketball national championship' in t: out.add('cbb_women_national')
        c=conf_name(t)
        if c and ('tournament champion' in t or 'conference tournament champion' in t or ('championship' in t and ('ncaa basketball' in t or 'college basketball' in t or 'men s conference tournament' in t))): out.add('cbb_conf_'+c)
    if d=='american_football':
        if 'cfp national championship' in t or 'college football playoff national championship' in t or 'ncaa football national champion' in t: out.add('cfb_national')
        c=conf_name(t)
        if c and ('college football' in t or 'ncaa football' in t or 'cfb' in t): out.add('cfb_conf_'+c)
        if 'nfl league championship' in t or 'pro football championship' in t or 'super bowl' in t: out.add('nfl_title')
        if 'afc championship' in t: out.add('nfl_afc')
        if 'nfc championship' in t: out.add('nfl_nfc')
        for div in ['afc east','afc west','afc north','afc south','nfc east','nfc west','nfc north','nfc south']:
            if div in t: out.add('nfl_div_'+div.replace(' ','_'))
    if d=='baseball':
        if 'world series' in t or 'pro baseball championship' in t: out.add('mlb_world')
        if 'american league championship' in t or 'alcs' in t: out.add('mlb_al')
        if 'national league championship' in t or 'nlcs' in t: out.add('mlb_nl')
        for div in ['al east','al west','al central','nl east','nl west','nl central']:
            if div in t: out.add('mlb_div_'+div.replace(' ','_'))
    if d=='formula1':
        if 'drivers champion' in t or 'drivers championship' in t: out.add('f1_driver_title')
        if 'constructors champion' in t or 'constructors championship' in t: out.add('f1_constructor_title')
        m=re.search(r'\b(italian|spanish|monaco|british|australian|japanese|canadian|mexican|brazilian|singapore|miami|las vegas|austrian|belgian|hungarian|dutch|azerbaijan|saudi arabian|chinese) grand prix\b',t)
        if m: out.add('f1_gp_'+m.group(1).replace(' ','_'))
    if d=='motogp':
        if 'teams world champion' in t: out.add('motogp_teams_title')
    return out


def sport_scope_extra(title,sig):
    t=sigtext(title,sig); out=set()
    if 'ranked team' in t or ('#1 ranked team' in t and 'defense' not in t): out.add('ranked_team')
    if 'ranked defense' in t: out.add('ranked_defense')
    if 'triple crown' in t: out.add('triple_crown')
    if 'silver slugger' in t: out.add('silver_slugger')
    if 'undefeated at home' in t: out.add('undefeated_home')
    elif 'undefeated' in t and ('regular season' in t or 'season' in t): out.add('undefeated_overall')
    if re.search(r'win 100 or more games|100\+ wins|100 wins',t): out.add('season_100_wins')
    if 'starting quarterback' in t or 'starting qb' in t: out.add('starting_qb')
    if 'win by ko or tko' in t or 'method of victory' in t: out.add('fight_method')
    elif ('mma ' in t or 'ufc' in t) and ('winner' in t or re.search(r'\bwins\b',t)): out.add('fight_winner')
    if 'finish top 3' in t or 'finish on the podium' in t or 'podium' in t: out.add('podium')
    if 'win the f1 drivers championship' in t or 'f1 drivers champion' in t: out.add('season_driver_title')
    if 'most home runs' in t: out.add('most_home_runs')
    if 'most passing yards' in t or 'passing yards leader' in t: out.add('passing_yards')
    if 'most rushing yards' in t or 'rushing yards leader' in t: out.add('rushing_yards')
    if 'most passing touchdowns' in t or 'passing touchdowns leader' in t: out.add('passing_touchdowns')
    if 'most rushing touchdowns' in t or 'rushing touchdowns leader' in t: out.add('rushing_touchdowns')
    return out


def _strip_tokens(text, tokens):
    s=norm(text)
    words=[w for w in s.split() if w not in tokens]
    return ' '.join(words)

def fixture_opponent(title,sig):
    """Extract opponent fragment from stored event identity, removing outcome subject."""
    ev=norm(sig.get('event_identity') or sig.get('context') or '')
    if ' vs ' not in (' '+ev+' '): return None
    # Remove match-scope boilerplate.
    ev=re.sub(r'\b(regulation time moneyline|second half result|first half result|second half winner|first half winner|second half|first half|moneyline|result|mma|main card|lightweight|heavyweight|welterweight|middleweight)\b',' ',ev)
    ev=' '.join(ev.split())
    subj=base.sports_outcome_entity(title,sig) or sig.get('subject') or ''
    st=set(canon_entity(subj,True).split())
    parts=[x.strip() for x in re.split(r'\bvs\.?\b',ev) if x.strip()]
    if len(parts)==1:
        return canon_entity(parts[0],True)
    if len(parts)>=2:
        scored=[]
        for part in parts[:2]:
            c=canon_entity(part,True); toks=set(c.split())
            overlap=len(st&toks)
            scored.append((overlap,c))
        # opponent is side with smaller overlap to subject; if tie we cannot know safely.
        if scored[0][0] < scored[1][0]: return scored[0][1]
        if scored[1][0] < scored[0][0]: return scored[1][1]
        # if one side is blank-ish due serialization, use the nonempty side
        return None
    return None


def sports_extra_veto(r,k,p):
    kt=r['kalshi_title']; pt=r['polymarket_question']
    dk=sport_discipline(kt,k); dp=sport_discipline(pt,p)
    if dk and dp and dk!=dp:
        return f'SPORT_DISCIPLINE:{dk}!={dp}'
    ak=sport_season_axis(kt,k); ap=sport_season_axis(pt,p)
    if ak and ap and ak.isdisjoint(ap):
        return 'SPORT_SEASON_AXIS:'+','.join(sorted(ak))+'!='+','.join(sorted(ap))
    sk=sport_scope_extra(kt,k); sp=sport_scope_extra(pt,p)
    exclusive=[
        {'ranked_team','ranked_defense'},
        {'triple_crown','silver_slugger','most_home_runs','passing_yards','rushing_yards','passing_touchdowns','rushing_touchdowns'},
        {'undefeated_home','undefeated_overall'},
        {'fight_winner','fight_method'},
        {'season_100_wins'},
    ]
    for g in exclusive:
        a=sk&g; b=sp&g
        if a and b and a.isdisjoint(b): return 'SPORT_EXTRA_SCOPE:'+','.join(a)+'!='+','.join(b)
    # statistic/award proposition is not a championship/title proposition.
    stat_or_award={'triple_crown','silver_slugger','most_home_runs','passing_yards','rushing_yards','passing_touchdowns','rushing_touchdowns','ranked_team','ranked_defense','season_100_wins'}
    bk=base.sports_scope(r['kalshi_ticker'].split('-')[0],kt,k); bp=base.sports_scope('',pt,p)
    if (sk&stat_or_award) and ('title_winner' in bp): return 'SPORT_STAT_VS_TITLE'
    if (sp&stat_or_award) and ('title_winner' in bk): return 'SPORT_TITLE_VS_STAT'
    # simple fight winner vs winning by a method
    if ('fight_winner' in sk and 'fight_method' in sp) or ('fight_method' in sk and 'fight_winner' in sp): return 'SPORT_FIGHT_METHOD'
    # podium vs outright race win
    if ('podium' in sk and ('title_winner' in bp or 'match_winner' in bp)) or ('podium' in sp and ('title_winner' in bk or 'match_winner' in bk)):
        return 'SPORT_PODIUM_VS_WIN'
    # explicit fixture opponent mismatch where both sides can be extracted.
    granular={'match_winner','second_half','first_half','first5'}
    if (bk&granular) and (bp&granular):
        ok=fixture_opponent(kt,k); op=fixture_opponent(pt,p)
        if ok and op and base.entity_equiv(ok,op,True) is False:
            return f'SPORT_FIXTURE:{ok}!={op}'
    return None

# ------------------------------------------------------------------
# POLITICS: compound assignment, nomination/general, seat bands, geography
# ------------------------------------------------------------------
def combo_assignment(text):
    t=norm(text)
    # Return (governor party, senate party) where explicit.
    if 'governor' not in t or 'senate' not in t:return None
    def party_after(pattern):
        m=re.search(pattern,t)
        if not m:return None
        x=m.group(1)
        if 'republican' in x:return 'R'
        if 'democrat' in x:return 'D'
        return None
    # Kalshi wording: Governor winner be R ... Senate winner be D
    mg=re.search(r'governor winner be (republican|democratic) party',t)
    ms=re.search(r'senate winner be (republican|democratic) party',t)
    if mg and ms:
        return ('R' if mg.group(1)=='republican' else 'D','R' if ms.group(1)=='republican' else 'D')
    # sweep
    if 'republicans sweep' in t:return ('R','R')
    if 'democrats sweep' in t:return ('D','D')
    # "Republicans win the X Governor election and Democrats win the X Senate election"
    mg=re.search(r'(republicans|democrats) win the [a-z ]+ governor election',t)
    ms=re.search(r'(republicans|democrats) win the [a-z ]+ senate election',t)
    if mg and ms:
        return ('R' if mg.group(1).startswith('republic') else 'D','R' if ms.group(1).startswith('republic') else 'D')
    return None


def explicit_seat_band(text):
    t=norm(text)
    return bool(re.search(r'\b(?:win|hold)\s+\d+\s+(?:or more\s+)?seats\b|\b\d+\s*[-–]\s*\d+\s+seats\b|\bbetween\s+\d+.*seats',str(text).lower()))

def generic_election_entity(sig):
    ctx=norm(sig.get('context') or '')
    ev=norm(sig.get('event_identity') or '')
    if any(x in ctx+' '+ev for x in ['election winner','general election winner','parliamentary election winner','nobel peace prize winner','ballon d or']):
        return norm(sig.get('subject')) or None
    return None

def politics_extra_veto(r,k,p):
    kt=r['kalshi_title'];pt=r['polymarket_question']; t1=norm(kt);t2=norm(pt)
    ca=combo_assignment(kt); cb=combo_assignment(pt)
    if ca and cb and ca!=cb:return f'POL_COMBO_ASSIGN:{ca}!={cb}'
    # election winner vs seat-count threshold/range
    if explicit_seat_band(kt)!=explicit_seat_band(pt):
        if any(x in t1+t2 for x in ['election','parliament','assembly']):return 'POL_SEAT_BAND_VS_WINNER'
    # Party nomination/primary vs general election winner.
    nomination=lambda t: ('party nomination' in t or 'nomination to contest' in t or 'primary election' in t or 'top four primary' in t or 'top-four primary' in t)
    general=lambda t: ('presidential election' in t or 'governor election' in t or 'governorship' in t or 'general election' in t)
    if nomination(t1)!=nomination(t2) and (general(t1) or general(t2)):
        return 'POL_PRIMARY_NOMINATION_VS_GENERAL'
    # top-four primary winner vs merely advancing
    if ('top four primary' in t1 or 'top-four primary' in t1 or 'top four primary' in norm(k.get('context'))) and ('advance from' in t2 or 'advances from' in t2):
        return 'POL_PRIMARY_WIN_VS_ADVANCE'
    if ('top four primary' in t2 or 'top-four primary' in t2 or 'top four primary' in norm(p.get('context'))) and ('advance from' in t1 or 'advances from' in t1):
        return 'POL_PRIMARY_WIN_VS_ADVANCE'
    # obvious national vs subnational election.
    if ('australian house' in t1 or 'australian house' in norm(k.get('context'))) and 'victorian state' in t2:return 'POL_NATIONAL_VS_STATE'
    if ('australian house' in t2 or 'australian house' in norm(p.get('context'))) and 'victorian state' in t1:return 'POL_NATIONAL_VS_STATE'
    # Generic multi-outcome contracts: compare serialized outcome identities, not generic titles.
    ea=generic_election_entity(k); eb=generic_election_entity(p)
    if ea and eb and base.entity_equiv(ea,eb,False) is False:
        return f'POL_GENERIC_ENTITY:{canon_entity(ea)}!={canon_entity(eb)}'
    # Run/participation versus dated announcement.
    if (('run for president' in t1 and 'announce' not in t1) and ('announce' in t2)) or (('run for president' in t2 and 'announce' not in t2) and ('announce' in t1)):
        return 'POL_RUN_VS_ANNOUNCEMENT'
    # "before term is up" versus a materially earlier explicit cutoff.
    if ('before his term is up' in t1 or 'before her term is up' in t1 or 'before the term is up' in t1) and cutoff(pt):return 'POL_TERM_VS_EARLY_CUTOFF'
    if ('before his term is up' in t2 or 'before her term is up' in t2 or 'before the term is up' in t2) and cutoff(kt):return 'POL_TERM_VS_EARLY_CUTOFF'
    return None

# ------------------------------------------------------------------
# ENTERTAINMENT: chart item identity, rank bands, awards, event type
# ------------------------------------------------------------------
def chart_rank_sem(text):
    raw=str(text); t=norm(raw)
    m=re.search(r'#\s*(\d+)|\bnumber\s+(\d+)\b',raw.lower())
    if m:return ('exact',int(next(g for g in m.groups() if g)))
    m=re.search(r'\btop\s*(\d+)\b',t)
    if m:return ('top',int(m.group(1)))
    if 'be on the billboard' in t:return ('presence',None)
    return None

def award_event(text):
    t=norm(text)
    if 'daytime emmy' in t:return 'daytime_emmy'
    if 'emmy' in t:return 'primetime_emmy'
    if 'oscar' in t or 'academy awards' in t:return 'oscars'
    if 'grammy' in t:return 'grammys'
    return None

def award_category_extra(text):
    t=norm(text)
    cats=['best cinematography','best picture','best director','best actor','best supporting actor','album of the year','album of year',
          'outstanding guest performance in a daytime drama series','outstanding guest actor in a drama series','outstanding guest actress in a drama series']
    for c in cats:
        if c in t:return c.replace('album of the year','album of year')
    return base.award_cat(text)

def entertainment_extra_veto(r,k,p):
    kt=r['kalshi_title'];pt=r['polymarket_question'];t1=norm(kt);t2=norm(pt)
    # chart rank semantics incl Top N vs exact #1.
    if 'billboard' in (t1+' '+t2):
        a=chart_rank_sem(kt);b=chart_rank_sem(pt)
        if a and b and a!=b:return f'ENT_RANK_SEM:{a}!={b}'
        # song / album / artist axis
        def typ(t):
            if 'top artist' in t or 'artist in 2026' in t:return 'artist'
            if 'billboard 200' in t or 'album' in t:return 'album'
            if 'hot 100' in t or 'song' in t or 'hit' in t:return 'song'
            return None
        x=typ(t1);y=typ(t2)
        if x and y and x!=y:return f'ENT_CHART_ITEM_TYPE:{x}!={y}'
        # item identity: compare title outcome subject, allowing PM artist suffix.
        ea=base.entertainment_entity(kt,k); eb=base.entertainment_entity(pt,p)
        if ea and eb and base.entity_equiv(ea,eb,False) is False:
            return f'ENT_CHART_ENTITY:{canon_entity(ea)}!={canon_entity(eb)}'
    # award ceremony and category
    ae1=award_event(kt);ae2=award_event(pt)
    if ae1 and ae2 and ae1!=ae2:return f'ENT_AWARD_EVENT:{ae1}!={ae2}'
    ac1=award_category_extra(kt);ac2=award_category_extra(pt)
    if ac1 and ac2 and ac1!=ac2:return f'ENT_AWARD_CATEGORY:{ac1}!={ac2}'
    # album feature vs live performance / other feature relation
    if ('featured on' in t1 or 'feature on' in t1) and ('perform' in t2 or 'performance' in t2):return 'ENT_FEATURE_VS_PERFORM'
    if ('featured on' in t2 or 'feature on' in t2) and ('perform' in t1 or 'performance' in t1):return 'ENT_FEATURE_VS_PERFORM'
    return None

# ------------------------------------------------------------------
# MISC: numeric intervals, cross-domain entities, path/deadline semantics
# ------------------------------------------------------------------
def rate_sem(text):
    t=norm(text)
    bank=None
    for key,vals in {'fed':['federal reserve','fed'],'boc':['bank of canada'],'brazil':['central bank of brazil','bank of brazil'],'boj':['bank of japan'],'bok':['bank of korea'],'boi':['bank of israel']}.items():
        if any(v in t for v in vals):bank=key;break
    if not bank:return None
    month=next((m for m in base.MONTHS if m in t or m[:3] in t.split()),None)
    act='increase' if ('hike' in t or 'increase' in t) else ('decrease' if ('cut' in t or 'decrease' in t) else ('no_change' if 'no change' in t else None))
    # bps event set
    m=re.search(r'\b(\d+)\s*[-–]\s*(\d+)\s*bps\b',str(text).lower())
    if m: interval=('range',int(m.group(1)),int(m.group(2)))
    else:
        m=re.search(r'(?:>|more than|greater than)\s*(\d+)\s*bps',str(text).lower())
        if m:interval=('gt',int(m.group(1)))
        else:
            m=re.search(r'\b(\d+)\+\s*bps\b',str(text).lower())
            if m:interval=('ge',int(m.group(1)))
            else:
                m=re.search(r'\b(\d+)\s*bps\b',str(text).lower())
                interval=('exact',int(m.group(1))) if m else None
    return (bank,month,act,interval)

def misc_extra_veto(r,k,p):
    kt=r['kalshi_title'];pt=r['polymarket_question'];t1=norm(kt);t2=norm(pt)
    a=rate_sem(kt);b=rate_sem(pt)
    if a and b and a!=b:return f'MISC_RATE_SET:{a}!={b}'
    # current-dollar/nominal GDP is not safe to equate with unspecified GDP growth.
    if 'gdp' in t1 and 'gdp' in t2:
        n1='current dollar' in t1 or 'nominal' in t1
        n2='current dollar' in t2 or 'nominal' in t2
        r1='real gdp' in t1;r2='real gdp' in t2
        if n1!=n2 or r1!=r2:return 'MISC_GDP_MEASURE'
    # Strict before date versus exactly/on date.
    if ('release' in t1 and 'release' in t2):
        c1=cutoff(kt);c2=cutoff(pt)
        if c1 and c2 and cutoff_equivalent(c1,c2) is False:return f'MISC_RELEASE_CUTOFF:{c1}!={c2}'
        if 'before ' in t1 and re.search(r'\breleased on\b',t2):return 'MISC_BEFORE_VS_ON_DATE'
        if 'before ' in t2 and re.search(r'\breleased on\b',t1):return 'MISC_BEFORE_VS_ON_DATE'
    # Bitcoin path-dependent condition vs one threshold.
    if ('bitcoin' in t1 or 'btc' in t1) and ('bitcoin' in t2 or 'btc' in t2):
        if ('before 100 000' in t1 or 'before 100000' in t1 or 'before 100k' in t1) != ('before 100 000' in t2 or 'before 100000' in t2 or 'before 100k' in t2):
            return 'CRYPTO_PATH_CONDITION'
    # Lead-left is a stricter role than generic lead underwriter.
    if 'underwriter' in t1 and 'underwriter' in t2:
        if ('lead left' in t1)!=('lead left' in t2):return 'MISC_LEAD_LEFT_VS_LEAD'
    # Next-meeting location: explicit place mismatch or finite cutoff on only one side.
    if 'putin' in t1 and 'zelensky' in t1 and 'putin' in t2 and 'zelensky' in t2 and 'meet next' in t1+t2:
        places=['usa','us','united states','ukraine','russia','germany','switzerland']
        def pl(t):
            if 'united states' in t or re.search(r'\busa\b|\bus\b',t):return 'us'
            return next((x for x in places if x in t),None)
        if pl(t1) and pl(t2) and pl(t1)!=pl(t2):return 'MISC_MEETING_LOCATION'
        if bool(cutoff(kt))!=bool(cutoff(pt)):return 'MISC_MEETING_HORIZON'
    # IPO race (which company first) is not equivalent to one company's IPO by a date.
    if 'openai' in t1 and 'anthropic' in t1 and 'ipo first' in t1 and 'ipo by' in t2:return 'MISC_IPO_RACE_VS_DEADLINE'
    if 'openai' in t2 and 'anthropic' in t2 and 'ipo first' in t2 and 'ipo by' in t1:return 'MISC_IPO_RACE_VS_DEADLINE'
    # Explicitly unrelated corporate metrics with same number are not equivalent.
    corp_terms=['amazon','boeing','netflix','velo','sea limited','shopee']
    c1={x for x in corp_terms if x in t1};c2={x for x in corp_terms if x in t2}
    if c1 and c2 and c1.isdisjoint(c2):return 'MISC_CORPORATE_ENTITY'
    return None


def veto(row,k,p):
    # First the demonstrated V3 contradiction set.
    v=base.veto(row,k,p)
    if v:return v
    # Then additional semantic-axis vetoes found by independent audit.
    if k.get('domain')=='sports' or p.get('domain')=='sports' or sport_discipline(row['kalshi_title'],k) or sport_discipline(row['polymarket_question'],p):
        v=sports_extra_veto(row,k,p)
        if v:return v
    if k.get('domain')=='politics' or p.get('domain')=='politics':
        v=politics_extra_veto(row,k,p)
        if v:return v
    if k.get('domain')=='entertainment' or p.get('domain')=='entertainment':
        v=entertainment_extra_veto(row,k,p)
        if v:return v
    v=misc_extra_veto(row,k,p)
    if v:return v
    return None

if __name__=='__main__':
    path='/mnt/data/matcher_review_4119.csv'
    out=[]
    with open(path,encoding='utf8') as f:
        for i,r in enumerate(csv.DictReader(f)):
            k=json.loads(r['ksig']);p=json.loads(r['psig']);reason=veto(r,k,p);out.append((i,bool(reason),reason))
    print('N',len(out),'veto',sum(x[1] for x in out),'pass',sum(not x[1] for x in out))
    print(collections.Counter((x[2] or 'PASS').split(':')[0] for x in out).most_common(80))
