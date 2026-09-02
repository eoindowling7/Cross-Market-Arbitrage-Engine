import re, json, unicodedata
from difflib import SequenceMatcher
from datetime import date, timedelta

# Production philosophy: DeBERTa remains the matcher. This module only vetoes
# explicit, high-confidence payout contradictions.

def norm(s):
    s = unicodedata.normalize('NFKD', str(s or '')).encode('ascii','ignore').decode().lower()
    s = s.replace('&',' and ')
    s = re.sub(r"[’'`]",'',s)
    s = re.sub(r'[^a-z0-9]+',' ',s)
    return ' '.join(s.split())

MONTHS = {m:i for i,m in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december'],1)}
MON_ABBR={m[:3]:i for m,i in MONTHS.items()}

def parse_date_like(text):
    t=norm(text)
    month_re='|'.join(MONTHS)
    m=re.search(r'\b(' + month_re + r')\s+(\d{1,2})(?:\s+(20\d{2}))?\b',t)
    if m:
        return (int(m.group(3)) if m.group(3) else None, MONTHS[m.group(1)], int(m.group(2)))
    m=re.search(r'\b('+'|'.join(MON_ABBR)+r')\s+(\d{1,2})(?:\s+(20\d{2}))?\b',t)
    if m:
        return (int(m.group(3)) if m.group(3) else None, MON_ABBR[m.group(1)], int(m.group(2)))
    return None

def cutoff(text):
    t=norm(text)
    dt=parse_date_like(text)
    if dt:
        # find operator immediately before month if possible
        op=None
        if re.search(r'\bbefore\s+(?:'+'|'.join(MONTHS)+r'|'+'|'.join(MON_ABBR)+r')\b',t): op='before'
        elif re.search(r'\b(?:by|on or before|on or prior to)\s+(?:'+'|'.join(MONTHS)+r'|'+'|'.join(MON_ABBR)+r')\b',t): op='by'
        elif re.search(r'\bon\s+(?:'+'|'.join(MONTHS)+r'|'+'|'.join(MON_ABBR)+r')\b',t): op='on'
        return (op,)+dt
    m=re.search(r'\b(before|by)\s+(20\d{2})\b',t)
    if m:return (m.group(1),int(m.group(2)),1,1)
    return None

def cutoff_equivalent(a,b):
    if not a or not b:return None
    opa,ya,ma,da=a;opb,yb,mb,db=b
    if ya is not None and yb is not None and ya!=yb:return False
    # fill same inferred year only for date arithmetic
    if ma==mb and da==db and opa==opb:return True
    # before Sep 16 == by Sep 15
    if ma==mb:
        if opa=='before' and opb=='by' and da==db+1:return True
        if opb=='before' and opa=='by' and db==da+1:return True
    # before Jan 1 Y == before Y
    if ma==mb==1 and da==db==1 and opa==opb=='before':return True
    return False

# ---------- entity normalization ----------
TEAM_ALIASES={
 'rayo vallecano de madrid':'vallecano','rayo vallecano':'vallecano',
 'internazionale':'inter','inter milan':'inter','fc internazionale milano':'inter',
 'hamburger sv':'hamburg','hamburger':'hamburg',
 'turkey':'turkiye','tuerkiye':'turkiye',
 'brondby':'brondby','brndby':'brondby','broendby':'brondby',
 'd c united':'dc','dc united':'dc','dc':'dc',
 'new york red bulls':'new york rb','new york rb':'new york rb',
 'st louis city':'saint louis','st louis city sc':'saint louis','saint louis city':'saint louis','saint louis':'saint louis',
 'paris saint germain':'psg','paris sg':'psg','psg':'psg',
 'queens park rangers':'qpr','qpr':'qpr',
 'stade rennais':'rennes','rennes':'rennes',
 'fc cologne':'koln','1 fc koln':'koln','koln':'koln',
 'slavia prague':'slavia praha','sk slavia praha':'slavia praha','slavia praha':'slavia praha',
 'pato oward':'patricio o ward','patricio o ward':'patricio o ward',
 'czechia':'czech republic','czech republic':'czech republic',
 'ole miss':'mississippi','mississippi rebels':'mississippi',
 'fukuoka softbank hawks':'fukuoka hawks','fukuoka hawks':'fukuoka hawks',
 'bologna fc 1909':'bologna','bologna fc':'bologna','bologna':'bologna',
 'aston villa fc':'aston villa','aston villa':'aston villa',
 'arsenal fc':'arsenal','arsenal':'arsenal',
 'rangers fc':'rangers','rangers':'rangers',
 'djurgardens if':'djurgarden','djurgarden':'djurgarden',
 'halmstads bk':'halmstad','halmstad':'halmstad',
 'erzurumspor fk':'erzurum','erzurum':'erzurum',
 'atletico de madrid':'atletico madrid','atletico madrid':'atletico madrid',
 'real sociedad de futbol':'real sociedad','real sociedad':'real sociedad',
 'deportivo la coruna':'deportivo de la coruna','deportivo de la coruna':'deportivo de la coruna',
 'victoria de guimaraes':'guimaraes','vitoria de guimaraes':'guimaraes','guimaraes':'guimaraes',
}
NICK={'matt':'matthew','mike':'michael','joe':'joseph','chris':'christopher','ben':'benjamin','steve':'steven','stephen':'steven','liz':'elizabeth','bill':'william','tom':'thomas','jim':'james','andy':'andrew','rick':'richard','ted':'theodore','pete':'peter','cam':'cameron'}
DROP={'fc','cf','sc','afc','bc','ac','ca','cd','sd','club','de','del','do','da','das','dos','team','1909','1901','1898','1905','09'}
SPORT_NICKS={'bulldogs','hornets','broncos','rockets','bobcats','aggies','raiders','longhorns','trojans','bruins','utes','commodores','badgers','huskies','seminoles','cyclones','cowboys','owls','rams','aztecs','buckeyes','tigers','rebels','deacons','hokies','roadrunners','panthers','eagles','wildcats','bears','cardinals','spartans','wolverines','ducks','beavers','cougars','mountaineers','mustangs','warhawks','jaguars','knights','pirates','miners','commanders','titans','buccaneers','seahawks','wizards','jazz','raptors','spurs','nets','pelicans','heat','pistons','nationals','jays','rangers','rays','angels','mariners','astros','brewers','giants','padres','whitecaps','earthquakes','kickers','gators','hurricanes','sox','mets','yankees','orioles','twins','phillies'}

def canon_entity(s, sports=False):
    s=norm(s)
    s=re.sub(r'\bj d\b','jd',s);s=re.sub(r'\bj b\b','jb',s)
    s=re.sub(r'\bst louis\b','saint louis',s)
    if s in TEAM_ALIASES:s=TEAM_ALIASES[s]
    toks=[]
    for t in s.split():
        t=NICK.get(t,t)
        if t in DROP:continue
        if sports and t in SPORT_NICKS:continue
        toks.append(t)
    s=' '.join(toks)
    if s in TEAM_ALIASES:s=TEAM_ALIASES[s]
    return s

def entity_equiv(a,b,sports=False):
    a=canon_entity(a,sports);b=canon_entity(b,sports)
    if not a or not b:return None
    # parties
    party_alias={'democratic':'democrat','democrats':'democrat','the democrats':'democrat','democratic party':'democrat','republican':'republican','republicans':'republican','the republicans':'republican','republican party':'republican','independents':'independent','an independent':'independent'}
    a=party_alias.get(a,a);b=party_alias.get(b,b)
    if a==b:return True
    A=a.split();B=b.split();sa=set(A);sb=set(B)
    # Jr/Sr are identity-sensitive; roman numerals may be omitted in common sports naming
    for suf in {'jr','sr'}:
        if (suf in sa)!=(suf in sb) and ((sa-{suf})&(sb-{suf})):return False
    # college State abbreviation: X St. commonly means X State.
    # Normalize only when the other side explicitly has State; do not collapse St. Louis etc.
    if 'st' in sa and 'state' in sb:
        aa=(sa-{'st'})|{'state'}
        if aa==sb or aa <= sb or sb <= aa:return True
    if 'st' in sb and 'state' in sa:
        bb=(sb-{'st'})|{'state'}
        if bb==sa or bb <= sa or sa <= bb:return True
    # college State distinction
    if ('state' in sa)!=('state' in sb) and ((sa-{'state'})&(sb-{'state'})):return False
    # normalize common trailing st to state only when exact other side has state
    if 'st' in sa and 'state' in sb:
        aa=(sa-{'st'})|{'state'}
        if aa==sb:return True
    if 'st' in sb and 'state' in sa:
        bb=(sb-{'st'})|{'state'}
        if bb==sa:return True
    # Roman numeral omission allowed
    ar=sa-{'ii','iii','iv'}; br=sb-{'ii','iii','iv'}
    if ar==br:return True
    if sa<=sb or sb<=sa:
        return True
    # concatenated spor
    ca=''.join(A);cb=''.join(B)
    for suf in ['spor','sport']:
        if ca+suf==cb or cb+suf==ca:return True
    ratio=SequenceMatcher(None,a,b).ratio()
    if ratio>=.88:return True
    if len(A)>=2 and len(B)>=2 and A[-1]==B[-1] and A[0][:1]==B[0][:1] and ratio>=.72:return True
    return False

# ---------- sports ----------
SPORT_COMP_PATTERNS=[
 ('ucl',[r'\bchampions league\b',r'\buefa champions league\b']),
 ('uel',[r'\beuropa league\b',r'\buefa europa league\b']),
 ('uecl',[r'\bconference league\b',r'\buefa conference league\b']),
 ('epl',[r'\benglish premier league\b',r'\bepl\b']),
 ('efl_cup',[r'\befl cup\b',r'\bcarabao cup\b']),('fa_cup',[r'\bfa cup\b']),
 ('efl_championship',[r'\befl championship\b']),('bundesliga',[r'\bbundesliga\b']),('dfb_pokal',[r'\bdfb pokal\b']),
 ('la_liga',[r'\bla liga\b',r'\blaliga\b']),('copa_del_rey',[r'\bcopa del rey\b']),('serie_a',[r'\bserie a\b']),
 ('ligue_1',[r'\bligue 1\b']),('liga_portugal',[r'\bliga portugal\b',r'\bprimeira liga\b']),('eredivisie',[r'\beredivisie\b']),
 ('mls_cup',[r'\bmls cup\b']),('mls_east',[r'\bmls eastern conference\b']),('mls_west',[r'\bmls western conference\b']),
 ('nwsl',[r'\bnwsl\b']),('liga_mx',[r'\bliga mx\b']),('liga_mx_clausura',[r'\bliga mx clausura\b']),
 ('copa_libertadores',[r'\bcopa libertadores\b']),('copa_do_brasil',[r'\bcopa do brasil\b']),
 ('brasileirao',[r'\bbrasileir(?:ao|o)\b']),('argentina_clausura',[r'\bargentina torneo clausura\b',r'\btorneo clausura\b']),
 ('uruguay_primera',[r'\buruguay(?:an)? primera\b']),('peru_liga1',[r'\bperu liga 1\b',r'\bliga 1 peru\b']),
 ('ecuador_ligapro',[r'\becuador ligapro\b',r'\bligapro serie a ecuador\b']),('croatian_hnl',[r'\bcroatian hnl\b']),
 ('danish_superliga',[r'\bdanish superliga\b',r'\bdenmark superliga\b']),('turkish_superlig',[r'\bturkish super lig\b',r'\bsuper lig\b']),
 ('usl_championship',[r'\busl championship\b']),('usl_league_one',[r'\busl league one\b']),
 ('nba',[r'\bnba finals\b',r'\bpro basketball championship\b']),('nba_east',[r'\bnba eastern conference\b']),('nba_west',[r'\bnba western conference\b']),
 ('wnba',[r'\bwnba finals\b',r'\bwomen s pro basketball championship\b',r'\bwomens pro basketball championship\b']),
 ('nfl',[r'\bnfl league championship\b',r'\bpro football championship\b']),('afc',[r'\bafc championship\b']),('nfc',[r'\bnfc championship\b']),
 ('mlb',[r'\bworld series\b',r'\bpro baseball championship\b']),('al',[r'\bamerican league championship\b',r'\balcs\b']),('nl',[r'\bnational league championship\b',r'\bnlcs\b']),
 ('fifa_womens_wc',[r'\bfifa women s world cup\b',r'\bfifa womens world cup\b']),('ryder',[r'\bryder cup\b']),('solheim',[r'\bsolheim cup\b']),
 ('tour_championship',[r'\btour championship\b']),('masters_golf',[r'\bmasters tournament\b']),('british_masters',[r'\bbritish masters\b']),('omega_european_masters',[r'\bomega european masters\b']),
 ('vuelta',[r'\bvuelta a espana\b']),('japan_series',[r'\bjapan series\b']),
]

def comps(text):
    t=norm(text);out=set()
    for key,pats in SPORT_COMP_PATTERNS:
        if any(re.search(p,t) for p in pats):out.add(key)
    # prefer specific Liga MX Clausura over generic Liga MX
    if 'liga_mx_clausura' in out: out.discard('liga_mx')
    # 'Women's Pro Basketball Championship' is WNBA, not NBA.
    if ('women s pro basketball' in t or 'womens pro basketball' in t) and 'wnba' in out:
        out.discard('nba')
    return out

def sports_scope(ticker,title,sig):
    t=norm(title+' '+str(sig.get('context') or '')+' '+str(sig.get('event_identity') or ''))
    pref=str(ticker or '').split('-')[0].upper()
    out=set()
    if 'SCORE' in pref or re.search(r'\b(final score|correct score)\b',t):out.add('exact_score')
    if '2H' in pref or 'second half' in t or '2nd half' in t:out.add('second_half')
    if '1H' in pref or 'first half' in t or '1st half' in t:out.add('first_half')
    if 'F5' in pref or 'first 5 innings' in t or 'after 5 innings' in t:out.add('first5')
    if 'PGAH2H' in pref or (' beat ' in t and 'round' in t):out.add('h2h_round')
    if re.search(r'\bend in a draw\b|\bmatch draw\b',t):out.add('draw')
    if 'CYCLINGSTAGE' in pref or re.search(r'\bwin stage \d+\b',t):out.add('stage_winner')
    if 'CYCLINGJERSEY' in pref or 'jersey' in t:out.add('jersey')
    if 'PLAYOFF' in pref or 'make the playoffs' in t or 'playoff qualifiers' in t:out.add('playoff_qualifier')
    if 'finals qualifiers' in t or 'final qualifiers' in t or 'qualify for' in t and 'championship game' in t:out.add('finalist')
    if 'FINALIST' in pref or 'selected as a finalist' in t:out.add('award_finalist')
    if pref.endswith('QUAL') or 'qualify for' in t or 'reach the' in t and 'championship game' in t:out.add('qualify')
    if 'TREBLE' in pref or 'treble' in t:out.add('treble')
    if 'COMBO' in pref or 'both win' in t or ' and ' in t and ('wins national league cy young and' in t or 'both win the us open' in t):out.add('compound')
    if 'COMPETE' in pref or ' to compete ' in (' '+t+' '):out.add('participation')
    if 'TOP' in pref or re.search(r'\btop\s*\d+\b|\bpodium\b',t):
        m=re.search(r'\btop\s*(\d+)\b',t)
        if m:out.add('top_'+m.group(1))
        if 'podium' in t or 'top 3' in t:out.add('top_3')
    if 'RELEGATION' in pref or 'relegat' in t:out.add('relegation')
    if 'PROMO' in pref or 'promot' in t:out.add('promotion')
    if 'STREAK' in pref or 'winning streak' in t or 'win streak' in t:out.add('streak')
    if 'ROTY' in pref or 'rookie of the year' in t:out.add('rookie_award')
    if 'DPOY' in pref or 'defensive player of the year' in t:out.add('dpoy')
    if 'OPOY' in pref or 'offensive player of the year' in t:out.add('opoy')
    if 'MVP' in pref and 'combo' not in pref.lower() or re.search(r'\bmvp\b',t):out.add('mvp')
    if 'CY YOUNG' in t or 'cy young' in t:out.add('cy_young')
    if 'silver slugger' in t:out.add('silver_slugger')
    if 'heisman' in t:out.add('heisman')
    if 'most home runs' in t:out.add('most_home_runs')
    if 'passing yards' in t:out.add('passing_yards')
    if 'passing touchdowns' in t:out.add('passing_touchdowns')
    if 'rushing yards' in t:out.add('rushing_yards')
    if 'rushing touchdowns' in t:out.add('rushing_touchdowns')
    if 'ranked defense' in t:out.add('ranked_defense')
    if 'fide' in t and 'top 5' in t:out.add('fide_rating_top5')
    if 'titled tuesday' in t and 'top 5' in t:out.add('titled_tuesday_top5')
    # generic match winner via prefix GAME or event identity vs, but don't add to half/score scopes
    if ('GAME' in pref or ' vs ' in (' '+norm(sig.get('event_identity'))+' ') or re.search(r'\bwin on 20\d{2}',t)) and not out.intersection({'second_half','first_half','first5','exact_score'}):
        out.add('match_winner')
    # title/championship target if explicit and not match/qualify/etc
    if any(w in t for w in ['champion','championship','win the','win copa','win la liga','win bundesliga','win the mls cup','win the europa league','win the conference league']) and not out.intersection({'match_winner','second_half','first_half','first5','exact_score','qualify','playoff_qualifier','award_finalist','h2h_round','stage_winner','participation'}):
        out.add('title_winner')
    return out

def sports_outcome_entity(title,sig):
    t=norm(title)
    pats=[
      r'^(?:reg time )?(.+?) wins(?: 2nd half| second half| 1st half| first half)?$',
      r'^final score (.+?) wins \d+ \d+$',
      r'^will (.+?) (?:win|be relegated|finish|qualify|reach|record|have|make|be selected|be starting|start|retire|leave|stay)',
      r'^(.+?) to win the second half',
    ]
    for pat in pats:
        m=re.match(pat,t)
        if m:return m.group(1)
    s=norm(sig.get('subject'))
    s=re.sub(r'^reg time ','',s);s=re.sub(r' wins.*$','',s)
    return s or None

def sports_veto(r,k,p):
    kt=r['kalshi_title'];pt=r['polymarket_question'];pref=r['kalshi_ticker'].split('-')[0]
    ks=sports_scope(pref,kt,k);ps=sports_scope('',pt,p)
    # mutually exclusive granular scopes
    groups=[
      {'match_winner','second_half','first_half','first5','exact_score','h2h_round','draw'},
      {'title_winner','playoff_qualifier','qualify','finalist','participation','stage_winner','h2h_round','treble','compound'},
      {'fide_rating_top5','titled_tuesday_top5'},
      {'mvp','cy_young','rookie_award','dpoy','opoy','silver_slugger','most_home_runs','passing_yards','passing_touchdowns','rushing_yards','rushing_touchdowns','heisman'},
    ]
    for g in groups:
        a=ks&g;b=ps&g
        if a and b and a.isdisjoint(b):return 'SPORT_SCOPE:'+','.join(sorted(a))+'!='+','.join(sorted(b))
    # one-side hard granular match scopes
    hard={'second_half','first_half','first5','exact_score','h2h_round','draw','stage_winner','treble','compound','participation','fide_rating_top5','titled_tuesday_top5'}
    a=ks&hard;b=ps&hard
    if bool(a)!=bool(b):
        # if other side is explicit same textual kind it would parse; safe veto
        return 'SPORT_SCOPE_ONE_SIDE:'+','.join(sorted(a or b))
    # title vs match-level explicit
    if 'match_winner' in ks and ('title_winner' in ps or 'qualify' in ps or 'playoff_qualifier' in ps):return 'SPORT_MATCH_VS_SEASON'
    if 'match_winner' in ps and ('title_winner' in ks or 'qualify' in ks or 'playoff_qualifier' in ks):return 'SPORT_SEASON_VS_MATCH'
    # qualification vs winning target
    if 'qualify' in ks and 'title_winner' in ps:return 'SPORT_QUALIFY_VS_WIN'
    if 'title_winner' in ks and 'qualify' in ps:return 'SPORT_WIN_VS_QUALIFY'
    if 'playoff_qualifier' in ks and 'title_winner' in ps:return 'SPORT_PLAYOFF_VS_TITLE'
    # award finalist vs winner
    if 'award_finalist' in ks and (ps & {'mvp','cy_young','rookie_award','dpoy','opoy','heisman'}):return 'SPORT_FINALIST_VS_AWARD_WIN'
    # top N mismatch explicit
    ka={x for x in ks if x.startswith('top_')};pa={x for x in ps if x.startswith('top_')}
    if ka and pa and ka.isdisjoint(pa):return 'SPORT_TOPN_MISMATCH'
    # competition conflicts - compare named comps only if both explicit. Collapse generic parent facets carefully.
    kc=comps(kt+' '+str(k.get('context') or '')+' '+str(k.get('event_identity') or ''))
    pc=comps(pt+' '+str(p.get('context') or '')+' '+str(p.get('event_identity') or ''))
    # remove parent generic if specific facet present
    def reduce(c):
        c=set(c)
        if c & {'nba_east','nba_west'}:c.discard('nba')
        if c & {'afc','nfc'}:c.discard('nfl')
        if c & {'al','nl'}:c.discard('mlb')
        return c
    kc=reduce(kc);pc=reduce(pc)
    if kc and pc and kc.isdisjoint(pc):return 'SPORT_COMP:'+','.join(sorted(kc))+'!='+','.join(sorted(pc))
    # Explicit structured comp disagreement catches parser-known named competition differences.
    if k.get('competition') and p.get('competition') and k.get('competition')!=p.get('competition'):
        return f'SPORT_STRUCT_COMP:{k.get("competition")}!={p.get("competition")}'
    # Gender / metric / stage explicit, with final/finals normalization
    kg=k.get('gender_scope');pg=p.get('gender_scope')
    if kg and pg and kg!=pg:return f'SPORT_GENDER:{kg}!={pg}'
    km=k.get('metric');pm=p.get('metric')
    if km and pm and km!=pm:return f'SPORT_METRIC:{km}!={pm}'
    st=lambda x: {'final':'finals'}.get(str(x),str(x)) if x else None
    if st(k.get('stage')) and st(p.get('stage')) and st(k.get('stage'))!=st(p.get('stage')):
        return f'SPORT_STAGE:{k.get("stage")}!={p.get("stage")}'
    # matchup identity for match-level rows
    if (ks & {'match_winner','second_half','first_half','first5'}) and (ps & {'match_winner','second_half','first_half','first5'}):
        ke=norm(k.get('event_identity'));pe=norm(p.get('event_identity'))
        if ' vs ' in (' '+ke+' ') and ' vs ' in (' '+pe+' '):
            def cleanmatch(z):
                z=re.sub(r'\b(second half result|first half result|regulation time moneyline|moneyline|fc|cf|sc|afc|bc|ac|club|1909|1901|1898)\b',' ',z)
                return ' '.join(z.split())
            a=cleanmatch(ke);b=cleanmatch(pe)
            A=set(a.split());B=set(b.split());sim=SequenceMatcher(None,a,b).ratio();ov=len(A&B)/max(1,min(len(A),len(B)))
            # Some Polymarket event identities are stored as just 'vs. OPPONENT'.
            # Only veto here when both sides carry enough matchup identity to compare safely.
            one_sided = a.startswith('vs ') or b.startswith('vs ') or a.endswith(' vs') or b.endswith(' vs')
            if not one_sided and len(A)>=3 and len(B)>=3 and sim<.48 and ov<.55:return 'SPORT_MATCHUP:'+a+'!='+b
    # outcome entity mismatch only for comparable simple one-entity markets
    comparable = not (ks & {'compound','treble'}) and not (ps & {'compound','treble'})
    if comparable:
        ea=sports_outcome_entity(kt,k);eb=sports_outcome_entity(pt,p)
        eq=entity_equiv(ea,eb,sports=True) if ea and eb else None
        if eq is False:
            # high-confidence only if both title templates refer to a single outcome entity
            if re.match(r'^(?:reg time )?.+ wins',norm(kt)) or re.match(r'^will .+? (?:win|finish|qualify|record|have|make|be relegated)',norm(kt)):
                if re.match(r'^will .+? (?:win|finish|qualify|record|have|make|be relegated)',norm(pt)) or ' to win the ' in norm(pt):
                    return f'SPORT_ENTITY:{canon_entity(ea,True)}!={canon_entity(eb,True)}'
    return None

# ---------- politics ----------
POL_OFFICES=[
 ('vice_president',[r'vice president',r'vice presidency',r'vice presidential']),('president',[r'president',r'presidential']),
 ('governor',[r'governor',r'governorship',r'gubernatorial']),('attorney_general',[r'attorney general']),('secretary_state',[r'secretary of state']),
 ('us_senate',[r'\bsenate\b']),('us_house',[r'\bhouse\b']),('prime_minister',[r'prime minister']),('senate_majority_leader',[r'senate majority leader'])]

def politics_kind(text,ticker=''):
    t=norm(text);pref=str(ticker).split('-')[0].upper();out=set()
    if 'VPRESNOM' in pref or 'vice presidential nominee' in t or 'vice presidency' in t:out.add('vp_nominee')
    if ('PRESNOM' in pref and 'VPRESNOM' not in pref) or (('presidential nominee' in t or 'presidential nomination' in t or 'nominee for the presidency' in t) and 'vice' not in t):out.add('pres_nominee')
    if 'PRESOUTCOME' in pref or ('presidential election' in t and ('win' in t or 'winner' in t or 'defeat' in t)):out.add('pres_election')
    if 'defeat' in t and 'democratic nominee' in t and 'republican nominee' in t:out.add('pres_exact_outcome')
    if 'ticket' in t and 'president' in t:out.add('pres_ticket')
    if 'caucus' in t or 'primary' in t:out.add('primary')
    if 'run for' in t or re.search(r'\brun for (?:president|governor)',t):out.add('participation')
    if 'announce' in t and ('president' in t or 'presidential' in t):out.add('announcement')
    if 'first' in t and 'announce' in t:out.add('first_announcement')
    if 'governor' in t or 'gubernatorial' in t:out.add('governor')
    if 'attorney general' in t:out.add('attorney_general')
    if 'secretary of state' in t:out.add('secretary_state')
    if 'house' in t and ('race' in t or 'seat' in t or 'election' in t):out.add('house')
    if 'senate' in t and ('race' in t or 'election' in t):out.add('senate')
    if 'prime minister' in t:out.add('prime_minister')
    if 'senate majority leader' in t:out.add('senate_majority_leader')
    if 'impeach' in t:out.add('impeachment')
    if 'removed from office' in t:out.add('removal')
    if 'resign' in t or 'departure' in t or 'leave the trump administration' in t:out.add('departure')
    if 'next prime minister' in t or 'next senate majority leader' in t:out.add('next_officeholder')
    if 'arrest' in t:out.add('arrest')
    if 'de facto leader' in t or 'head of state' in t:out.add('leader_status')
    if 'out as leader' in t or 'out as dnc chair' in t:out.add('out_by_date')
    return out

def party(text):
    t=norm(text)
    if 'republican' in t or 'republicans' in t:return 'R'
    if 'democrat' in t or 'democrats' in t or 'democratic' in t:return 'D'
    if 'independent' in t:return 'I'
    return None

def house_district(text):
    t=norm(text)
    # preserve AL at-large
    m=re.search(r'\b([a-z]{2})\s*(?:0)?(\d{1,2}|al)\b',t)
    if not m or 'house' not in t:return None
    return (m.group(1),m.group(2).lstrip('0') or '0',party(t))

def state_name(text):
    t=norm(text)
    states=['new hampshire','new jersey','new mexico','new york','north carolina','north dakota','south carolina','south dakota','west virginia','rhode island','massachusetts','pennsylvania','connecticut','california','washington','minnesota','mississippi','tennessee','wisconsin','maryland','colorado','arizona','alabama','alaska','arkansas','florida','georgia','hawaii','idaho','illinois','indiana','iowa','kansas','kentucky','louisiana','maine','michigan','missouri','montana','nebraska','nevada','ohio','oklahoma','oregon','texas','utah','vermont','virginia','wyoming','delaware']
    return next((s for s in states if s in t),None)

def margin_band(text):
    t=norm(text)
    if not ('%' in str(text) or 'percent' in t or 'margin' in t):return None
    nums=[float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*%?',t)]
    # year pollution removed
    nums=[x for x in nums if x<100]
    if 'or more' in t or 'at least' in t:return ('ge',nums[-1] if nums else None)
    if 'between' in t or (len(nums)>=2 and '-' in str(text)):
        return ('range',nums[-2] if len(nums)>=2 else None,nums[-1] if nums else None)
    return ('margin',tuple(nums[-2:]))

def politics_outcome_entity(title,sig):
    t=norm(title)
    # party-based district/state outcomes
    if ('house' in t or 'governor' in t or 'gubernatorial' in t or 'senate race' in t or 'presidency' in t or 'presidential election' in t) and party(t):
        # Don't override named candidate nominations that contain party adjectives
        if not re.search(r'^will [a-z].+? (?:be the|win the).*?(?:nominee|nomination)',t):
            return {'R':'republican','D':'democratic','I':'independent'}[party(t)]
    if 'defeat' in t and 'presidential election' in t:return None
    m=re.match(r'^will (.+?) (?:be|win|become|announce|serve|lose|finish|run|resign|leave)',t)
    if m:return m.group(1)
    return norm(sig.get('subject')) or None

def politics_veto(r,k,p):
    kt=r['kalshi_title'];pt=r['polymarket_question'];pref=r['kalshi_ticker'].split('-')[0]
    kk=politics_kind(kt+' '+str(k.get('context') or '')+' '+str(k.get('event_identity') or ''),pref)
    pk=politics_kind(pt+' '+str(p.get('context') or '')+' '+str(p.get('event_identity') or ''),'')
    # outcome-type exclusivity
    groups=[
      {'vp_nominee','pres_nominee','pres_election','pres_exact_outcome','pres_ticket','primary'},
      {'participation','pres_nominee','pres_election','governor'},
      {'departure','next_officeholder'},
      {'leader_status','out_by_date'},
    ]
    for g in groups:
        a=kk&g;b=pk&g
        if a and b and a.isdisjoint(b):return 'POL_SCOPE:'+','.join(sorted(a))+'!='+','.join(sorted(b))
    # Participation/run is not the same as winning an office or nomination.
    if 'participation' in kk and (pk & {'governor','pres_nominee','pres_election'}):return 'POL_PARTICIPATION_VS_OUTCOME'
    if 'participation' in pk and (kk & {'governor','pres_nominee','pres_election'}):return 'POL_PARTICIPATION_VS_OUTCOME'
    # first announcement vs ordinary announcement
    if ('first_announcement' in kk)!=( 'first_announcement' in pk) and ('announcement' in kk or 'announcement' in pk):return 'POL_FIRST_ANNOUNCEMENT'
    # office mismatch, based raw explicit office types
    office_k={x for x in ['governor','attorney_general','secretary_state','senate','house','prime_minister','senate_majority_leader'] if x in kk}
    office_p={x for x in ['governor','attorney_general','secretary_state','senate','house','prime_minister','senate_majority_leader'] if x in pk}
    if office_k and office_p and office_k.isdisjoint(office_p):return 'POL_OFFICE:'+','.join(office_k)+'!='+','.join(office_p)
    # presidential VP/president office scope from kinds catches above
    # margin-qualified outcome vs simple winner
    mk=margin_band(kt);mp=margin_band(pt)
    if bool(mk)!=bool(mp):return 'POL_MARGIN_QUALIFIER'
    if mk and mp and mk!=mp:return f'POL_MARGIN:{mk}!={mp}'
    # house district / party
    hk=house_district(kt);hp=house_district(pt)
    if hk and hp and hk!=hp:return f'POL_HOUSE_DIST:{hk}!={hp}'
    # state if both explicit for same office types
    sk=state_name(kt);sp=state_name(pt)
    if sk and sp and sk!=sp and (office_k or office_p):return f'POL_STATE:{sk}!={sp}'
    # party mismatch for party-based election questions
    pkty=party(kt);ppty=party(pt)
    if pkty and ppty and pkty!=ppty and (kk&{'house','governor','senate'} or pk&{'house','governor','senate'}):return f'POL_PARTY:{pkty}!={ppty}'
    # national German election vs named state election
    if ('bundestag' in norm(kt) or 'german federal' in norm(kt)) and 'sachsen anhalt' in norm(pt):return 'POL_NATIONAL_VS_STATE'
    if ('bundestag' in norm(pt) or 'german federal' in norm(pt)) and 'sachsen anhalt' in norm(kt):return 'POL_NATIONAL_VS_STATE'
    # impeachment removal is stricter
    if ('removal' in kk)!=( 'removal' in pk) and ('impeachment' in kk and 'impeachment' in pk):return 'POL_IMPEACH_REMOVAL'
    # cutoff checks for deadline-style events
    if (kk|pk) & {'announcement','departure','arrest','out_by_date'}:
        a=cutoff(kt);b=cutoff(pt)
        eq=cutoff_equivalent(a,b)
        if eq is False:return f'POL_CUTOFF:{a}!={b}'
    # candidate-specific vs party winner is not payoff equivalent
    ea=politics_outcome_entity(kt,k);eb=politics_outcome_entity(pt,p)
    if ea and eb:
        ce=entity_equiv(ea,eb,False)
        if ce is False:
            # Apply only when both are clearly outcome identities for same election-style kind
            if (kk|pk)&{'vp_nominee','pres_nominee','pres_election','governor','senate','house'}:
                return f'POL_ENTITY:{canon_entity(ea)}!={canon_entity(eb)}'
    return None

# ---------- entertainment ----------
AWARD_CATS=[
 'outstanding supporting actor in a limited or anthology series or movie','outstanding supporting actress in a limited or anthology series or movie','outstanding lead actor in a limited or anthology series or movie','outstanding lead actress in a limited or anthology series or movie',
 'outstanding supporting actor in a comedy series','outstanding supporting actress in a comedy series','outstanding supporting actor in a drama series','outstanding supporting actress in a drama series',
 'outstanding lead actor in a drama series','outstanding lead actress in a drama series','outstanding lead actor in a comedy series','outstanding lead actress in a comedy series',
 'outstanding guest actor in a drama series','outstanding guest actress in a drama series','outstanding guest actor in a comedy series','outstanding guest actress in a comedy series',
 'outstanding comedy series','outstanding drama series','outstanding variety special live','outstanding variety series','outstanding reality competition series','outstanding unstructured reality program','outstanding short form comedy drama or variety series',
 'best actor','best supporting actor','best picture','best original screenplay','best director','best rap album','album of the year','album of year'
]

def award_cat(text):
    t=norm(text)
    t=t.replace(' a comedy series',' in a comedy series').replace(' a drama series',' in a drama series').replace(' a limited or anthology series or movie',' in a limited or anthology series or movie')
    for c in sorted(AWARD_CATS,key=len,reverse=True):
        cc=norm(c)
        if cc in t:return cc.replace('album of the year','album of year')
    return None

def chart_axis(text, extra=""):
    raw=str(text or '');t=norm(raw+' '+str(extra or ''));x={}
    if 'billboard' in t:
        x['platform']='billboard'
        x['type']='album' if 'billboard 200' in t or ('album' in t and 'hot 100' not in t) else 'song'
    elif 'spotify' in t or 'top song for 2026' in t or 'top song in the us for 2026' in t or 'top album for 2026' in t:
        x['platform']='spotify'
        x['type']='album' if 'album' in t else 'song'
    elif 'song of the summer' in t:
        x['platform']='song_of_summer';x['type']='song'
    else:return None
    if 'billboard' in t and ('be on the billboard' in t) and not re.search(r'#\s*\d+|\bnumber\s+\d+',raw.lower()):x['rank']='presence'
    m=re.search(r'#\s*(\d+)|\bnumber\s+(\d+)\b',raw.lower())
    if m:x['rank']=int(next(g for g in m.groups() if g))
    if 'top song' in t or 'top album' in t:x['rank']=1
    if 'for 2026' in t or 'wrapped' in t:x['period']='annual'
    if 'week of' in t or 'this week' in t or 'during the week' in t:x['period']='weekly'
    # week date
    d=parse_date_like(raw)
    if d:x['date']=(d[1],d[2])
    if ' in the us ' in ' '+t+' ' or ' usa ' in ' '+t+' ':x['region']='us'
    elif x['platform']=='spotify':x['region']='global'
    return x

def netflix_axis(text):
    raw=str(text);t=norm(raw)
    if 'netflix' not in t:return None
    x={'region':'us' if re.search(r'\bus\b',t) else ('global' if 'global' in t else None),'type':'movie' if 'movie' in t else ('show' if 'show' in t else None)}
    m=re.search(r'#\s*(\d+)|\bnumber\s+(\d+)',raw.lower())
    if m:x['rank']=int(next(g for g in m.groups() if g))
    elif 'top ' in t:x['rank']=1
    m=re.search(r'(?:at least|above|over)\s*(\d+)\s*(?:million|m)\b',t)
    if m:x['views_m']=int(m.group(1))
    m=re.search(r'season\s+(\d+)',t)
    if m:x['season']=int(m.group(1))
    return x

def entertainment_entity(title,sig):
    t=norm(title)
    m=re.match(r'^will "?(.+?)"? (?:be|win|release|have|rank|finish|get|perform)',t)
    if m:return m.group(1)
    return norm(sig.get('subject')) or None

def entertainment_veto(r,k,p):
    kt=r['kalshi_title'];pt=r['polymarket_question'];t1=norm(kt);t2=norm(pt)
    # awards
    c1=award_cat(kt+' '+str(k.get('context') or ''));c2=award_cat(pt+' '+str(p.get('context') or ''))
    if c1 and c2 and c1!=c2:return f'ENT_AWARD_CAT:{c1}!={c2}'
    n1='nominated for' in t1;n2='nominated for' in t2
    w1=' win ' in ' '+t1+' ';w2=' win ' in ' '+t2+' '
    if n1!=n2 and (n1 or n2) and (w1 or w2):return 'ENT_WIN_VS_NOMINATION'
    # chart axes
    a=chart_axis(kt, str(k.get('context') or '')+' '+str(k.get('event_identity') or ''));b=chart_axis(pt, str(p.get('context') or '')+' '+str(p.get('event_identity') or ''))
    if a and b:
        for fld in ['platform','type','region','period','rank','date']:
            if fld in a and fld in b and a[fld]!=b[fld]:return f'ENT_CHART_{fld}:{a[fld]}!={b[fld]}'
        # presence vs exact rank when one lacks explicit key due parse
        if a.get('rank')=='presence' and isinstance(b.get('rank'),int):return 'ENT_CHART_PRESENCE_VS_RANK'
        if b.get('rank')=='presence' and isinstance(a.get('rank'),int):return 'ENT_CHART_PRESENCE_VS_RANK'
    elif bool(a)!=bool(b):
        # chart vs fundamentally non-chart event
        other=t2 if a else t1
        if any(z in other for z in ['release','headliner','perform','delay','game of the year','engaged']):return 'ENT_CHART_VS_OTHER'
    # Netflix
    a=netflix_axis(kt);b=netflix_axis(pt)
    if a and b:
        for fld in ['region','type','rank','views_m','season']:
            if a.get(fld) is not None and b.get(fld) is not None and a[fld]!=b[fld]:return f'ENT_NETFLIX_{fld}:{a[fld]}!={b[fld]}'
    # release / delay / performance semantics
    axes=[]
    def kind(t):
        s=set()
        if 'release' in t:s.add('release')
        if 'delay' in t:s.add('delay')
        if 'headliner' in t or 'headline' in t:s.add('headliner')
        if 'perform' in t:s.add('perform')
        if 'game of the year' in t:s.add('game_award')
        if 'engaged' in t or 'get engaged' in t:s.add('engagement')
        if 'die in' in t:s.add('death_in_story')
        return s
    a1=kind(t1);a2=kind(t2)
    # mutually exclusive core event types
    core={'release','delay','headliner','perform','game_award','engagement','death_in_story'}
    x=a1&core;y=a2&core
    if x and y and x.isdisjoint(y):return 'ENT_EVENT:'+','.join(x)+'!='+','.join(y)
    # headliner is stricter than perform
    if ('headliner' in x and 'perform' in y) or ('perform' in x and 'headliner' in y):return 'ENT_HEADLINER_VS_PERFORM'
    # joint album vs solo album
    if 'release' in x and 'release' in y:
        if ('joint album' in t1)!=( 'joint album' in t2):return 'ENT_JOINT_VS_SOLO_RELEASE'
        ca=cutoff(kt);cb=cutoff(pt)
        if ca and cb and cutoff_equivalent(ca,cb) is False:return f'ENT_RELEASE_CUTOFF:{ca}!={cb}'
        # in 2026 is equivalent to by Dec 31 2026; handled as no conflict if only one cutoff
        if 'in 2026' in t1 and cb:
            if cb[1] in (None,2026) and cb[2:]==(12,31) and cb[0]=='by':pass
            else:return 'ENT_RELEASE_YEAR_VS_EARLY_CUTOFF'
        elif 'in 2026' in t2 and ca:
            if ca[1] in (None,2026) and ca[2:]==(12,31) and ca[0]=='by':pass
            else:return 'ENT_RELEASE_YEAR_VS_EARLY_CUTOFF'
    # annual #1 any-week vs one specific week
    if re.search(r'have a #?1 (?:album|hit).*this year',t1) and ('week of' in t2 or 'this week' in t2):return 'ENT_ANNUAL_ANY_WEEK_VS_SPECIFIC_WEEK'
    if re.search(r'have a #?1 (?:album|hit).*this year',t2) and ('week of' in t1 or 'this week' in t1):return 'ENT_ANNUAL_ANY_WEEK_VS_SPECIFIC_WEEK'
    # #1 hit/song vs album
    if ('#1 hit' in t1 or '# 1 hit' in t1) and 'album' in t2:return 'ENT_SONG_VS_ALBUM'
    if ('#1 hit' in t2 or '# 1 hit' in t2) and 'album' in t1:return 'ENT_SONG_VS_ALBUM'
    # entity mismatch only within aligned award/chart/netflix/basic event
    ea=entertainment_entity(kt,k);eb=entertainment_entity(pt,p)
    if ea and eb:
        eq=entity_equiv(ea,eb,False)
        if eq is False and ((a and b) or (c1 and c2) or (x and y and not x.isdisjoint(y))):return f'ENT_ENTITY:{canon_entity(ea)}!={canon_entity(eb)}'
    return None

# ---------- economics/weather/tech/crypto/other ----------
def parse_rate(text):
    t=norm(text);bank=None
    for key,vals in {'fed':['federal reserve','fed'],'boc':['bank of canada'],'brazil':['central bank of brazil'],'boj':['bank of japan'],'bok':['bank of korea'],'boi':['bank of israel']}.items():
        if any(v in t for v in vals):bank=key;break
    if not bank:return None
    month=next((m for m in MONTHS if m in t or m[:3] in t.split()),None)
    act='increase' if ('hike' in t or 'increase' in t) else ('decrease' if ('cut' in t or 'decrease' in t) else ('no_change' if 'no change' in t else None))
    nums=[int(x) for x in re.findall(r'\b(\d+)\s*bps\b',t)]
    bps=nums[-1] if nums else None
    gt='more than' in t or '>' in str(text)
    return bank,month,act,bps,gt

def misc_veto(r,k,p):
    d=k.get('domain');kt=r['kalshi_title'];pt=r['polymarket_question'];t1=norm(kt);t2=norm(pt)
    # structured domain mismatch is suspicious but don't veto other<->sports/econ merely from parser
    # central-bank decisions
    a=parse_rate(kt);b=parse_rate(pt)
    if a and b and a!=b:return f'MISC_RATE:{a}!={b}'
    # GDP
    if 'gdp' in t1 and 'gdp' in t2:
        if ('real gdp' in t1)!=( 'real gdp' in t2) and ('current dollar' in t1 or 'nominal' in t1 or 'real gdp' in t2 or 'current dollar' in t2 or 'nominal' in t2):return 'MISC_GDP_REAL_NOMINAL'
        q1=next((q for q in ['q1','q2','q3','q4'] if q in t1),None);q2=next((q for q in ['q1','q2','q3','q4'] if q in t2),None)
        if bool(q1)!=bool(q2):return 'MISC_GDP_QUARTER_VS_ANNUAL'
        if q1 and q2 and q1!=q2:return 'MISC_GDP_QUARTER'
    # WTI date
    if 'wti' in t1 and 'wti' in t2:
        d1=parse_date_like(kt);d2=parse_date_like(pt)
        if d1 and d2 and d1[1:]!=d2[1:]:return f'MISC_WTI_DATE:{d1}!={d2}'
    # rain same city/date is no veto; different city/date explicit is veto
    if 'rain' in t1 and 'rain' in t2:
        d1=parse_date_like(kt);d2=parse_date_like(pt)
        if d1 and d2 and d1[1:]!=d2[1:]:return 'WEATHER_RAIN_DATE'
        # city from "in X on"
        m1=re.search(r'will it rain in (.+?) on ',t1);m2=re.search(r'will it rain in (.+?)(?: ca| wa| az| pa| ny| la| fl| tx| co| il| ma| ga)? on ',t2)
        if m1 and m2 and norm(m1.group(1))!=norm(m2.group(1)):
            # NYC alias
            a=norm(m1.group(1)).replace('new york city','new york');b=norm(m2.group(1)).replace('new york city','new york')
            if a!=b:return 'WEATHER_RAIN_LOCATION'
    # hurricane location explicit
    if 'hurricane' in t1 and 'hurricane' in t2:
        if ('north carolina' in t1)!=( 'north carolina' in t2):return 'WEATHER_HURRICANE_LOCATION'
    # tech Gemini model identity and cutoff
    if 'gemini' in t1 and 'gemini' in t2:
        def model(t):
            if 'flash' in t:return 'flash'
            if re.search(r'\b3 5 pro\b',t):return '3.5_pro'
            if 'pro' in t:return 'pro'
            return 'gemini'
        if model(t1)!=model(t2):return f'TECH_MODEL:{model(t1)}!={model(t2)}'
        neg1='not be released' in t1 or 'no next' in t1;neg2='not be released' in t2 or 'no next' in t2
        if neg1!=neg2:return 'TECH_RELEASE_POLARITY'
        c1=cutoff(kt);c2=cutoff(pt)
        if c1 and c2 and cutoff_equivalent(c1,c2) is False:return f'TECH_CUTOFF:{c1}!={c2}'
    # other model Astra cutoff/polarity
    if 'astra' in t1 and 'astra' in t2:
        neg1='not be released' in t1 or 'no next' in t1;neg2='not be released' in t2 or 'no next' in t2
        if neg1!=neg2:return 'TECH_RELEASE_POLARITY'
    # crypto speech/post, path dependence
    if ('bitcoin' in t1 or 'crypto' in t1) and ('bitcoin' in t2 or 'crypto' in t2):
        if ('truth social' in t1)!=( 'truth social' in t2) and ('say' in t1 or 'post' in t2):return 'CRYPTO_SPEECH_VS_POST'
        if ('before 100 000' in t1 or 'before 100000' in t1 or 'before 100k' in t1) and ('100k' in t2 or '100 000' in t2 or '100000' in t2):return 'CRYPTO_PATH_VS_THRESHOLD'
    return None

def veto(row,k,p):
    # Return None to allow DeBERTa match, or string reason to reject.
    d=k.get('domain');pd=p.get('domain')
    # hard structured exact counts/ranks/thresholds where both explicit and semantically same metric
    # Domain-specific
    if d=='sports' or pd=='sports':
        v=sports_veto(row,k,p)
        if v:return v
    if d=='politics' or pd=='politics':
        v=politics_veto(row,k,p)
        if v:return v
    if d=='entertainment' or pd=='entertainment':
        v=entertainment_veto(row,k,p)
        if v:return v
    v=misc_veto(row,k,p)
    if v:return v
    return None

if __name__=='__main__':
    import csv, collections, sys
    path=sys.argv[1] if len(sys.argv)>1 else '/mnt/data/matcher_review_4119.csv'
    out=[]
    with open(path,encoding='utf8') as f:
        for i,r in enumerate(csv.DictReader(f)):
            k=json.loads(r['ksig']);p=json.loads(r['psig']);reason=veto(r,k,p)
            out.append((i, bool(reason), reason))
    print('N',len(out),'veto',sum(x[1] for x in out),'pass',sum(not x[1] for x in out))
    print(collections.Counter((x[2] or 'PASS').split(':')[0] for x in out).most_common(50))
