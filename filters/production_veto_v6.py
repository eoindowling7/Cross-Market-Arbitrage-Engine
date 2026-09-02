
"""Production equivalence veto V6.

V6 keeps V5 intact and adds conservative protections discovered when V5
was applied to the full 11,025 accepted universe.

Changes:
- calendar-boundary equivalence:
    before Jan 1 Y == by Dec 31 Y-1
- Lyon / Olympique Lyonnais alias
- Juventude RS / EC Juventude alias
- Ratinho Junior / Carlos Roberto Massa Junior alias
- prevent cycling "Grand Prix Cycliste" events being parsed as Formula 1

DeBERTa remains the primary matcher.
"""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from datetime import date, timedelta
import re
import csv
import json
import collections


# ------------------------------------------------------------
# Load V5
# ------------------------------------------------------------

_spec = spec_from_file_location(
    "v5",
    str(Path(__file__).with_name("production_veto_v5.py"))
)

v5 = module_from_spec(_spec)
_spec.loader.exec_module(v5)

v4 = v5.v4
base = v5.base
norm = v5.norm


# ------------------------------------------------------------
# Alias protections
# ------------------------------------------------------------

base.TEAM_ALIASES.update({

    # Olympique Lyonnais
    "olympique lyonnais": "lyon",
    "olympique lyonnais fc": "lyon",
    "lyon": "lyon",

    # Esporte Clube Juventude
    "ec juventude": "juventude",
    "juventude rs": "juventude",
    "juventude": "juventude",

    # Same political identity
    "ratinho junior": "ratinho junior",
    "carlos roberto massa junior": "ratinho junior",
})


# ------------------------------------------------------------
# Date-boundary equivalence protection
# ------------------------------------------------------------

_old_cutoff_equivalent = base.cutoff_equivalent


def cutoff_equivalent_v6(a, b):

    if not a or not b:
        return None

    opa, ya, ma, da = a
    opb, yb, mb, db = b

    if a == b:
        return True

    # If both dates have explicit years, compare actual
    # calendar boundaries.
    #
    # Example:
    #
    # before Jan 1 2027
    # ==
    # by Dec 31 2026

    if ya is not None and yb is not None:

        try:
            A = date(
                int(ya),
                int(ma),
                int(da)
            )

            B = date(
                int(yb),
                int(mb),
                int(db)
            )

        except Exception:
            return _old_cutoff_equivalent(a, b)

        if (
            opa == "before"
            and opb == "by"
            and A - timedelta(days=1) == B
        ):
            return True

        if (
            opb == "before"
            and opa == "by"
            and B - timedelta(days=1) == A
        ):
            return True

    return _old_cutoff_equivalent(a, b)


# V3 politics rules resolve this dynamically.
base.cutoff_equivalent = cutoff_equivalent_v6

# V4 also holds its own reference.
v4.cutoff_equivalent = cutoff_equivalent_v6


# ------------------------------------------------------------
# Sports discipline parser protection
# ------------------------------------------------------------

def sport_discipline_v6(title, sig):

    t = v4.sigtext(title, sig)

    if "motogp" in t:
        return "motogp"

    # IMPORTANT:
    # Grand Prix Cycliste is CYCLING, not Formula 1.
    #
    # Cycling therefore gets checked before generic
    # "Grand Prix" detection.

    if re.search(
        r"cycling|cycliste|vuelta a espana|"
        r"tour de france|white jersey",
        t
    ):
        return "cycling"

    if re.search(
        r"\bformula 1\b|\bf1\b|grand prix|constructors champion",
        t
    ):
        return "formula1"

    if "lacrosse" in t:
        return "lacrosse"

    if re.search(
        r"\bnba\b|\bwnba\b|basketball|march madness",
        t
    ):
        return "basketball"

    if re.search(
        r"\bnfl\b|\bcfb\b|ncaa football|college football|"
        r"pro football|quarterback|super bowl",
        t
    ):
        return "american_football"

    if re.search(
        r"\bmlb\b|baseball|world series|silver slugger|cy young",
        t
    ):
        return "baseball"

    if re.search(
        r"\bnhl\b|ice hockey|hockey",
        t
    ):
        return "hockey"

    if re.search(
        r"\bufc\b|\bmma\b|ko or tko|knockout",
        t
    ):
        return "mma"

    if re.search(
        r"\bpga\b|golf|masters tournament|"
        r"ryder cup|solheim cup",
        t
    ):
        return "golf"

    if re.search(
        r"\bfide\b|titled tuesday|chess",
        t
    ):
        return "chess"

    if re.search(
        r"\batp\b|\bwta\b|tennis",
        t
    ):
        return "tennis"

    if re.search(
        r"uefa|fifa|premier league|\bepl\b|la liga|"
        r"bundesliga|serie a|ligue 1|mls|nwsl|efl |"
        r"fa cup|europa league|conference league|copa |"
        r"liga mx|usl |eredivisie|superliga|soccer",
        t
    ):
        return "soccer"

    return None


v4.sport_discipline = sport_discipline_v6


# ------------------------------------------------------------
# Production entry point
# ------------------------------------------------------------

def veto(row, k, p):
    return v5.veto(row, k, p)


# ------------------------------------------------------------
# Standalone regression runner
# ------------------------------------------------------------

if __name__ == "__main__":

    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "matcher_review_4119.csv"
    )

    out = []

    with open(path, encoding="utf8") as f:

        for i, r in enumerate(csv.DictReader(f)):

            k = json.loads(r["ksig"])
            p = json.loads(r["psig"])

            reason = veto(r, k, p)

            out.append(
                (i, bool(reason), reason)
            )

    print(
        "N",
        len(out),
        "veto",
        sum(x[1] for x in out),
        "pass",
        sum(not x[1] for x in out)
    )

    print(
        collections.Counter(
            (x[2] or "PASS").split(":")[0]
            for x in out
        ).most_common(100)
    )
