import re
import unicodedata


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = (
        text.encode(
            "ascii",
            "ignore"
        )
        .decode("ascii")
        .lower()
    )

    text = re.sub(
        r"[^a-z0-9\s\.\-%]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def extract_year(text):
    text = normalize_text(text)

    match = re.search(
        r"\b(20\d{2})\b",
        text
    )

    if not match:
        return None

    return match.group(1)


def extract_number(text):
    text = normalize_text(text)

    match = re.search(
        r"(\d+(?:\.\d+)?)",
        text
    )

    if not match:
        return None

    return float(
        match.group(1)
    )


# ============================================================
# PROPOSITION TYPE
# ============================================================

def classify_proposition(text):
    """
    Determine what the contract is actually asking.

    This is deliberately conservative.
    """

    text = normalize_text(text)

    if not text:
        return None

    # --------------------------------------------------------
    # Election-stage markets are NOT equivalent to winning
    # the overall election.
    # --------------------------------------------------------

    election_stage_terms = (
        "first round",
        "1st round",
        "second round",
        "2nd round",
        "runoff",
        "run off",
    )

    if any(
        term in text
        for term in election_stage_terms
    ):
        return "election_stage"

    # --------------------------------------------------------
    # Participation is NOT equivalent to winning.
    # --------------------------------------------------------

    participation_terms = (
        "run for",
        "running for",
        "enter the race",
        "enter race",
        "announce a run",
        "declare candidacy",
        "qualify for",
        "make the ballot",
    )

    if any(
        term in text
        for term in participation_terms
    ):
        return "participation"

    # --------------------------------------------------------
    # Primary winner
    # --------------------------------------------------------

    if (
        "primary" in text
        and "win" in text
    ):
        return "primary_winner"

    # --------------------------------------------------------
    # Nomination winner
    # --------------------------------------------------------

    if (
        (
            "nominee" in text
            or "nomination" in text
        )
        and (
            "win" in text
            or "who will" in text
            or "be the" in text
        )
    ):
        return "winner"

    # --------------------------------------------------------
    # General winner
    # --------------------------------------------------------

    if (
        "win" in text
        or "winner" in text
    ):

        blocked = (
            "by more than",
            "over ",
            "under ",
            "total",
            "points",
            "goals",
            "maps",
            "sets",
            "margin",
        )

        if not any(
            term in text
            for term in blocked
        ):
            return "winner"

    # --------------------------------------------------------
    # Above-threshold contracts
    # --------------------------------------------------------

    if any(
        term in text
        for term in (
            "more than",
            "above",
            "at least",
            "over ",
            "greater than",
        )
    ):
        return "above_threshold"

    # --------------------------------------------------------
    # Below-threshold contracts
    # --------------------------------------------------------

    if any(
        term in text
        for term in (
            "less than",
            "below",
            "under ",
            "fewer than",
        )
    ):
        return "below_threshold"

    # --------------------------------------------------------
    # Range contracts
    # --------------------------------------------------------

    if (
        "between" in text
        and "and" in text
    ):
        return "range"

    # --------------------------------------------------------
    # Binary occurrence
    # --------------------------------------------------------

    if any(
        term in text
        for term in (
            "before",
            "happen",
            "occur",
            "retire",
            "resign",
            "leave",
            "out as",
            "banned",
            "unban",
        )
    ):
        return "binary_event"

    return None


# ============================================================
# SUBJECT EXTRACTION
# ============================================================

def extract_winner_subject(text):
    if not isinstance(text, str):
        return None

    patterns = [
        r"^Will (.+?) win\b",
        r"^Will (.+?) be the .+? nominee\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text.strip(),
            flags=re.IGNORECASE
        )

        if match:

            return normalize_text(
                match.group(1)
            )

    return None


# ============================================================
# PRESIDENTIAL COUNTRY DETECTION
# ============================================================

def presidential_country(text):
    """
    Determine presidential-election country only when
    the country is explicit enough to trust.
    """

    text = normalize_text(text)

    if (
        "french" in text
        or "france" in text
    ):
        return "france"

    if (
        "brazil" in text
        or "brazilian" in text
    ):
        return "brazil"

    if (
        "us presidential" in text
        or "u s presidential" in text
        or "united states presidential" in text
    ):
        return "united_states"

    return None


# ============================================================
# TOPIC CLASSIFICATION
# ============================================================

def canonical_topic(text):
    text = normalize_text(text)

    if not text:
        return None

    # --------------------------------------------------------
    # US party nominations
    # --------------------------------------------------------

    if (
        "democratic" in text
        and (
            "presidential nomination" in text
            or "presidential nominee" in text
        )
    ):
        return "democratic_presidential_nomination"

    if (
        "republican" in text
        and (
            "presidential nomination" in text
            or "presidential nominee" in text
        )
    ):
        return "republican_presidential_nomination"

    # --------------------------------------------------------
    # Presidential elections
    # --------------------------------------------------------

    if "presidential election" in text:

        country = presidential_country(
            text
        )

        if country is None:
            return None

        return (
            f"{country}_presidential_election"
        )

    # --------------------------------------------------------
    # Ballon d'Or
    # --------------------------------------------------------

    if "ballon d or" in text:
        return "ballon_dor"

    # --------------------------------------------------------
    # Economics
    # --------------------------------------------------------

    if "inflation" in text:

        if "canada" in text:
            return "canada_inflation"

        return "inflation"

    if "unemployment" in text:
        return "unemployment"

    if (
        "fed" in text
        and (
            "rate" in text
            or "interest" in text
        )
    ):
        return "fed_rates"

    # --------------------------------------------------------
    # Crypto
    # --------------------------------------------------------

    if "bitcoin" in text:
        return "bitcoin"

    if "ethereum" in text:
        return "ethereum"

    return None


# ============================================================
# STRUCTURAL SIGNATURE
# ============================================================

def build_signature(
    title,
    competitor=None,
    event_text=None,
):

    title = str(
        title or ""
    )

    event_text = str(
        event_text or ""
    )

    combined = (
        title
        + " "
        + event_text
    )

    proposition = classify_proposition(
        title
    )

    topic = canonical_topic(
        combined
    )

    if (
        proposition is None
        or topic is None
    ):
        return None

    # --------------------------------------------------------
    # Reject structures we do not yet consider safely
    # equivalent across platforms.
    # --------------------------------------------------------

    if proposition in (
        "participation",
        "primary_winner",
        "election_stage",
    ):
        return None

    year = extract_year(
        combined
    )

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    if competitor:

        subject = normalize_text(
            competitor
        )

    else:

        subject = extract_winner_subject(
            title
        )

    if (
        proposition == "winner"
        and not subject
    ):
        return None

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = None

    if proposition in (
        "above_threshold",
        "below_threshold",
        "range",
    ):

        threshold = extract_number(
            title
        )

        if threshold is None:
            return None

    return {
        "proposition":
            proposition,

        "topic":
            topic,

        "year":
            year,

        "subject":
            subject,

        "threshold":
            threshold,
    }


# ============================================================
# KALSHI ADAPTER
# ============================================================

def kalshi_signature(row):

    title = str(
        row.get(
            "title",
            ""
        )
    )

    yes_sub_title = str(
        row.get(
            "yes_sub_title",
            ""
        )
    ).strip()

    event_ticker = str(
        row.get(
            "event_ticker",
            ""
        )
    )

    subtitle = str(
        row.get(
            "subtitle",
            ""
        )
    )

    event_text = (
        event_ticker
        + " "
        + subtitle
    )

    competitor = (
        yes_sub_title
        if yes_sub_title
        else None
    )

    return build_signature(
        title=title,
        competitor=competitor,
        event_text=event_text,
    )


# ============================================================
# POLYMARKET ADAPTER
# ============================================================

def polymarket_signature(market):

    question = str(
        market.get(
            "question",
            ""
        )
    )

    competitor = str(
        market.get(
            "groupItemTitle",
            ""
        )
    ).strip()

    events = market.get(
        "events",
        []
    )

    event_title = ""
    event_slug = ""

    if events:

        event = events[0]

        event_title = str(
            event.get(
                "title",
                ""
            )
        )

        event_slug = str(
            event.get(
                "slug",
                ""
            )
        )

    event_text = (
        event_title
        + " "
        + event_slug
    )

    return build_signature(
        title=question,
        competitor=(
            competitor
            if competitor
            else None
        ),
        event_text=event_text,
    )


# ============================================================
# EQUIVALENCE CHECK
# ============================================================

def signatures_equivalent(
    left,
    right,
):

    if (
        left is None
        or right is None
    ):
        return False

    if (
        left["proposition"]
        != right["proposition"]
    ):
        return False

    if (
        left["topic"]
        != right["topic"]
    ):
        return False

    # --------------------------------------------------------
    # Explicit years must agree.
    # --------------------------------------------------------

    if (
        left["year"] is not None
        and right["year"] is not None
        and left["year"] != right["year"]
    ):
        return False

    # --------------------------------------------------------
    # Subjects must agree.
    # --------------------------------------------------------

    if (
        left["subject"]
        or right["subject"]
    ):

        if (
            left["subject"]
            != right["subject"]
        ):
            return False

    # --------------------------------------------------------
    # Thresholds must agree.
    # --------------------------------------------------------

    if (
        left["threshold"] is not None
        or right["threshold"] is not None
    ):

        if (
            left["threshold"]
            != right["threshold"]
        ):
            return False

    return True


# ============================================================
# INDEX-BASED DISCOVERY
# ============================================================

def find_structural_matches(
    kalshi_markets,
    polymarket_markets,
):

    poly_index = {}

    # --------------------------------------------------------
    # Index Polymarket signatures
    # --------------------------------------------------------

    for market in polymarket_markets:

        signature = (
            polymarket_signature(
                market
            )
        )

        if signature is None:
            continue

        key = (
            signature[
                "proposition"
            ],
            signature[
                "topic"
            ],
            signature[
                "year"
            ],
            signature[
                "subject"
            ],
            signature[
                "threshold"
            ],
        )

        poly_index.setdefault(
            key,
            []
        ).append(
            market
        )

    # --------------------------------------------------------
    # Match Kalshi signatures
    # --------------------------------------------------------

    matches = []

    for _, row in kalshi_markets.iterrows():

        signature = kalshi_signature(
            row
        )

        if signature is None:
            continue

        key = (
            signature[
                "proposition"
            ],
            signature[
                "topic"
            ],
            signature[
                "year"
            ],
            signature[
                "subject"
            ],
            signature[
                "threshold"
            ],
        )

        candidates = poly_index.get(
            key,
            []
        )

        for market in candidates:

            poly_sig = (
                polymarket_signature(
                    market
                )
            )

            if not signatures_equivalent(
                signature,
                poly_sig,
            ):
                continue

            matches.append({
                "kalshi_ticker":
                    row.get(
                        "ticker"
                    ),

                "kalshi_title":
                    row.get(
                        "title"
                    ),

                "kalshi_yes_sub_title":
                    row.get(
                        "yes_sub_title"
                    ),

                "signature":
                    signature,

                "polymarket_question":
                    market.get(
                        "question"
                    ),

                "polymarket_market":
                    market,
            })

    return matches