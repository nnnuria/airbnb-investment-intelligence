"""Sale-listing amenity extraction from free text.

Idealista *sale* listings carry no structured amenity array (unlike Inside
Airbnb — see :mod:`airbnb_iip.features.amenities`). The signal lives in the
free-text ``description`` and the ``suggestedTexts.title``. This module mines
binary amenity flags from that Spanish prose with keyword regexes, so the
sell-price model can learn an amenity → resale-value relationship.

Two text quirks shape the implementation:

1. **Baked-in encoding corruption.** The scraped text stores accented
   characters as the Unicode replacement char (e.g. ``"Más"`` → ``"M�s"``,
   ``"balcón"`` → ``"balc�n"``). :func:`_normalize` strips accents *and*
   the replacement char, so patterns match ASCII stems (``"balcon"`` and the
   corrupted ``"balcn"`` both reduce predictably). Patterns therefore list both
   forms where an accent is involved.
2. **Negation.** "sin piscina" / "sin garaje" must read as *absent*, not
   present. Each flag has a negation guard: a hit is suppressed when the stem is
   preceded by ``sin`` within two words.

Signal is a **lower bound** — a non-mention is not proof of absence — so a
flag of 0 means "not advertised", which is the contract the model is trained on
and the serving layer mirrors (an un-toggled amenity → 0).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping

import numpy as np
import pandas as pd

# ── Canonical pattern table ──────────────────────────────────────────────────
# column_name → include_regex (matched against accent-stripped lowercase text).
# Where a Spanish term carries an accent, BOTH the accent-stripped form and the
# mojibake-collapsed form are listed (e.g. balcón→"balcon", balc�n→"balcn").
SALE_AMENITY_PATTERNS: dict[str, str] = {
    "has_pool":    r"piscina",
    "has_parking": r"garaje|garage|parking|aparcamiento",
    "has_terrace": r"terraza",
    "has_ac":      r"aire acondicionado|climatiz",
    "has_garden":  r"jardin|jardn",          # jardín / jardines / jard�n
    "has_storage": r"trastero",
    "has_heating": r"calefacc",              # calefacción / calefacci�n
    "has_balcony": r"balcon|balcn",          # balcón / balcones / balc�n
}

SALE_AMENITY_COLUMNS: tuple[str, ...] = tuple(SALE_AMENITY_PATTERNS)

# A stem is read as *absent* when "sin" precedes it within two words
# ("sin piscina", "sin gran piscina", "sin plaza de garaje").
_NEG = r"\bsin\s+(?:\w+\s+){0,2}(?:%s)"


# ── Public API ───────────────────────────────────────────────────────────────

def parse_sale_amenity_flags(
    text: pd.Series,
    patterns: Mapping[str, str] = SALE_AMENITY_PATTERNS,
) -> pd.DataFrame:
    """Parse a free-text Series into binary amenity flag columns.

    Parameters
    ----------
    text
        Series of listing text (description, optionally with the ad title
        appended). NaN/None rows are treated as empty.
    patterns
        Mapping ``column_name → include_regex``. Defaults to
        :data:`SALE_AMENITY_PATTERNS`.

    Returns
    -------
    DataFrame
        One ``int8`` ``{0, 1}`` column per pattern key, indexed like ``text``.
        A row is ``1`` when the include stem matches and is **not** negated by a
        preceding ``sin``.
    """
    norm = text.apply(_normalize)
    compiled = {
        col: (re.compile(inc, re.I), re.compile(_NEG % inc, re.I))
        for col, inc in patterns.items()
    }
    flags: dict[str, np.ndarray] = {}
    for col, (inc, neg) in compiled.items():
        flags[col] = norm.apply(
            lambda s, _i=inc, _n=neg: int(bool(_i.search(s)) and not _n.search(s))
        ).astype(np.int8).to_numpy()
    return pd.DataFrame(flags, index=text.index)


def add_sale_amenity_flags(
    df: pd.DataFrame,
    *,
    desc_col: str = "description",
    title_col: str | None = "_ad_title",
    patterns: Mapping[str, str] = SALE_AMENITY_PATTERNS,
) -> pd.DataFrame:
    """Return a copy of ``df`` with sale amenity flag columns appended.

    Combines ``desc_col`` and (if present) ``title_col`` into the text scanned.
    Idempotent: existing flag columns are overwritten so re-running doesn't
    double-stack. Missing ``desc_col`` is tolerated (treated as empty text) so
    the loader never crashes on an older dump that lacks the field.
    """
    desc = df[desc_col] if desc_col in df.columns else pd.Series("", index=df.index)
    text = desc.fillna("").astype(str)
    if title_col and title_col in df.columns:
        text = text.str.cat(df[title_col].fillna("").astype(str), sep=" ")

    flags = parse_sale_amenity_flags(text, patterns)
    out = df.drop(columns=[c for c in flags.columns if c in df.columns])
    return pd.concat([out, flags], axis=1)


# ── internals ────────────────────────────────────────────────────────────────

def _normalize(value: object) -> str:
    """Lowercase + drop accents and the U+FFFD replacement char.

    NFKD decomposition followed by ASCII-ignore encoding removes both combining
    accents (``ó`` → ``o``) and the baked-in replacement char (``�`` →
    dropped), collapsing every text variant to a single ASCII stem space.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).lower()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
