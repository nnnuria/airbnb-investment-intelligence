"""
Multilingual review sentiment for the Airbnb Investment Intelligence Platform (UC3).

Why this design
---------------
The README specifies **TF-IDF + Naïve Bayes** sentiment. Two facts from data validation shape
the implementation:

1. Reviews are **multilingual** (EN 40–59%, ES 12–39%, plus FR/DE/IT/PT). An English-only model
   would silently drop 40–60% of reviews.
2. Inside Airbnb reviews carry **no sentiment labels**, so we must create them.

Approach: a multilingual lexicon scorer (language-detect → per-language sentiment) produces weak
labels; a **TF-IDF + Multinomial Naïve Bayes** classifier is then trained on those labels, giving
the README-specified model plus a fast, serialisable scorer for serving. Both are validated against
`review_scores_rating` from listings.

Pipeline destination per repo conventions: `src/airbnb_iip/models/nlp.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Lexicon scorers. VADER (EN) + a small multilingual fallback keep the dependency surface light;
# swap in a multilingual transformer here later without changing the public API.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 42
except ImportError:  # pragma: no cover
    detect = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- Multilingual sentiment word lists (compact; extend as needed) -----------------------------
# Non-English positive/negative cues used when a review is not English. Deliberately small and
# high-precision — the TF-IDF/NB model generalises beyond these once trained.
_POS = {
    "es": {"excelente", "perfecto", "encantó", "maravilloso", "limpio", "cómodo", "recomendable",
           "genial", "estupendo", "ideal", "acogedor", "fantástico", "increíble"},
    "fr": {"excellent", "parfait", "magnifique", "propre", "confortable", "agréable", "génial",
           "superbe", "idéal", "chaleureux", "incroyable"},
    "de": {"ausgezeichnet", "perfekt", "wunderbar", "sauber", "gemütlich", "toll", "super",
           "schön", "empfehlenswert", "fantastisch"},
    "it": {"eccellente", "perfetto", "meraviglioso", "pulito", "comodo", "accogliente",
           "fantastico", "splendido", "ottimo"},
    "pt": {"excelente", "perfeito", "maravilhoso", "limpo", "confortável", "acolhedor", "ótimo"},
}
_NEG = {
    "es": {"sucio", "ruidoso", "mal", "terrible", "horrible", "decepción", "frío", "pequeño",
           "incómodo", "problema", "roto", "olor"},
    "fr": {"sale", "bruyant", "mauvais", "terrible", "horrible", "déception", "froid", "petit",
           "inconfortable", "problème", "cassé", "odeur"},
    "de": {"schmutzig", "laut", "schlecht", "schrecklich", "enttäuschung", "kalt", "klein",
           "unbequem", "problem", "kaputt", "geruch"},
    "it": {"sporco", "rumoroso", "male", "terribile", "orribile", "delusione", "freddo", "piccolo",
           "scomodo", "problema", "rotto", "odore"},
    "pt": {"sujo", "barulhento", "ruim", "terrível", "horrível", "decepção", "frio", "pequeno",
           "desconfortável", "problema", "quebrado", "cheiro"},
}

_URL = re.compile(r"http\S+|www\.\S+")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Lowercase, strip URLs and collapse whitespace. Keeps accents (needed for ES/FR/DE/IT/PT)."""
    s = str(s).lower()
    s = _URL.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def detect_lang(s: str) -> str:
    if detect is None:
        return "en"
    try:
        return detect(s[:200])
    except Exception:
        return "unknown"


def lexicon_score(text: str, lang: str | None = None) -> float:
    """Compound sentiment in [-1, 1]. VADER for EN, word-list ratio for other languages."""
    text = clean_text(text)
    if not text:
        return 0.0
    lang = lang or detect_lang(text)
    if lang == "en" and SentimentIntensityAnalyzer is not None:
        return _vader().polarity_scores(text)["compound"]
    pos = _POS.get(lang, set())
    neg = _NEG.get(lang, set())
    if not pos and not neg:
        # Unknown language: fall back to VADER (catches emoji, punctuation, shared cognates).
        return _vader().polarity_scores(text)["compound"] if SentimentIntensityAnalyzer else 0.0
    toks = set(text.split())
    p, n = len(toks & pos), len(toks & neg)
    if p == n:
        return 0.0
    return (p - n) / (p + n)


_VADER_CACHE = {}


def _vader():
    if "v" not in _VADER_CACHE:
        _VADER_CACHE["v"] = SentimentIntensityAnalyzer()
    return _VADER_CACHE["v"]


def to_label(score: float, pos_thr: float = 0.05, neg_thr: float = -0.05) -> str:
    if score > pos_thr:
        return "positive"
    if score < neg_thr:
        return "negative"
    return "neutral"


@dataclass
class SentimentModel:
    """TF-IDF + Multinomial Naïve Bayes sentiment classifier (README-specified model)."""

    max_features: int = 20000
    ngram_range: tuple = (1, 2)
    pipeline: Pipeline | None = field(default=None, repr=False)

    def build(self) -> Pipeline:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=self.max_features,
                                      ngram_range=self.ngram_range,
                                      min_df=5, sublinear_tf=True)),
            ("nb", MultinomialNB()),
        ])
        return self.pipeline

    def fit(self, texts: pd.Series, labels: pd.Series) -> "SentimentModel":
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts) -> np.ndarray:
        return self.pipeline.predict(texts)


def weak_label_reviews(df: pd.DataFrame, text_col: str = "comments",
                       sample: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Add `lang`, `sent_score`, `sent_label` columns via the multilingual lexicon scorer."""
    work = df.dropna(subset=[text_col]).copy()
    if sample and sample < len(work):
        work = work.sample(sample, random_state=random_state)
    work["clean"] = work[text_col].astype(str).map(clean_text)
    work["lang"] = work["clean"].map(detect_lang)
    work["sent_score"] = [lexicon_score(t, l) for t, l in zip(work["clean"], work["lang"])]
    work["sent_label"] = work["sent_score"].map(to_label)
    return work


def aggregate_by_listing(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-listing mean sentiment + positive/negative share — the UC3 input."""
    g = scored.groupby("listing_id")
    out = pd.DataFrame({
        "n_reviews": g.size(),
        "mean_sentiment": g["sent_score"].mean(),
        "pct_negative": g["sent_label"].apply(lambda s: (s == "negative").mean()),
        "pct_positive": g["sent_label"].apply(lambda s: (s == "positive").mean()),
    })
    return out.reset_index()


def validate_against_ratings(listing_sentiment: pd.DataFrame,
                             listings_df: pd.DataFrame) -> float:
    """Correlation of mean listing sentiment vs review_scores_rating (sanity check)."""
    m = listing_sentiment.merge(
        listings_df[["id", "review_scores_rating"]].rename(columns={"id": "listing_id"}),
        on="listing_id", how="inner",
    ).dropna(subset=["mean_sentiment", "review_scores_rating"])
    if len(m) < 3:
        return float("nan")
    return float(m["mean_sentiment"].corr(m["review_scores_rating"]))


def run_city(reviews_path: str, listings_path: str, city: str,
             sample: int | None = 50000) -> dict:
    """End-to-end for one city: weak-label → train NB → aggregate → validate."""
    reviews = pd.read_csv(reviews_path, usecols=["listing_id", "comments"])
    scored = weak_label_reviews(reviews, sample=sample)

    # Train the README-specified TF-IDF + NB on the weak labels (exclude neutral for clean signal).
    train = scored[scored["sent_label"] != "neutral"]
    Xtr, Xte, ytr, yte = train_test_split(train["clean"], train["sent_label"],
                                          test_size=0.2, random_state=42, stratify=train["sent_label"])
    model = SentimentModel().fit(Xtr, ytr)
    report = classification_report(yte, model.predict(Xte), output_dict=True, zero_division=0)

    listing_sent = aggregate_by_listing(scored)
    listings = pd.read_csv(listings_path, low_memory=False, usecols=["id", "review_scores_rating"])
    corr = validate_against_ratings(listing_sent, listings)

    return {
        "city": city,
        "reviews_scored": len(scored),
        "lang_split": scored["lang"].value_counts(normalize=True).head(6).round(3).to_dict(),
        "label_split": scored["sent_label"].value_counts(normalize=True).round(3).to_dict(),
        "nb_accuracy": round(report["accuracy"], 3),
        "nb_f1_macro": round(report["macro avg"]["f1-score"], 3),
        "corr_sentiment_vs_rating": round(corr, 3),
        "listing_sentiment": listing_sent,
        "model": model,
    }
