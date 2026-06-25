"""
Optimisation Agent — ranked, cost-aware revenue-improvement plan.
src/airbnb_iip/agents/optimisation.py

The secondary product flow ("I've decided to Airbnb — how do I maximise
revenue?"). Per docs/UC2_Ordered_Task_Backlog.md Phase 7b, it combines three
ideas, each backed by real data rather than hardcoded heuristics:

1. Feature-gap framing (#35): where the property sits versus top-performing
   comparable listings in the same city / district / room-type peer group.
2. Per-amenity uplift magnitude — the defensible number:
     * air conditioning is an actual price-model feature, so its uplift is a
       true *counterfactual*: re-predict nightly price with the flag flipped.
     * the remaining amenities are not model features; their uplift is the
       ``price − price_hat`` residual (median with-minus-without, per city),
       which already controls for location/size/capacity via ``price_hat``.
   Raw observational revenue deltas are deliberately NOT used — they are badly
   confounded (e.g. AC looks negative simply because ~85 % of central listings
   already have it).
3. Apriori-style association rules (#36/#42) as the explanation layer: how much
   more often listings with an amenity land in the top revenue tercile (lift).

The agent owns its data (the price-hat listings table) like the Comparables
agent, and reuses the shared :class:`PricePredictor` service for the AC
counterfactual — "agents call services, not notebooks".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = ROOT / "Data" / "processed" / "listings_with_price_hat.parquet"

REVENUE_COL = "estimated_revenue_l365d"
DISCLAIMER = "Indicative only — uplift estimates are modelled, not guaranteed."

# Realistically owner-addable amenities only (no building-level lifts/gyms or
# location attributes like views). Each: display label, typical install cost
# (EUR), and how its uplift is estimated.
#   "counterfactual" → flip the price-model flag and re-predict.
#   "residual"       → price − price_hat median delta (per city).
_CounterfactualOrResidual = Literal["counterfactual", "residual"]
_AMENITY_CATALOG: dict[str, tuple[str, float, _CounterfactualOrResidual]] = {
    "has_ac":           ("Air conditioning",          1800.0, "counterfactual"),
    "has_workspace":    ("Dedicated workspace",        350.0, "residual"),
    "has_self_checkin": ("Self check-in (smart lock)", 250.0, "residual"),
    "has_dishwasher":   ("Dishwasher",                 450.0, "residual"),
    "has_washer":       ("In-unit washer",             600.0, "residual"),
    "has_dryer":        ("Dryer",                      500.0, "residual"),
    "has_balcony":      ("Stage the balcony/terrace",  200.0, "residual"),
    "has_outdoor_space":("Outdoor seating area",       400.0, "residual"),
    "has_parking":      ("Parking arrangement",       1500.0, "residual"),
    "has_ev_charger":   ("EV charger",                1200.0, "residual"),
    "has_hot_tub":      ("Hot tub",                   5000.0, "residual"),
    "has_pool":         ("Pool",                     25000.0, "residual"),
}

# Guards so we don't recommend something on a handful of noisy rows or surface a
# trivial uplift.
_MIN_GROUP = 30            # min listings with/without an amenity to trust a delta
_MIN_PEER = 20            # min peer-group size for the gap benchmark
_MIN_ANNUAL_UPLIFT = 60.0  # EUR/yr below which a recommendation isn't worth showing
_MAX_PAYBACK_MONTHS = 120  # 10 yrs; beyond this it's not an actionable improvement
_TERCILE = 2 / 3           # "high revenue" = top third within the city


@dataclass
class AmenityRecommendation:
    amenity: str
    name: str
    annual_uplift_eur: float
    investment_eur: float
    payback_months: float
    confidence: Literal["high", "medium", "low"]
    method: str
    lift: float | None
    rationale: str


@dataclass
class OptimisationPlan:
    city: str
    district: str
    room_type: str
    peer_n: int
    peer_median_revenue_eur: float
    peer_target_revenue_eur: float          # P75 of the peer group
    gap_to_top_quartile_eur: float
    projected_annual_nights: int
    recommendations: list[AmenityRecommendation] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "district": self.district,
            "room_type": self.room_type,
            "peer_n": self.peer_n,
            "peer_median_revenue_eur": self.peer_median_revenue_eur,
            "peer_target_revenue_eur": self.peer_target_revenue_eur,
            "gap_to_top_quartile_eur": self.gap_to_top_quartile_eur,
            "projected_annual_nights": self.projected_annual_nights,
            "recommendations": [vars(r) for r in self.recommendations],
            "disclaimer": self.disclaimer,
        }


class OptimisationService:
    """Builds a data-driven, ranked amenity-improvement plan for a property."""

    def __init__(self, data_path: Path | str = DEFAULT_DATA_PATH):
        import pandas as pd

        flags = list(_AMENITY_CATALOG)
        cols = [
            "city", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
            "room_type", "price", "price_hat", REVENUE_COL, "reviews_per_month",
        ] + flags
        df = pd.read_parquet(Path(data_path), columns=cols)
        df = df[(df["price"] > 20) & (df["price"] < 2000)].copy()
        df["_city_norm"] = df["city"].map(_normalize_city)
        df["_residual"] = df["price"] - df["price_hat"]
        self.df = df.reset_index(drop=True)

        # Precompute the two market-derived tables once.
        self._residuals = self._precompute_residuals(flags)
        self._rules = self._precompute_rules(flags)

    # ── public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        property_spec: Mapping[str, Any],
        *,
        projected_annual_nights: float | None = None,
        baseline_net_revenue_eur: float | None = None,  # accepted for parity; not used in the peer gap
    ) -> OptimisationPlan:
        """Return a ranked :class:`OptimisationPlan` for ``property_spec``.

        ``property_spec`` keys: ``city`` (required), ``district`` (or
        ``neighbourhood_cleansed``), ``room_type``, ``accommodates``,
        ``bedrooms``, ``bathrooms`` / ``bathrooms_number``, plus any current
        amenity flags (``has_ac`` … as truthy). Amenities not present in the
        spec are treated as absent and therefore candidates.
        """
        import pandas as pd

        city = _normalize_city(property_spec.get("city", ""))
        if not city:
            raise ValueError("property_spec must include 'city'")
        district = property_spec.get("district") or property_spec.get("neighbourhood_cleansed") or ""
        room_type = property_spec.get("room_type") or "Entire home/apt"

        peer, applied = self._peer_group(city, district, room_type)
        peer_n = len(peer)
        peer_median = float(peer[REVENUE_COL].median()) if peer_n else 0.0
        peer_target = float(peer[REVENUE_COL].quantile(0.75)) if peer_n else 0.0
        gap = max(peer_target - peer_median, 0.0)

        nights = projected_annual_nights
        if nights is None:
            nights = self._estimate_nights(peer)
        nights = int(round(nights))

        recs: list[AmenityRecommendation] = []
        for flag, (label, cost, method) in _AMENITY_CATALOG.items():
            if _has_amenity(property_spec, flag):
                continue  # owner already has it
            nightly_delta = self._nightly_uplift(flag, method, city, property_spec)
            if nightly_delta <= 0:
                continue
            annual_uplift = round(nightly_delta * nights, 0)
            if annual_uplift < _MIN_ANNUAL_UPLIFT:
                continue
            payback_months = round(cost / (annual_uplift / 12.0), 1) if annual_uplift > 0 else 999.0
            if payback_months > _MAX_PAYBACK_MONTHS:
                continue  # uplift is real but the investment never realistically pays back
            lift, conf = self._rule_for(flag, city)
            confidence = self._confidence(method, lift)
            recs.append(AmenityRecommendation(
                amenity=flag,
                name=label,
                annual_uplift_eur=annual_uplift,
                investment_eur=cost,
                payback_months=payback_months,
                confidence=confidence,
                method=method,
                lift=round(lift, 2) if lift is not None else None,
                rationale=self._rationale(label, method, nightly_delta, annual_uplift, lift, conf, city),
            ))

        recs.sort(key=lambda r: r.payback_months)
        return OptimisationPlan(
            city=city,
            district=district or applied.get("district", ""),
            room_type=room_type,
            peer_n=peer_n,
            peer_median_revenue_eur=round(peer_median, 0),
            peer_target_revenue_eur=round(peer_target, 0),
            gap_to_top_quartile_eur=round(gap, 0),
            projected_annual_nights=nights,
            recommendations=recs,
        )

    # ── uplift magnitude ──────────────────────────────────────────────────────

    def _nightly_uplift(
        self, flag: str, method: str, city: str, spec: Mapping[str, Any],
    ) -> float:
        """EUR/night the amenity is worth, holding the property otherwise fixed."""
        if method == "counterfactual":
            return self._counterfactual_nightly(flag, spec)
        return max(self._residuals.get(city, {}).get(flag, 0.0), 0.0)

    def _counterfactual_nightly(self, flag: str, spec: Mapping[str, Any]) -> float:
        """Δ nightly price from flipping a price-model flag (currently has_ac)."""
        try:
            from airbnb_iip.models.price import get_price_predictor
            predictor = get_price_predictor()
        except Exception:
            return 0.0

        base = {
            "accommodates": spec.get("accommodates"),
            "bedrooms": spec.get("bedrooms"),
            "bathrooms_number": spec.get("bathrooms") or spec.get("bathrooms_number"),
            "neighbourhood_cleansed": spec.get("district") or spec.get("neighbourhood_cleansed"),
            "instant_bookable": 1,
        }
        # The price-model feature is the flag itself (e.g. "has_ac"), so pass it
        # through unchanged — only the value flips between the two predictions.
        p_off = predictor.predict({**base, flag: 0})
        p_on = predictor.predict({**base, flag: 1})
        return max(p_on - p_off, 0.0)

    def _precompute_residuals(self, flags: list[str]) -> dict[str, dict[str, float]]:
        """Per-city ``price − price_hat`` median delta (with minus without)."""
        import pandas as pd

        out: dict[str, dict[str, float]] = {}
        for city_norm, sub in self.df.groupby("_city_norm"):
            city_map: dict[str, float] = {}
            for flag in flags:
                with_a = sub.loc[sub[flag] == 1, "_residual"]
                without = sub.loc[sub[flag] == 0, "_residual"]
                if len(with_a) >= _MIN_GROUP and len(without) >= _MIN_GROUP:
                    delta = float(with_a.median() - without.median())
                    if pd.notna(delta):
                        city_map[flag] = delta
            out[str(city_norm)] = city_map
        return out

    # ── association rules (Apriori-style, single-antecedent lift) ──────────────

    def _precompute_rules(self, flags: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
        """Per city: amenity → (lift, confidence) toward the top revenue tercile.

        lift = P(top-tercile | amenity) / P(top-tercile); >1 means the amenity
        is associated with higher-revenue listings. Computed within each city so
        a low-revenue market isn't structurally excluded from "high".
        """
        import pandas as pd

        out: dict[str, dict[str, tuple[float, float]]] = {}
        for city_norm, sub in self.df.groupby("_city_norm"):
            rev = sub[REVENUE_COL]
            if rev.notna().sum() < _MIN_GROUP:
                out[str(city_norm)] = {}
                continue
            thr = rev.quantile(_TERCILE)
            high = (rev >= thr).astype(int)
            base_rate = high.mean() or 1.0
            rules: dict[str, tuple[float, float]] = {}
            for flag in flags:
                present = sub[flag] == 1
                if present.sum() < _MIN_GROUP:
                    continue
                conf = float(high[present].mean())          # P(high | amenity)
                lift = conf / base_rate
                if pd.notna(lift):
                    rules[flag] = (lift, conf)
            out[str(city_norm)] = rules
        return out

    def _rule_for(self, flag: str, city: str) -> tuple[float | None, float | None]:
        rule = self._rules.get(city, {}).get(flag)
        return (rule[0], rule[1]) if rule else (None, None)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _peer_group(self, city: str, district: str, room_type: str):
        """Most specific peer group with enough rows: city+district+room → city+room → city."""
        base = self.df[self.df["_city_norm"] == city]
        district_norm = _normalize_city(district) if district else ""

        if district_norm:
            m = base[
                (base["neighbourhood_group_cleansed"].map(_normalize_city) == district_norm)
                & (base["room_type"] == room_type)
            ]
            if len(m) >= _MIN_PEER:
                return m, {"district": district, "room_type": room_type}
        m = base[base["room_type"] == room_type]
        if len(m) >= _MIN_PEER:
            return m, {"room_type": room_type}
        return base, {}

    def _estimate_nights(self, peer) -> float:
        """Annual nights from the calendar-based LightGBM occupancy model.

        Builds a peer-representative property (modal city / district / room-type,
        median nightly price) and predicts occupancy with the production
        occupancy model, so a standalone /optimise call uses the same occupancy
        basis as the decision engine instead of the retired SF estimator. Only a
        fallback — the UI passes ``projected_annual_nights`` from the scenario.
        """
        try:
            from airbnb_iip.models.occupancy import get_occupancy_predictor

            if len(peer) == 0:
                return 0.40 * 365

            def _mode(col):
                m = peer[col].dropna().mode()
                return m.iloc[0] if len(m) else None

            spec = {
                "city": _mode("city"),
                "neighbourhood_cleansed": _mode("neighbourhood_cleansed"),
                "room_type": _mode("room_type"),
                "price": float(peer["price"].median()),
            }
            rate = get_occupancy_predictor().predict(spec)
            return rate * 365
        except Exception:
            return 0.40 * 365

    @staticmethod
    def _confidence(method: str, lift: float | None) -> Literal["high", "medium", "low"]:
        if method == "counterfactual":
            return "high"  # a genuine model feature
        if lift is None:
            return "low"
        if lift >= 1.20:
            return "high"
        if lift >= 1.05:
            return "medium"
        return "low"

    @staticmethod
    def _rationale(
        label: str, method: str, nightly: float, annual: float,
        lift: float | None, conf: float | None, city: str,
    ) -> str:
        city_disp = city.title()
        if method == "counterfactual":
            base = (
                f"The price model attributes about €{nightly:.0f}/night to {label.lower()}, "
                f"holding location and size fixed — roughly €{annual:,.0f}/yr at your "
                f"projected occupancy."
            )
        else:
            base = (
                f"Comparable {city_disp} listings with {label.lower()} command about "
                f"€{nightly:.0f}/night more than equivalent listings without it "
                f"(≈ €{annual:,.0f}/yr), after controlling for location and size."
            )
        if lift is not None and lift > 1.0:
            base += (
                f" Market signal: such listings reach the top revenue third "
                f"{lift:.2f}× as often."
            )
        return base


# ── module helpers ────────────────────────────────────────────────────────────

def _has_amenity(spec: Mapping[str, Any], flag: str) -> bool:
    return bool(spec.get(flag))


def _normalize_city(value: Any) -> str:
    """Accent-strip + lowercase ("Málaga"/"Salamanca" → "malaga"/"salamanca")."""
    import unicodedata

    ascii_only = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore")
    return ascii_only.decode().strip().lower()


@lru_cache(maxsize=1)
def get_optimisation_service() -> OptimisationService:
    """Process-wide singleton — loads + precomputes the market tables once."""
    return OptimisationService()


def optimise(
    property_spec: Mapping[str, Any],
    *,
    projected_annual_nights: float | None = None,
    baseline_net_revenue_eur: float | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning a plain dict (the API / UI shape)."""
    plan = get_optimisation_service().recommend(
        property_spec,
        projected_annual_nights=projected_annual_nights,
        baseline_net_revenue_eur=baseline_net_revenue_eur,
    )
    return plan.to_dict()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plan = get_optimisation_service().recommend(
        {
            "city": "madrid",
            "district": "Salamanca",
            "room_type": "Entire home/apt",
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms": 1.0,
            "has_elevator": True,  # everything else treated as absent → candidate
        },
        projected_annual_nights=186,
    )
    print(f"Peer n={plan.peer_n} median €{plan.peer_median_revenue_eur:,.0f} "
          f"P75 €{plan.peer_target_revenue_eur:,.0f} gap €{plan.gap_to_top_quartile_eur:,.0f}")
    for r in plan.recommendations:
        print(f"  {r.name:28s} +€{r.annual_uplift_eur:>6,.0f}/yr  "
              f"invest €{r.investment_eur:>6,.0f}  payback {r.payback_months:>5.1f}mo  "
              f"[{r.confidence}/{r.method}] lift={r.lift}")
