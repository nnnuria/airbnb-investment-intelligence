"""NPV comparison: keep as Airbnb vs. sell — the product's core decision engine.

Implements Sections 2, 4.4 and 7 of
``docs/INVESTMENT_DECISION_FRAMEWORK.md``. Pure, deterministic financial
functions on top of :mod:`airbnb_iip.finance.costs` (NOI) and
:mod:`airbnb_iip.models.sale` (sale-price prediction) — no ML happens here.

Uncertainty is handled by :func:`monte_carlo`, which resamples ``price_hat``
and ``occ_hat`` around their point estimates and produces P10/P50/P90 NPV
bands plus ``P(Airbnb > Sell)``. The point-estimate path
(:func:`npv_airbnb`, :func:`npv_sell`, :func:`break_even_horizon`) is kept
separate and fully testable without randomness.

Every output is **indicative, not financial or legal advice** — see
:func:`recommend`.
"""

from __future__ import annotations

from typing import Any, Sequence

from airbnb_iip.config import CGT_BRACKETS, FINANCE

DISCLAIMER = "Indicative only — not financial or legal advice."


# ── Capital gains tax (Section 4.4) ──────────────────────────────────────────

def capital_gains_tax(
    gain: float,
    *,
    brackets: Sequence[tuple[float, float]] = tuple(CGT_BRACKETS),
    exempt: bool = False,
) -> float:
    """Progressive Spanish CGT on a sale gain.

    ``brackets`` is a sequence of ``(upper_bound_eur, rate)`` applied like an
    income-tax ladder: the first ``6,000`` of gain is taxed at the first
    bracket's rate, the next slice up to the second bound at the second
    rate, and so on. ``exempt=True`` (primary-residence reinvestment, or
    over-65 selling a primary residence) returns 0.

    Parameters
    ----------
    gain
        ``sale_price - (purchase_price + purchase_costs + improvements)``.
        Negative or zero gain (a loss) owes no CGT.

    Raises
    ------
    ValueError
        If brackets are not strictly increasing in ``upper_bound``.
    """
    if exempt or gain <= 0:
        return 0.0
    bounds = [b for b, _ in brackets]
    if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
        raise ValueError(f"brackets must have strictly increasing bounds, got {bounds}")

    tax = 0.0
    lower = 0.0
    for upper, rate in brackets:
        if gain <= lower:
            break
        slice_amount = min(gain, upper) - lower
        tax += slice_amount * rate
        lower = upper
    return tax


# ── Sell scenario (Section 2, Option A) ──────────────────────────────────────

def npv_sell(
    sale_price: float,
    *,
    purchase_price: float,
    purchase_costs: float = 0.0,
    documented_improvements: float = 0.0,
    agent_rate: float = FINANCE["agent_commission_pct"],
    notary_rate: float = FINANCE["notary_registry_pct"],
    cgt_brackets: Sequence[tuple[float, float]] = tuple(CGT_BRACKETS),
    cgt_exempt: bool = False,
) -> dict[str, float]:
    """Net cash proceeds from selling today — ``NPV_sell`` (Section 2, Option A).

    ``NPV_sell = sale_price × (1 − agent_rate − notary_rate) − CGT(gain)``.
    The sell option is an immediate lump sum, so there is no discounting —
    the "NPV" is simply the net cash the owner walks away with today.

    Returns a dict with ``gain``, ``cgt``, ``transaction_costs`` and
    ``net_proceeds`` (the comparable NPV figure) for a full breakdown.
    """
    if sale_price < 0:
        raise ValueError(f"sale_price must be >= 0, got {sale_price}")
    if not 0 <= agent_rate <= 1 or not 0 <= notary_rate <= 1:
        raise ValueError("agent_rate and notary_rate must be in [0, 1]")

    gain = sale_price - (purchase_price + purchase_costs + documented_improvements)
    cgt = capital_gains_tax(gain, brackets=cgt_brackets, exempt=cgt_exempt)
    transaction_costs = sale_price * (agent_rate + notary_rate)
    net_proceeds = sale_price - transaction_costs - cgt

    return {
        "sale_price": sale_price,
        "gain": gain,
        "cgt": cgt,
        "transaction_costs": transaction_costs,
        "net_proceeds": net_proceeds,
    }


# ── Airbnb scenario (Section 2, Option B) ────────────────────────────────────

def npv_airbnb(
    noi_annual: float | Sequence[float],
    *,
    years: int | None = None,
    terminal_value_net: float = 0.0,
    discount_rate: float = 0.07,
    setup_cost: float = 0.0,
    noi_growth_rate: float = 0.0,
) -> float:
    """Discounted present value of holding the property as an Airbnb.

    ``NPV_airbnb(T) = Σ_t NOI_t / (1+r)^t  +  P_terminal / (1+r)^T  − C_setup``
    (Section 2, Option B).

    Parameters
    ----------
    noi_annual
        Either a constant annual NOI (flat across the horizon — pass
        ``years``) or a sequence of per-year NOI values (one entry per year,
        ``years`` inferred from its length). When a scalar is passed and
        ``noi_growth_rate > 0``, each year's NOI compounds at that rate:
        ``NOI_t = noi_annual × (1 + g)^(t-1)``.
    years
        Holding horizon. Required if ``noi_annual`` is a single number;
        ignored (and inferred) if it is a sequence.
    terminal_value_net
        Net sale proceeds at the end of the horizon (already net of CGT and
        agent fees — typically the ``net_proceeds`` from :func:`npv_sell`
        applied to the appreciated terminal price).
    discount_rate
        Opportunity cost of capital, ``r`` in ``(0, 1)``. Default 7% per the
        framework doc's typical range (6–9%).
    setup_cost
        One-time upfront cost (STR licence, furnishing/refurbishment).
    noi_growth_rate
        Annual NOI growth rate ``g`` (CPI/HPI blend). Applied only when
        ``noi_annual`` is a scalar; ignored when a series is passed explicitly.
        Defaults to 0 (flat NOI, backwards-compatible).
    """
    if discount_rate <= 0:
        raise ValueError(f"discount_rate must be > 0, got {discount_rate}")
    if setup_cost < 0:
        raise ValueError(f"setup_cost must be >= 0, got {setup_cost}")
    if noi_growth_rate < 0:
        raise ValueError(f"noi_growth_rate must be >= 0, got {noi_growth_rate}")

    if isinstance(noi_annual, (int, float)):
        if not years or years <= 0:
            raise ValueError("years must be a positive int when noi_annual is scalar")
        noi_series = [
            float(noi_annual) * (1 + noi_growth_rate) ** t
            for t in range(years)
        ]
    else:
        noi_series = list(noi_annual)
        if not noi_series:
            raise ValueError("noi_annual sequence must not be empty")
        years = len(noi_series)

    pv_noi = sum(
        noi / (1 + discount_rate) ** t for t, noi in enumerate(noi_series, start=1)
    )
    pv_terminal = terminal_value_net / (1 + discount_rate) ** years
    return pv_noi + pv_terminal - setup_cost


def irr_airbnb(
    noi_series: Sequence[float],
    *,
    setup_cost: float = 0.0,
    terminal_value_net: float = 0.0,
) -> float | None:
    """Internal Rate of Return for the Airbnb cash-flow sequence.

    Solves for ``r`` such that NPV = 0:

    ``0 = −C_setup + Σ_t NOI_t / (1+r)^t  +  P_terminal / (1+r)^T``

    Uses ``scipy.optimize.brentq`` bracketed on (−0.5, 2.0). Returns ``None``
    if the cash-flow sequence has no sign change (e.g. always negative) or if
    the solver fails to converge within the bracket.

    Parameters
    ----------
    noi_series
        Per-year NOI values (already tax-adjusted). Length = holding horizon T.
    setup_cost
        One-time upfront investment (subtracted at t=0).
    terminal_value_net
        Net sale proceeds credited at the end of year T.
    """
    from scipy.optimize import brentq

    cash_flows = [-setup_cost] + list(noi_series)
    cash_flows[-1] += terminal_value_net  # add terminal value to last year

    def _npv(r: float) -> float:
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

    try:
        f_lo, f_hi = _npv(-0.50), _npv(2.0)
        if f_lo * f_hi >= 0:
            return None  # no sign change → no real IRR in this range
        return brentq(_npv, -0.50, 2.0, xtol=1e-8, maxiter=200)
    except Exception:
        return None


def break_even_horizon(
    noi_annual: float,
    *,
    npv_sell_value: float,
    discount_rate: float = 0.07,
    terminal_value_fn: Any = None,
    setup_cost: float = 0.0,
    max_years: int = 30,
) -> int | None:
    """Smallest holding horizon ``T*`` at which ``NPV_airbnb(T) > NPV_sell``.

    Assumes constant annual NOI (the common case for a quick sensitivity
    check; pass a custom ``terminal_value_fn`` if appreciation should grow
    the terminal value with ``T``). Returns ``None`` if break-even is not
    reached within ``max_years`` — i.e. selling now is preferable for any
    horizon the owner would realistically consider.

    Parameters
    ----------
    terminal_value_fn
        Callable ``T -> net_terminal_value_eur``. Defaults to a constant 0
        (no terminal sale credited) if omitted — pass the property's
        appreciated, CGT-net sale value as a function of ``T`` for a
        realistic break-even.
    """
    if max_years <= 0:
        raise ValueError(f"max_years must be > 0, got {max_years}")
    terminal_fn = terminal_value_fn or (lambda _t: 0.0)

    for t in range(1, max_years + 1):
        npv = npv_airbnb(
            noi_annual,
            years=t,
            terminal_value_net=terminal_fn(t),
            discount_rate=discount_rate,
            setup_cost=setup_cost,
        )
        if npv > npv_sell_value:
            return t
    return None


# ── Recommendation (Section 2, Decision Rule) ────────────────────────────────

def recommend(npv_airbnb_value: float, npv_sell_value: float) -> str:
    """``"Airbnb"`` if ``NPV_airbnb > NPV_sell``, else ``"Sell"`` (Section 2)."""
    return "Airbnb" if npv_airbnb_value > npv_sell_value else "Sell"


# ── Uncertainty bands (P10/P50/P90) ──────────────────────────────────────────

def monte_carlo(
    price_hat: float,
    occ_hat: float,
    *,
    years: int,
    npv_sell_value: float,
    cost_kwargs: dict[str, Any],
    price_sigma_pct: float = 0.225,
    occ_sigma_pct: float = 0.20,
    discount_rate: float = 0.07,
    setup_cost: float = 0.0,
    terminal_value_net: float = 0.0,
    noi_growth_rate: float = 0.0,
    shock_prob: float = 0.0,
    n_simulations: int = 10_000,
    random_state: int | None = None,
) -> dict[str, float]:
    """Resample price/occupancy uncertainty into NPV_airbnb P10/P50/P90 bands.

    Per Section 7 of the framework doc: price and occupancy are each drawn
    from a Normal centred on the ML point estimate, with sigma a fraction of
    the point estimate (defaults: 22.5% for price, 20% for occupancy — the
    midpoints of the doc's suggested ranges). Each draw computes year-1 NOI
    (via :func:`airbnb_iip.finance.costs.compute_noi`) and applies
    ``noi_growth_rate`` compounding across the holding period to get one
    NPV_airbnb sample.

    ``cost_kwargs`` are passed straight to ``compute_noi`` (minus
    ``gross_revenue``, which this function computes per-sample) — e.g.
    ``property_value``, ``management_fee_rate``, fixed cost lines.

    ``shock_prob`` is the **annual** probability of a regulatory shock
    (licence loss / moratorium / saturation cap) ending STR operation. Each
    simulation draws an independent Bernoulli per year; the first year it
    fires is the licence-loss year, and the Airbnb NOI stream is **truncated**
    from that year onward (those years contribute zero NOI). The terminal
    value is still realised at the end of the horizon — the owner keeps the
    property, they just can no longer let it short-term. ``shock_prob=0``
    (the default) reproduces the no-shock behaviour exactly. See §7.2 of the
    framework doc; per-city probabilities live in
    :data:`airbnb_iip.config.REGULATORY_SHOCK_PROB_BY_CITY`.

    Returns a dict with ``p10``, ``p50``, ``p90`` (NPV_airbnb, EUR) and
    ``p_airbnb_gt_sell`` (share of simulations where Airbnb beats selling
    now) — feeds the API's ``confidence`` field directly.
    """
    import numpy as np

    from airbnb_iip.finance.costs import annual_gross_revenue, compute_noi

    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be > 0, got {n_simulations}")
    if price_hat < 0 or occ_hat < 0:
        raise ValueError("price_hat and occ_hat must be >= 0")
    if not 0 <= shock_prob <= 1:
        raise ValueError(f"shock_prob must be in [0, 1], got {shock_prob}")

    rng = np.random.default_rng(random_state)
    price_draws = rng.normal(price_hat, price_hat * price_sigma_pct, n_simulations).clip(min=0)
    occ_draws = rng.normal(occ_hat, occ_hat * occ_sigma_pct, n_simulations).clip(min=0, max=365)

    # Per-year licence-loss draws. shock_year[i] = first year (1-indexed) a shock
    # fires for simulation i, or 0 if it never fires across the horizon.
    if shock_prob > 0:
        shocks = rng.random((n_simulations, years)) < shock_prob
        any_shock = shocks.any(axis=1)
        shock_year = np.where(any_shock, shocks.argmax(axis=1) + 1, 0)
    else:
        shock_year = np.zeros(n_simulations, dtype=int)

    # Per-year growth factors for the flat-NOI-with-growth path: year t (1-indexed)
    # earns noi_yr1 * (1+g)^(t-1), matching npv_airbnb's scalar compounding.
    growth = (1 + noi_growth_rate) ** np.arange(years)

    npvs = np.empty(n_simulations)
    for i in range(n_simulations):
        gross = annual_gross_revenue(price_draws[i], occ_draws[i])
        noi_yr1 = compute_noi(gross, **cost_kwargs)["noi"]
        noi_series = noi_yr1 * growth
        sy = shock_year[i]
        if sy:
            noi_series = noi_series.copy()
            noi_series[sy - 1:] = 0.0  # licence lost from year `sy` onward
        npvs[i] = npv_airbnb(
            noi_series,
            terminal_value_net=terminal_value_net,
            discount_rate=discount_rate,
            setup_cost=setup_cost,
        )

    p10, p50, p90 = np.percentile(npvs, [10, 50, 90])
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "p_airbnb_gt_sell": float(np.mean(npvs > npv_sell_value)),
    }