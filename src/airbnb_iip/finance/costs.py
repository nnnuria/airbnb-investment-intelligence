"""Airbnb gross revenue and Net Operating Income (NOI) — pure functions.

Implements Sections 3–4 of ``docs/INVESTMENT_DECISION_FRAMEWORK.md``: turn
``price_hat`` (from the price model) and ``occ_hat`` (from the occupancy
model) into annual gross revenue, then deduct platform fees, cleaning,
fixed running costs, management, and tax to get NOI.

Nothing here touches the sale side or NPV discounting — that is
:mod:`airbnb_iip.finance.scenarios`. Everything below is a pure function
of its arguments; defaults come from :data:`airbnb_iip.config.FINANCE` and
:data:`airbnb_iip.config.TOURIST_TAX_EUR_PER_GUEST_NIGHT` but every default
can be overridden by the caller for sensitivity analysis or Monte Carlo.
"""

from __future__ import annotations

from airbnb_iip.config import FINANCE, TOURIST_TAX_EUR_PER_GUEST_NIGHT

DAYS_PER_YEAR = 365


def annual_gross_revenue(price_per_night: float, occupied_nights_per_year: float) -> float:
    """Gross Airbnb revenue for the year, before any deductions.

    ``Annual_gross = price_per_night × occupied_nights_per_year``. Seasonality
    (monthly multipliers) redistributes *when* nights are booked but — under
    the constant-nightly-price assumption used here — does not change the
    annual total, so it is intentionally not a parameter of this function.
    Apply seasonal multipliers to ``occupied_nights_per_year`` upstream if you
    need a month-by-month series (see Section 5.5 of the framework doc).
    """
    if price_per_night < 0:
        raise ValueError(f"price_per_night must be >= 0, got {price_per_night}")
    if occupied_nights_per_year < 0:
        raise ValueError(
            f"occupied_nights_per_year must be >= 0, got {occupied_nights_per_year}"
        )
    return price_per_night * occupied_nights_per_year


def tourist_tax_annual(
    occupied_nights_per_year: float,
    *,
    city: str,
    guests_per_night: float = 2.0,
    rate_per_guest_night: float | None = None,
) -> float:
    """Pass-through municipal tourist tax (Barcelona: €4.40/guest/night; else 0).

    Spent by the guest, but it raises the effective price and therefore
    affects demand — modelled here as a cost line so it shows up in the NOI
    breakdown for transparency, per Section 3.2 of the framework doc.
    """
    if occupied_nights_per_year < 0:
        raise ValueError(
            f"occupied_nights_per_year must be >= 0, got {occupied_nights_per_year}"
        )
    if guests_per_night <= 0:
        raise ValueError(f"guests_per_night must be > 0, got {guests_per_night}")
    rate = (
        rate_per_guest_night
        if rate_per_guest_night is not None
        else TOURIST_TAX_EUR_PER_GUEST_NIGHT.get(city.strip().lower(), 0.0)
    )
    return rate * guests_per_night * occupied_nights_per_year


def cleaning_cost_annual(
    occupied_nights_per_year: float,
    *,
    avg_length_of_stay: float = 3.0,
    cost_per_clean_eur: float | None = None,
    self_cleaning: bool = False,
) -> float:
    """Annual cleaning cost: ``turnovers × cost_per_clean``.

    ``turnovers = occupied_nights_per_year / avg_length_of_stay`` (Section
    4.1). ``self_cleaning=True`` returns 0 (no out-of-pocket cost), matching
    the "self-cleaning" row in the framework's cost table.
    """
    if occupied_nights_per_year < 0:
        raise ValueError(
            f"occupied_nights_per_year must be >= 0, got {occupied_nights_per_year}"
        )
    if avg_length_of_stay <= 0:
        raise ValueError(f"avg_length_of_stay must be > 0, got {avg_length_of_stay}")
    if self_cleaning:
        return 0.0
    cost_per_clean = (
        cost_per_clean_eur
        if cost_per_clean_eur is not None
        else FINANCE["cleaning_fee_per_stay_eur"]
    )
    turnovers = occupied_nights_per_year / avg_length_of_stay
    return turnovers * cost_per_clean


def compute_noi(
    gross_revenue: float,
    *,
    property_value: float,
    platform_fee_rate: float = FINANCE["platform_fee_pct"],
    cleaning_cost_eur: float = 0.0,
    tourist_tax_eur: float = 0.0,
    utilities_eur: float = 0.0,
    ibi_eur: float = 0.0,
    basuras_eur: float = 0.0,
    community_fee_eur: float = 0.0,
    insurance_eur: float = FINANCE["str_insurance_eur"],
    maintenance_rate: float = FINANCE["maintenance_pct_of_value"],
    management_fee_rate: float = 0.0,
    accounting_fee_eur: float = FINANCE["accounting_fee_eur"],
    effective_tax_rate: float = FINANCE["income_tax_pct"],
) -> dict[str, float]:
    """Net Operating Income after all running costs and tax — Section 4.5.

    ``management_fee_rate``: 0 for DIY hosting, ~0.15–0.25 for an agency
    (Section 4.2). ``effective_tax_rate``: a single parameterised rate
    standing in for IRPF progressive brackets, since those depend on the
    owner's total income from all sources (deliberately not computed here).

    Returns a dict with every deduction line plus ``noi`` (post-tax), so
    callers can render a full income-statement breakdown rather than just
    the bottom line.

    Raises
    ------
    ValueError
        If any rate is outside ``[0, 1]``, or any EUR amount is negative.
    """
    _validate_rate("platform_fee_rate", platform_fee_rate)
    _validate_rate("maintenance_rate", maintenance_rate)
    _validate_rate("management_fee_rate", management_fee_rate)
    _validate_rate("effective_tax_rate", effective_tax_rate)
    for name, value in [
        ("gross_revenue", gross_revenue),
        ("property_value", property_value),
        ("cleaning_cost_eur", cleaning_cost_eur),
        ("tourist_tax_eur", tourist_tax_eur),
        ("utilities_eur", utilities_eur),
        ("ibi_eur", ibi_eur),
        ("basuras_eur", basuras_eur),
        ("community_fee_eur", community_fee_eur),
        ("insurance_eur", insurance_eur),
        ("accounting_fee_eur", accounting_fee_eur),
    ]:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    platform_fee = platform_fee_rate * gross_revenue
    maintenance = maintenance_rate * property_value
    management_fee = management_fee_rate * gross_revenue

    deductions = {
        "platform_fee": platform_fee,
        "cleaning_cost": cleaning_cost_eur,
        "tourist_tax": tourist_tax_eur,
        "utilities": utilities_eur,
        "ibi": ibi_eur,
        "basuras": basuras_eur,
        "community_fee": community_fee_eur,
        "insurance": insurance_eur,
        "maintenance": maintenance,
        "management_fee": management_fee,
        "accounting_fee": accounting_fee_eur,
    }
    pre_tax_noi = gross_revenue - sum(deductions.values())
    # Tax applies to taxable profit, not a loss; a loss-making year owes no
    # income tax on STR activity (still pre_tax_noi as the cash result).
    tax = effective_tax_rate * max(pre_tax_noi, 0.0)
    noi = pre_tax_noi - tax

    return {
        "gross_revenue": gross_revenue,
        **deductions,
        "pre_tax_noi": pre_tax_noi,
        "tax": tax,
        "noi": noi,
    }


def _validate_rate(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")