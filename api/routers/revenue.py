"""Finance endpoints: /estimate_revenue and /airbnb_vs_sell.

Both are wired live to the finance engine (``airbnb_iip.finance``) and the
price/sell ML models. ``/estimate_revenue`` is pure finance (caller supplies
price + occupancy); ``/airbnb_vs_sell`` is a composite that pulls the nightly
price and sale value from the models, then runs the NPV comparison.

Inputs the request schema doesn't carry are taken from documented defaults,
all overridable via the request body (``extra="allow"``):

* occupancy for /airbnb_vs_sell — explicit ``occupancy_rate`` if supplied, else
  the calendar-based LightGBM occupancy model on the property spec.
* ``purchase_price`` for CGT — defaults to ``sale_value`` (no gain ⇒ no CGT).
* terminal value — the sell ``net_proceeds`` with zero appreciation
  (conservative; matches the Streamlit decision engine).
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from airbnb_iip.config import OCCUPANCY
from airbnb_iip.models.occupancy import get_occupancy_predictor
from airbnb_iip.finance.costs import (
    annual_gross_revenue,
    cleaning_cost_annual,
    compute_noi,
    tourist_tax_annual,
)
from airbnb_iip.finance.scenarios import (
    break_even_horizon,
    monte_carlo,
    npv_airbnb,
    npv_sell,
    recommend,
)
from airbnb_iip.models.price import PricePredictor
from airbnb_iip.models.sale import SalePredictor

from api.dependencies import price_predictor, sale_predictor
from api.schemas import (
    AirbnbVsSellRequest,
    AirbnbVsSellResponse,
    EstimateRevenueRequest,
    EstimateRevenueResponse,
)

router = APIRouter(tags=["finance"])

DAYS_PER_YEAR = 365
LOS = OCCUPANCY["avg_length_of_stay"]


@router.post("/estimate_revenue", response_model=EstimateRevenueResponse)
def estimate_revenue(req: EstimateRevenueRequest) -> EstimateRevenueResponse:
    """Annual Airbnb gross/net revenue with P10/P50/P90 bands (live)."""
    extra = req.model_extra or {}
    price = req.price_per_night or 130.0
    nights = (req.occupancy_rate or 0.55) * DAYS_PER_YEAR
    city = req.city or "madrid"
    property_value = float(extra.get("property_value") or 0.0)  # maintenance base; 0 ⇒ skipped

    def net_for(p: float, n: float) -> float:
        return compute_noi(
            annual_gross_revenue(p, n),
            property_value=property_value,
            cleaning_cost_eur=cleaning_cost_annual(n, avg_length_of_stay=LOS),
            tourist_tax_eur=tourist_tax_annual(n, city=city),
        )["noi"]

    gross = annual_gross_revenue(price, nights)
    net = net_for(price, nights)

    # Bands: resample price/occupancy with the finance module's MC sigmas.
    rng = np.random.default_rng(42)
    p = rng.normal(price, price * 0.225, 5000).clip(min=0)
    n = rng.normal(nights, nights * 0.20, 5000).clip(min=0, max=365)
    nets = np.fromiter((net_for(p[i], n[i]) for i in range(p.size)), float)
    p10, p50, p90 = (round(float(x), 2) for x in np.percentile(nets, [10, 50, 90]))

    resp = EstimateRevenueResponse(
        annual_gross_eur=round(gross, 2),
        annual_net_eur=round(net, 2),
        p10_eur=p10,
        p50_eur=p50,
        p90_eur=p90,
    )
    resp.stub, resp.note = False, "Live — airbnb_iip.finance.costs"
    return resp


@router.post("/airbnb_vs_sell", response_model=AirbnbVsSellResponse)
def airbnb_vs_sell(
    req: AirbnbVsSellRequest,
    price_pred: PricePredictor = Depends(price_predictor),
    sale_pred: SalePredictor = Depends(sale_predictor),
) -> AirbnbVsSellResponse:
    """Keep-as-Airbnb vs sell on an NPV basis (live)."""
    extra = req.model_extra or {}
    city = req.city or "madrid"

    # Inputs from the ML models.
    nightly = price_pred.predict(
        {
            "city": city,
            "neighbourhood_cleansed": req.neighbourhood_cleansed,
            **{k: extra[k] for k in ("accommodates", "bedrooms", "bathrooms_number") if k in extra},
        }
    )
    sale_value = sale_pred.predict(
        {"city": city, "district": req.neighbourhood_cleansed, "size_m2": req.sq_m}
    )

    # Occupancy: explicit rate if supplied, else the calendar-based LightGBM model.
    if extra.get("occupancy_rate") is not None:
        occ_rate = float(extra["occupancy_rate"])
    else:
        occ_rate = get_occupancy_predictor().predict({
            "city": city,
            "neighbourhood_cleansed": req.neighbourhood_cleansed,
            "room_type": extra.get("room_type", "Entire home/apt"),
            "price": nightly,
            **{k: extra[k] for k in ("accommodates", "bedrooms", "bathrooms_number") if k in extra},
        })
    nights = occ_rate * DAYS_PER_YEAR

    cost_kwargs = dict(
        property_value=sale_value,
        cleaning_cost_eur=cleaning_cost_annual(nights, avg_length_of_stay=LOS),
        tourist_tax_eur=tourist_tax_annual(nights, city=city),
    )
    noi = compute_noi(annual_gross_revenue(nightly, nights), **cost_kwargs)["noi"]

    # Sell side: purchase_price defaults to sale_value ⇒ no gain ⇒ no CGT.
    purchase_price = float(extra.get("purchase_price") or sale_value)
    npv_sell_value = npv_sell(sale_value, purchase_price=purchase_price)["net_proceeds"]

    # Airbnb side + break-even + uncertainty (flat terminal value = sell proceeds).
    mc = monte_carlo(
        nightly,
        nights,
        years=req.holding_years,
        npv_sell_value=npv_sell_value,
        cost_kwargs=cost_kwargs,
        discount_rate=req.discount_rate,
        terminal_value_net=npv_sell_value,
        random_state=42,
    )
    break_even = break_even_horizon(
        noi,
        npv_sell_value=npv_sell_value,
        discount_rate=req.discount_rate,
        terminal_value_fn=lambda _t: npv_sell_value,
    )

    resp = AirbnbVsSellResponse(
        recommendation=recommend(mc["p50"], npv_sell_value),
        npv_airbnb_p50_eur=round(mc["p50"], 2),
        npv_sell_eur=round(npv_sell_value, 2),
        break_even_years=break_even,
        confidence={
            "p_airbnb_gt_sell": round(mc["p_airbnb_gt_sell"], 3),
            "p10_eur": round(mc["p10"], 2),
            "p90_eur": round(mc["p90"], 2),
        },
    )
    resp.stub, resp.note = False, "Live — airbnb_iip.finance.scenarios + models"
    return resp
