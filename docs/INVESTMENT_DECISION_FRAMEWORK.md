# Investment Decision Framework
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**  
*Drafted 2026-06-14 — covers analytical design, data findings, and model plan*

---

## 1. The Core Decision

The platform answers one question: **should the owner keep the property as an Airbnb short-term rental, or sell it?**

Both options are valued using Net Present Value over a common holding horizon T, so they are directly comparable on the same monetary basis.

---

## 2. NPV Comparison Framework

### Option A — Sell

```
NPV_sell = P_sale × (1 − tx_sell)
```

Where `tx_sell` covers:
- Real estate agent commission: 3–6% of P\_sale
- Notary + property registry: ~1% of P\_sale
- Capital gains tax (see Section 4.3)

This is an immediate lump sum. The owner walks away with cash.

### Option B — Keep as Airbnb

```
NPV_airbnb(T) = Σ_{t=1}^{T}  [ NOI_t / (1+r)^t ]
              + P_terminal_T / (1+r)^T
              − C_setup
```

Where:
- `NOI_t` = Net Operating Income in year t (revenue minus all costs, after tax)
- `P_terminal_T` = estimated sale value at end of horizon = P\_today × (1 + g)^T, net of CGT and agent fees
- `r` = discount rate (opportunity cost of capital)
- `C_setup` = one-time setup costs (license, furnishing/refurbishment if needed)

### The Decision Rule

```
Airbnb if NPV_airbnb(T) > NPV_sell
Sell    otherwise
```

Supplementary output: **break-even horizon T\*** = the minimum T at which NPV\_airbnb overtakes NPV\_sell. If T\* = 7 years and the owner plans to hold for 10 years → Airbnb is recommended. If T\* = 25 years → sell.

### User Input Parameters

| Parameter | Typical range | Notes |
|---|---|---|
| Holding horizon T | 5 / 10 / 20 years | Sensitivity chart across multiple T |
| Discount rate r | 6–9% | Opportunity cost of capital; default 7% |
| Capital appreciation g | 2–6% / yr | City-specific; user sets scenario range |
| Purchase year & price | — | Needed for CGT calculation |
| Management model | DIY / agency | Drives cost structure (see Section 4.2) |
| Effective income tax rate | 19–47% | IRPF progressive; user knows their bracket |
| CGT exemptions | Yes / No | Primary residence reinvestment; over-65 |

---

## 3. Revenue Model

### 3.1 Gross Revenue

```
Annual_gross = Σ_{m=1}^{12}  price_hat × occ_hat[m]
```

Where `price_hat` and `occ_hat` come from the ML models (Section 6), and `occ_hat[m]` is the monthly distribution produced by the seasonality layer (Section 5).

### 3.2 Platform & Booking Deductions

| Item | Rate | Notes |
|---|---|---|
| Airbnb host service fee | 3% of booking subtotal | Excludes cleaning fee; tax-deductible |
| Tourist tax | Barcelona: €4.40/night/person; Madrid: €0; Málaga: €0 | Pass-through; affects guest price sensitivity |

---

## 4. Cost Model — Spain Specifics

### 4.1 Cleaning

```
cleaning_cost_annual = cost_per_clean × (occ_days / avg_stay_nights)
```

A flat with 60% occupancy (~219 days/year) and an average 3-night stay has ~73 turnovers/year.

| Scenario | Per-turnover cost | Annual (73 turnovers) |
|---|---|---|
| Self-cleaning | €0 out-of-pocket | — |
| Professional cleaning (1-2 bed) | €50–120 | €3,650–8,760 |
| Guest cleaning fee offsets cost | Depends on fee set | Net ~€0 if fee = actual cost |

### 4.2 Operating Costs (annual, 1-2 bed flat)

| Cost item | Typical range | Notes |
|---|---|---|
| Electricity | €1,200–2,400 | Higher with AC-heavy summers |
| Water | €360–600 | |
| Internet (fibre) | €480 | |
| Gas (heating/hot water) | €400–800 | |
| **Total utilities** | **€2,500–4,300** | Host-paid; standard for Airbnb listings |
| IBI (property tax) | €200–800 | 0.4–1.1% of *cadastral value* (cadastral << market value) |
| Basuras (waste tax) | €80–300 | Municipal; varies by city |
| Community fee (comunidad) | €600–3,600 | Highly variable; central Madrid/Barcelona buildings expensive |
| Building insurance | €150–400 | Often included in community fee |
| STR liability insurance | €400–1,200 | Standard home insurance does NOT cover STR; critical gap |
| Maintenance & repairs | 1–1.5% of property value | Boilers, appliances, painting; budget higher for older stock |
| Property management (if agency) | 15–25% of gross revenue | Covers pricing, guest comms, check-in, light maintenance |
| Accounting / tax advisory | €500–1,500 | Annual IRPF + quarterly tax models |

### 4.3 Taxes on STR Income — Spain

STR income is classified as **rendimiento del capital inmobiliario** (not exempt; not treated as business income unless hotel-like services are provided).

```
Taxable_base = Gross_revenue − Deductible_expenses
```

**Deductible expenses:**
- Platform fees (Airbnb 3%)
- Cleaning, management, utilities, insurance
- IBI, community fee, maintenance
- Accountant fees
- Mortgage interest (if leveraged)
- Depreciation (amortización): 3% × max(construction cost, cadastral construction value) — capped at gross income

**Tax rates:**
- EU residents: IRPF progressive brackets 19%→47%
- Non-EU residents: 24% flat rate on gross income (fewer deductions)

**Effective rate after deductions** is typically 15–25% for a mid-income owner. Parameterise this as a single input rather than computing IRPF brackets (which depend on the owner's total income from all sources).

### 4.4 Capital Gains Tax on Sale — Spain (2024 rates)

| Gain | Rate |
|---|---|
| First €6,000 | 19% |
| €6,000 – €50,000 | 21% |
| €50,000 – €200,000 | 23% |
| Above €200,000 | 26% |

Gain = P\_sale − (original\_purchase\_price + purchase\_costs + documented\_improvements).

**Exemptions:**
- Reinvestment in primary residence → full exemption
- Over-65 selling primary residence → full exemption
- These are boolean flags in the user input form

### 4.5 NOI Formula

```
NOI = (
    gross_revenue
  − platform_fee           # 3% × gross_revenue
  − cleaning_cost          # turnovers × cost_per_clean
  − utilities              # fixed annual
  − ibi                    # fixed annual
  − basuras                # fixed annual
  − community_fee          # fixed annual
  − insurance              # fixed annual
  − maintenance            # 1.25% × property_value
  − management_fee         # 0 (DIY) or ~20% × gross_revenue (agency)
  − accounting_fee         # fixed annual
) × (1 − effective_tax_rate)
```

---

## 5. Seasonality Layer

### 5.1 What the Calendar Data Contains

All three cities have calendar files (Madrid: 9.1M rows, Barcelona: 7.1M, Málaga: 3.5M). Each row is one listing × one date for the 365 days following the scrape date (Sept 14, 2025 → Sept 14, 2026).

**Critical finding: the `price` column is 100% null in all three cities.** InsideAirbnb stopped populating calendar prices. Only `available` ('t'/'f'), `minimum_nights`, and `maximum_nights` are present.

This means:
- There is no price time series → **no time series price model is possible or needed**
- The calendar can only inform **demand seasonality** (how full listings are by month)

### 5.2 The Scrape-Date Artifact

Because the calendar is forward-looking from a single scrape date, bookings within the next 2–4 weeks are already committed and show as 'unavailable' regardless of genuine seasonal demand. This inflates apparent booking rates for dates close to the scrape.

Measured effect for Madrid:

| Lookahead window | Apparent booking rate |
|---|---|
| 0–2 days | 85.8% — committed reservations |
| 3–7 days | 83.4% — committed |
| 1–2 weeks | 78.8% — mostly committed |
| 6–12 months | 58.8% — genuine seasonal signal |

**Fix:** filter to dates ≥ 60 days from scrape before computing multipliers.

### 5.3 Computing Monthly Multipliers

```python
cal['days_ahead'] = (cal['date'] - pd.Timestamp('2025-09-14')).dt.days
cal_clean = cal[cal['days_ahead'] >= 60].copy()
cal_clean['is_booked'] = (cal_clean['available'] == 'f')

# Monthly booking rate (booked / total days in that month)
monthly_rate = cal_clean.groupby(cal_clean['date'].dt.month)['is_booked'].mean()

# Normalise to mean = 1 → multiplicative multipliers
monthly_multipliers = monthly_rate / monthly_rate.mean()
```

Run this separately per city. Store results as a JSON lookup: `{city: {month: multiplier}}`.

### 5.4 Actual Seasonal Shapes (corrected)

| Month | Madrid | Barcelona | Málaga |
|---|---|---|---|
| Jan | ~0.90 | ~0.76 | ~0.70 |
| Feb | ~0.88 | ~0.75 | ~0.72 |
| Mar | ~0.92 | ~0.77 | ~0.70 |
| Apr | ~0.96 | ~0.78 | ~0.78 |
| May | ~0.94 | ~0.81 | ~0.77 |
| Jun | ~1.04 | ~0.91 | ~0.78 |
| Jul | ~1.07 | ~0.95 | ~0.93 |
| Aug | ~1.10 | ~0.98 | ~0.94 |
| Sep | Scrape artifact | Scrape artifact | Scrape artifact |
| Oct | ~1.27 | ~1.19 | ~1.19 |
| Nov | ~0.97 | ~0.85 | ~0.73 |
| Dec | ~0.92 | ~0.73 | ~0.69 |

*Note: Sep figures in the raw data are inflated by committed bookings near the scrape date; bias-corrected multipliers will be used. Oct is the genuine post-summer peak for all three cities.*

**Key insight:** Madrid is relatively flat year-round (city tourism, conferences), while Málaga has the most pronounced seasonal swing — a property there in November earns roughly half what it earns in October.

### 5.5 Applying Multipliers in the NPV Calculation

```
occ_hat[month] = annual_occ_hat × multiplier[month] / Σ multipliers
revenue_hat[month] = occ_hat[month] × price_hat
NOI[month] = revenue_hat[month] − monthly_fixed_costs − variable_costs(revenue_hat[month])
NPV_airbnb = Σ_{t,m} NOI[t,m] / (1+r)^(t + m/12)  +  P_terminal / (1+r)^T
```

---

## 6. ML Model Plan

### 6.1 Architecture Overview

```
Input: target property characteristics + user cost parameters
          │
          ▼
    ┌─────────────┐
    │ Price model │  → price_hat (€/night)
    └─────────────┘
          │ price_hat fed as feature
          ▼
    ┌──────────────────┐
    │ Occupancy model  │  → occ_hat (annual days)
    └──────────────────┘
          │ occ_hat distributed across months
          ▼
    ┌──────────────────────┐
    │ Seasonal multipliers │  → occ_hat[month] × 12
    │ (from calendar data) │
    └──────────────────────┘
          │
          ▼
    ┌─────────────┐
    │  NPV engine │  → NPV_airbnb, NPV_sell, T*, Monte Carlo bands
    └─────────────┘
```

### 6.2 Price Model — ✅ Complete

**Target:** `log(price)` — log-transform handles the right-skewed price distribution  
**Production model: LightGBM** — selected by Test R² from a 5-model comparison  
**Artefacts:** `models/price_best_model.pkl`, `price_feature_cols.json`, `price_cat_encoders.pkl`, `price_scaler.pkl`

#### Model comparison (held-out test set)

| Model | Test R² | Test MAE (€) | MdAPE | Train–Test gap |
|-------|---------|-------------|-------|----------------|
| **LightGBM** ⭐ | **0.803** | **€34.6** | **16.0%** | 0.116 |
| XGBoost | 0.799 | €35.1 | 16.3% | 0.095 |
| Gradient Boosting | 0.768 | €38.1 | 17.9% | 0.036 |
| Random Forest | 0.766 | €37.9 | 17.4% | 0.131 |
| Linear / Ridge | 0.602 | €49.7 | 25.6% | ~0.000 |

#### Per-city performance (LightGBM)

| City | n (test) | R² | MAE (€/night) | MdAPE |
|------|----------|----|---------------|-------|
| Madrid | 3,772 | 0.797 | €32.9 | 16.5% |
| Barcelona | 3,031 | 0.819 | €39.5 | 15.7% |
| Málaga | 1,719 | 0.752 | €29.4 | 15.3% |

#### Top SHAP drivers (mean |SHAP| on log-price scale)

| Rank | Feature | Mean \|SHAP\| | Note |
|------|---------|-------------|------|
| 1 | `property_type_std` | 0.216 | Entire home vs room is the largest single lever |
| 2 | `accommodates` | 0.133 | Each additional guest capacity raises price materially |
| 3 | `minimum_nights` | 0.126 | Long-stay listings form a distinct pricing segment |
| 4 | `neighbourhood_target_enc` | 0.076 | Neighbourhood premium (target-encoded) |
| 5 | `latitude` | 0.064 | Micro-location within neighbourhood |
| 6 | `bathrooms_number` | 0.053 | Extra bathrooms add premium for larger groups |
| 7 | `reviews_per_month` | 0.047 | Listing velocity / market position signal |
| 8 | `longitude` | 0.044 | Coastal / central premium axis |
| 9 | `bedrooms` | 0.040 | Bedroom count for multi-room properties |
| 10 | `calculated_host_listings_count_entire_homes` | 0.034 | Professional host pricing behaviour |

#### Features used / excluded

The ~30 selected features include: `property_type_std`, `accommodates`, `minimum_nights`, `neighbourhood_target_enc`, `latitude`, `longitude`, `bathrooms_number`, `bedrooms`, `room_type`, `city`, `host_is_superhost`, `host_tenure_years`, `has_reviews`, amenity dummies (pool, AC, elevator, parking, washer, workspace, etc.), `competitive_density_500m`.

**Explicitly excluded (leakage / endogeneity):** `availability_30/60/90/365`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `days_since_last_review`, `review_scores_communication/_checkin/_accuracy` (collinear with `_rating`), `source`, `scrape_id`.

### 6.3 Occupancy Model

**Target:** `estimated_occupancy_l365d` (annual days booked, from InsideAirbnb SF model)  
**Algorithm:** LightGBM  
**Two-stage design:** trained after price model; uses `price_hat` (not actual price) to capture demand elasticity without target leakage

**Additional features over price model:**

| Feature | Rationale |
|---|---|
| `price_hat` (from Stage 1) | Demand elasticity — higher price → lower occupancy |
| `availability_30`, `availability_60` | Now safe to include — host intent / short-term demand signal |
| `reviews_per_month` | Now safe — reflects booking velocity |
| `number_of_reviews_ltm` | Recent demand signal |
| `host_response_rate`, `host_acceptance_rate` | Booking conversion drivers |
| District listing density (engineered) | Supply pressure: listings per km² within 500m |

**District competitive density** (feature engineering step):
```python
from sklearn.neighbors import BallTree
# For each listing, count neighbours within 0.5 km radius
# Use Haversine distance on lat/lon
```

### 6.4 One Model vs. Per-City

**Decision: one combined model with `city` as a categorical feature.**

Rationale:
- Málaga (~5k listings) is too small to train a robust gradient boosting model independently
- Feature effects (bedrooms → price, superhost → occupancy) are directionally consistent across cities; only magnitudes differ — exactly what `city` as a feature + tree splits handle
- `neighbourhood_group_cleansed` is already city-specific, providing fine-grained location adjustment within the single model
- SHAP values can be filtered post-hoc by city to get city-specific attribution

**Seasonal multipliers are city-specific** (computed separately from calendar data, not learned by the model).

### 6.5 Model Outputs and SHAP Attribution

For the optimisation flow, SHAP values are as important as predictions. The price model should surface, for any given property:

> *"Predicted price: €148/night. Key drivers:*
> *+€32 — Salamanca district premium*
> *+€18 — superhost status*
> *−€14 — below-average review score*
> *−€8 — no dedicated workspace*
> *+€5 — instant bookable"*

This bridges the decision flow and the optimisation flow: the model tells the owner not just what their property is worth but **which levers to pull** to improve it.

### 6.6 Pending Feature Engineering Steps

Before model training:

1. **Parse amenities JSON** → binary columns for top-30 amenities by city frequency  
2. **Compute district competitive density** → BallTree on lat/lon, count listings within 500m  
3. **Compute monthly seasonal multipliers** → from calendar data (60d+ bias correction) per city  
4. **Create `has_reviews` binary flag** → handles the 16% of listings with no reviews (impute review scores to city median; flag the imputation)  
5. **Clip `minimum_nights`** → long-stay listings (>30 nights) target a different market; consider filtering or separate modelling

---

## 7. NPV Engine Design

Pure functions in `finance/scenarios.py`. No ML — these are deterministic financial calculations taking model outputs as inputs.

```python
def compute_noi(gross_revenue, platform_fee_rate, cleaning_cost,
                utilities, ibi, basuras, community_fee,
                insurance, maintenance_rate, property_value,
                management_fee_rate, accounting_fee,
                effective_tax_rate) -> float: ...

def npv_airbnb(noi_series, terminal_value, discount_rate, setup_cost) -> float: ...

def npv_sell(sale_price, purchase_price, purchase_year,
             agent_rate, cgt_brackets, cgt_exemption) -> float: ...

def break_even_horizon(noi_annual, terminal_value_fn,
                       npv_sell_value, discount_rate,
                       max_years=30) -> int: ...

def monte_carlo(price_hat, price_sigma,
                occ_hat, occ_sigma,
                appreciation_low, appreciation_high,
                regulatory_shock_prob,
                cost_params, n_simulations=10_000) -> pd.DataFrame: ...
```

### Monte Carlo Inputs and Distributions

| Input | Distribution | Typical parameters |
|---|---|---|
| Price per night | Normal(price\_hat, σ from model prediction interval) | σ ≈ 20–25% of price\_hat |
| Annual occupancy | Normal(occ\_hat, σ from model) | σ ≈ 20% of occ\_hat |
| Capital appreciation | Uniform(g\_low, g\_high) | User sets; e.g. Uniform(0.02, 0.06) |
| Regulatory shock | Bernoulli(p) → occupancy = 0 from year T\_shock | p = 0.05–0.20 depending on city |
| Vacancy months | Poisson(λ=0.5/yr) zero-revenue months | Renovation, disputes |

**Output:** probability distribution of NPV\_airbnb → P(Airbnb > Sell), P10/P50/P90 NPV bands.

---

## 8. Regulatory Risk by City

Incorporated as an input to Monte Carlo, not a hard block:

| City | License | Current enforcement risk | Notes |
|---|---|---|---|
| Madrid | VT (declaración responsable) | Low–medium | No new quota limits; enforcement increasing |
| Barcelona | HUT (moratorium) | High | No new licenses since 2028; existing licenses transferable at premium |
| Málaga | Andalucía STR registration | Low | Growing market; lighter regulatory touch than Catalonia |

The regulatory shock probability (Bernoulli input) should default to: Madrid 8%, Barcelona 20%, Málaga 5%.

---

## 9. Implementation Sequence

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Amenity parsing + competitive density features | ✅ Done (in `price_ml_model.ipynb`) |
| 2 | Calendar seasonality multipliers (per city) | ✅ Done (in `calendar_seasonality.ipynb`) |
| 3 | Price model (LightGBM + SHAP) | ✅ Done — R² 0.803, MAE €34.6, artefacts in `models/` |
| 4 | Occupancy model (LightGBM + SHAP) | ⬜ Next — uses `listings_with_price_hat.parquet` |
| 5 | `finance/scenarios.py` — pure NPV functions | ⬜ |
| 6 | Idealista price join (€/m² by district) | ⬜ Scraper done; join not yet built |
| 7 | End-to-end pipeline: property in → decision out | ⬜ Blocked on phases 4–6 |
| 8 | Monte Carlo wrapper | ⬜ |
| 9 | Streamlit decision tab | ⬜ |

---

## 10. Key Design Decisions (rationale on record)

| Decision | Chosen approach | Rejected alternative | Why |
|---|---|---|---|
| Revenue modelling | Two separate models: price then occupancy | Single revenue model | Separation enables optimisation SHAP ("raise price by €X") and demand elasticity |
| Seasonality | Calendar-derived multipliers applied post-prediction | Monthly dummy features in ML model | Calendar has 12 months of signal; listing data is cross-sectional (no time dimension) |
| Time series model | Not built | ARIMA / Prophet on price | No historical price time series exists — calendar prices are 100% null |
| City scope | One combined model, `city` as feature | Three separate models | Málaga too small; feature effects are transferable; neighbourhood handles local variation |
| Occupancy target | `estimated_occupancy_l365d` from InsideAirbnb | Derive from calendar `available` | SF model estimate is already noise-reduced; calendar derivation would require booking-window correction |
| NPV comparison | Both sides discounted to today | Simple payback period | Payback ignores time value of money and terminal value; NPV is the correct frame |
| Scrape artifact in calendar | Filter to dates ≥ 60 days ahead | Use raw availability | Near-term dates are committed reservations, not seasonal signal |
