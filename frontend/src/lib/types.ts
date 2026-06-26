/**
 * TypeScript mirror of the FastAPI contract (api/schemas.py on main).
 * Keep these in sync with the backend; field names match 1:1 so responses map
 * straight onto the UI. Only the fields the frontend consumes are typed here;
 * `extra="allow"` request bodies may carry more (e.g. the full amenity set).
 */

export type Recommendation = "airbnb" | "sell" | "marginal";
export type RiskFlag = "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";

// ── /scenario ────────────────────────────────────────────────────────────────

export interface ScenarioRequest {
  city: string;
  district: string;
  size_m2: number;
  bedrooms: number;
  bathrooms: number;
  accommodates: number;
  room_type?: string;
  managed?: boolean;

  // amenity flags (extra="allow" on the backend — pass any has_* booleans)
  [amenity: `has_${string}`]: boolean | undefined;

  nickname?: string | null;
  notes?: string | null;
  extra_amenities?: string[];

  // financial assumption overrides
  ibi_eur?: number | null;
  cadastral_value?: number | null;
  basuras_eur?: number | null;
  setup_cost_eur?: number;
  noi_growth_rate?: number;
  property_appreciation_rate?: number;
  include_income_tax?: boolean;
  purchase_price?: number | null;
  holding_years?: number; // 5–30, default 10  (PR #38)
  discount_rate?: number; // 0.03–0.20, default 0.07  (PR #38)
}

export type CostLine = [string, number]; // (label, eur)
export type FeatureDriver = [string, number]; // (label, fraction)

export interface ScenarioResponse {
  recommendation: Recommendation;
  confidence: number;

  predicted_nightly_eur: number;
  occupancy_rate_annual: number;
  nights_booked_year: number;

  gross_revenue_year_eur: number;
  net_revenue_year_eur: number;
  net_revenue_p10_eur: number;
  net_revenue_p90_eur: number;

  sale_price_eur: number;
  sale_price_per_m2_eur: number;
  breakeven_years: number;

  npv_airbnb_p50_eur: number;
  npv_sell_eur: number;
  p_airbnb_gt_sell: number;

  irr_airbnb_pct: number | null;
  npv_airbnb_pretax_p50_eur: number;
  npv_sell_pretax_eur: number;
  npv_advantage_eur: number; // PR #38 — Airbnb − Sell, one number
  holding_years: number; // PR #38
  discount_rate: number; // PR #38

  ibi_eur_used: number;
  ibi_method: string;
  ibi_explanation: string;
  basuras_eur_used: number;
  setup_cost_eur_used: number;

  monthly_seasonality: number[];
  cost_breakdown: CostLine[];
  feature_drivers: FeatureDriver[];
  governance: Record<string, unknown>;
}

// ── /chat ────────────────────────────────────────────────────────────────────

export interface ChatRequest {
  message: string;
  property?: Record<string, unknown> | null;
  scenario?: Record<string, unknown> | null;
  use_llm?: boolean;
}

export interface ChatResponse {
  answer: string;
  intent: string;
  sources: unknown[];
  meta: Record<string, unknown>;
  disclaimer: string;
}

// ── /market_report ───────────────────────────────────────────────────────────

export interface MarketReportResponse {
  city: string;
  neighbourhood: string;
  recommendation: string;
  confidence: Record<string, unknown>;
  subject: Record<string, unknown>;
  positioning: Record<string, unknown>;
  drivers: unknown[];
  comparables: Record<string, unknown>[];
  narrative: string;
  disclaimer: string;
}

// ── /optimise ─────────────────────────────────────────────────────────────────

export interface OptimiseAction {
  action: string;
  annual_uplift_eur: number;
  investment_eur: number;
  payback_months: number;
  confidence: "high" | "medium" | "low";
  method: string;
  lift: number | null;
  rationale: string;
}

export interface OptimiseResponse {
  actions: OptimiseAction[];
  city: string;
  district: string;
  room_type: string;
  peer_n: number;
  peer_median_revenue_eur: number;
  peer_target_revenue_eur: number;
  gap_to_top_quartile_eur: number;
  projected_annual_nights: number;
  disclaimer: string;
}

// ── /comparables ──────────────────────────────────────────────────────────────

export interface ComparablesResponse {
  comparables: Record<string, unknown>[];
  filters_applied: Record<string, unknown>;
  benchmark: Record<string, unknown>;
}

// ── /regulatory_risk ──────────────────────────────────────────────────────────

export interface RegulatoryResponse {
  risk_flag: RiskFlag;
  reason: string;
  sources: string[];
  city: string | null;
  governance: Record<string, unknown>;
  disclaimer: string;
}

// ── /saved ────────────────────────────────────────────────────────────────────

export interface SavedCreateRequest {
  property: Record<string, unknown>;
  scenario: Record<string, unknown>;
  nickname?: string | null;
}

export interface SavedRecord {
  id: string;
  saved_at: string;
  nickname?: string | null;
  property: Record<string, unknown>;
  scenario: Record<string, unknown>;
}

// ── /sentiment ────────────────────────────────────────────────────────────────

export interface SentimentExample {
  text: string;
  score: number;
}

export interface SentimentBucket {
  label: string; // left edge of the mean_sentiment bin, e.g. "-0.2"
  count: number;
}

export interface SentimentResponse {
  city: string;
  reviews_scored: number;
  label_split: Record<string, number>; // positive/neutral/negative shares (donut)
  lang_split: Record<string, number>; // top review-language shares
  metrics: Record<string, number>; // nb_accuracy / nb_f1_macro / corr_sentiment_vs_rating
  examples: Record<string, SentimentExample[]>; // keyed "positive" / "negative"
  listing_histogram: SentimentBucket[];
  n_listings: number;
  disclaimer: string;
}

// ── /health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  price_model_loaded: boolean;
}
