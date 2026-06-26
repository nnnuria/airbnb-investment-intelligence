/**
 * Market-tab data shapes + derivations.
 *
 * The comparables agent (`/comparables`) returns rows limited to
 * `DISPLAY_COLUMNS` in src/airbnb_iip/agents/comparables.py. Fields marked
 * "pending exposure" below already live in
 * Data/processed/listings_segmented.parquet but are NOT in DISPLAY_COLUMNS yet —
 * a one-line backend change (Marco's track). Everything here treats them as
 * optional and degrades gracefully when they are absent.
 */
import { ALL_AMENITY_KEYS } from "./constants";
import type { ScenarioRequest } from "./types";

export interface Comp {
  id?: number | string;
  name?: string;
  city?: string;
  neighbourhood_cleansed?: string;
  room_type?: string;
  property_type_std?: string;
  Segment_Name?: string;
  accommodates?: number;
  bedrooms?: number;
  bathrooms_number?: number;
  price?: number;
  estimated_revenue_l365d?: number;
  review_scores_rating?: number;
  distance?: number;

  // ── pending exposure in comparables.py DISPLAY_COLUMNS (graceful) ──────────
  picture_url?: string;
  latitude?: number;
  longitude?: number;
  review_scores_cleanliness?: number;
  review_scores_location?: number;
  review_scores_value?: number;
  review_scores_accuracy?: number;
}

export interface BenchmarkStat {
  median: number;
  p25: number;
  p75: number;
}

export interface Benchmark {
  price?: BenchmarkStat;
  estimated_revenue_l365d?: BenchmarkStat;
  review_scores_rating?: BenchmarkStat;
}

/** Count of truthy `has_*` amenity flags on the active property. */
export function amenityCount(prop: ScenarioRequest): number {
  return ALL_AMENITY_KEYS.reduce(
    (n, key) => n + (prop[key as `has_${string}`] ? 1 : 0),
    0,
  );
}

/** Build the `/comparables` request body from the active property. */
export function comparablesRequest(prop: ScenarioRequest, k = 12) {
  return {
    city: prop.city,
    district: prop.district,
    room_type: prop.room_type,
    accommodates: prop.accommodates,
    bedrooms: prop.bedrooms,
    bathrooms_number: prop.bathrooms,
    amenity_count: amenityCount(prop),
    k,
  };
}

/** The `n` nearest comps (ascending `distance`; rows without a distance sort
 *  last). Falls back to input order when the field is absent. */
export function nearest(comps: Comp[], n: number): Comp[] {
  return [...comps]
    .sort((a, b) => (a.distance ?? Infinity) - (b.distance ?? Infinity))
    .slice(0, n);
}

/** Signed percentage delta, e.g. +4% / −12%. Null-safe on a zero base. */
export function signedPct(subject: number, base: number | undefined): string {
  if (!base) return "—";
  const frac = (subject - base) / base;
  const sign = frac >= 0 ? "+" : "−";
  return `${sign}${Math.round(Math.abs(frac) * 100)}%`;
}

/** Rating to 2dp (Airbnb-style "4.88"). */
export function rating2(n: number | undefined): string {
  return n == null ? "—" : n.toFixed(2);
}

/**
 * Guest-sentiment proxy from comp guest ratings (the real /sentiment NLP
 * service is a separate backend track). Buckets each comp's overall rating:
 * ≥4.7 positive, 4.3–4.7 neutral, <4.3 negative. Labelled as rating-based in
 * the UI so it's never mistaken for review-text sentiment.
 */
export function ratingSentiment(comps: Comp[]): {
  positive: number;
  neutral: number;
  negative: number;
  n: number;
} {
  const rated = comps.filter((c) => typeof c.review_scores_rating === "number");
  const n = rated.length;
  if (n === 0) return { positive: 0, neutral: 0, negative: 0, n: 0 };
  let positive = 0;
  let neutral = 0;
  let negative = 0;
  for (const c of rated) {
    const r = c.review_scores_rating as number;
    if (r >= 4.7) positive += 1;
    else if (r >= 4.3) neutral += 1;
    else negative += 1;
  }
  return {
    positive: Math.round((positive / n) * 100),
    neutral: Math.round((neutral / n) * 100),
    negative: Math.round((negative / n) * 100),
    n,
  };
}

/** Map an Airbnb 0–5 rating onto a −1…+1 sentiment proxy (4.5 = neutral). */
export function ratingToScore(rating: number): number {
  return Math.max(-1, Math.min(1, (rating - 4.5) / 0.5));
}

/**
 * Normalize a share map (label_split / lang_split — values may be 0–1 fractions
 * or raw counts) to integer percentages summing to ~100, sorted desc.
 */
export function toPercentShares(
  raw: Record<string, number>,
): { key: string; pct: number }[] {
  const entries = Object.entries(raw).filter(([, v]) => v > 0);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total <= 0) return [];
  return entries
    .map(([key, v]) => ({ key, pct: Math.round((v / total) * 100) }))
    .sort((a, b) => b.pct - a.pct);
}

/** Do the comps carry the per-category review subscores (radar inputs)? */
export function hasSubscores(comps: Comp[]): boolean {
  return comps.some(
    (c) =>
      c.review_scores_cleanliness != null ||
      c.review_scores_location != null ||
      c.review_scores_value != null ||
      c.review_scores_accuracy != null,
  );
}
