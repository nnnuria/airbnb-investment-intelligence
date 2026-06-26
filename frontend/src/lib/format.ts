/** Display formatters. EUR with thousands separators; "k" compaction for bands. */

export function eur(n: number): string {
  return `€${Math.round(n).toLocaleString("en-US")}`;
}

export function eurCompact(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1000) return `€${Math.round(n / 1000)}k`;
  return `€${Math.round(n)}`;
}

/** Signed EUR for the NPV-advantage centrepiece (+€… teal / −€… red). */
export function signedEur(n: number): string {
  const sign = n >= 0 ? "+" : "−";
  return `${sign}€${Math.abs(Math.round(n)).toLocaleString("en-US")}`;
}

export function pct0(frac: number): string {
  return `${Math.round(frac * 100)}%`;
}

export function years(n: number): string {
  return n >= 50 ? "50+ yrs" : `${n.toFixed(1)} yrs`;
}
