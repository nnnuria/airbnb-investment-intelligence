import { Sparkles } from "lucide-react";
import { eur, pct0 } from "@/lib/format";
import { signedPct, type Benchmark } from "@/lib/market";
import type { ScenarioResponse } from "@/lib/types";

/**
 * Purple-tinted analyst brief. Synthesised deterministically from the scenario
 * + comp benchmarks (no extra LLM round-trip on tab load). The richer
 * /market_report narrative can swap in here later behind the same card.
 */
export function AnalystBrief({
  scen,
  benchmark,
  compCount,
}: {
  scen: ScenarioResponse;
  benchmark: Benchmark;
  compCount: number;
}) {
  const medianNightly = benchmark.price?.median;
  const p25 = benchmark.price?.p25;
  const p75 = benchmark.price?.p75;
  const nightly = scen.predicted_nightly_eur;

  let headline = "Strong & stable";
  let position = "in line with the local market";
  if (p75 != null && nightly > p75) {
    headline = "Premium positioning";
    position = "above the local top quartile";
  } else if (p25 != null && nightly < p25) {
    headline = "Value positioning";
    position = "below the local lower quartile";
  }

  const winsAirbnb = scen.recommendation === "airbnb";
  const verdict =
    scen.recommendation === "marginal"
      ? "The case is finely balanced"
      : winsAirbnb
        ? "Short-term letting looks like the stronger play"
        : "Selling looks like the stronger play";

  return (
    <div className="rounded-card border border-[#E4DBFB] bg-[#F4F1FF] p-5">
      <div className="flex items-center gap-2">
        <Sparkles size={16} className="text-kpmg-purple" />
        <p className="eyebrow text-kpmg-purple">Analyst brief</p>
      </div>
      <p className="mt-2 text-[19px] font-semibold tracking-[-0.01em] text-hof">
        {headline}
      </p>
      <p className="mt-2 max-w-3xl text-[14px] leading-relaxed text-hof-secondary">
        At a modelled <strong>{eur(nightly)}/night</strong>, this property sits{" "}
        {position}
        {medianNightly != null && (
          <> ({signedPct(nightly, medianNightly)} vs the median of {eur(medianNightly)})</>
        )}{" "}
        across {compCount} comparable {compCount === 1 ? "listing" : "listings"}.{" "}
        {verdict} — at {pct0(scen.confidence)} confidence, with a{" "}
        {pct0(scen.p_airbnb_gt_sell)} modelled probability that listing beats
        selling over the {scen.holding_years}-year horizon.
      </p>
    </div>
  );
}
