import { Card } from "@/components/ui/Card";
import { PositioningScatter } from "./charts/PositioningScatter";
import { rating2, type Benchmark, type Comp } from "@/lib/market";
import type { ScenarioResponse } from "@/lib/types";

/** Diverging bar from centre: positive = teal to the right, negative = red to the left. */
function DivergingBar({ label, frac }: { label: string; frac: number | null }) {
  const cap = 0.5; // clamp at ±50% so bars stay in frame
  const mag = frac == null ? 0 : (Math.min(Math.abs(frac), cap) / cap) * 50;
  const positive = (frac ?? 0) >= 0;
  const pctLabel =
    frac == null ? "—" : `${positive ? "+" : "−"}${Math.round(Math.abs(frac) * 100)}%`;

  return (
    <div>
      <div className="flex justify-between text-[13px]">
        <span className="text-hof-secondary">{label}</span>
        <span className="font-semibold" style={{ color: positive ? "#00A699" : "#FF385C" }}>
          {pctLabel}
        </span>
      </div>
      <div className="relative mt-1.5 h-2 rounded-pill bg-track">
        <div className="absolute left-1/2 top-0 h-full w-px bg-[#D5D5D5]" />
        <div
          className="absolute top-0 h-full rounded-pill transition-all"
          style={{
            background: positive ? "#00A699" : "#FF385C",
            left: positive ? "50%" : `${50 - mag}%`,
            width: `${mag}%`,
          }}
        />
      </div>
    </div>
  );
}

export function MarketPositioning({
  scen,
  benchmark,
  comps,
}: {
  scen: ScenarioResponse;
  benchmark: Benchmark;
  comps: Comp[];
}) {
  const medianNightly = benchmark.price?.median;
  const medianRevenue = benchmark.estimated_revenue_l365d?.median;
  const medianRating = benchmark.review_scores_rating?.median;

  const nightlyDelta =
    medianNightly ? (scen.predicted_nightly_eur - medianNightly) / medianNightly : null;
  const revenueDelta =
    medianRevenue ? (scen.gross_revenue_year_eur - medianRevenue) / medianRevenue : null;

  return (
    <Card>
      <p className="text-[15px] font-semibold text-hof">Market positioning</p>
      <p className="mb-4 text-[12.5px] text-foggy">
        Your modelled property vs comparable listings
      </p>

      <div className="flex flex-col gap-3.5">
        <DivergingBar label="Nightly rate vs market" frac={nightlyDelta} />
        <DivergingBar label="Annual revenue vs market" frac={revenueDelta} />
        <div className="flex items-center justify-between rounded-card bg-track/60 px-3.5 py-2.5">
          <span className="text-[13px] text-hof-secondary">Peer guest rating</span>
          <span className="text-[15px] font-semibold text-kpmg-cobalt">
            ★ {rating2(medianRating)}
          </span>
        </div>
      </div>

      <div className="mt-5 border-t border-hairline pt-4">
        <p className="mb-1 text-[12.5px] font-semibold text-hof">
          Price vs revenue
        </p>
        <PositioningScatter
          comps={comps}
          subject={{
            nightly: scen.predicted_nightly_eur,
            revenue: scen.gross_revenue_year_eur,
          }}
        />
      </div>
    </Card>
  );
}
