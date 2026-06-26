import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { RecommendationPill } from "@/components/ui/RecommendationPill";
import { KpiCard } from "@/features/decision/KpiCard";
import { AnalystBrief } from "./AnalystBrief";
import { MarketPositioning } from "./MarketPositioning";
import { RateDrivers } from "./RateDrivers";
import { Comparables } from "./Comparables";
import { GuestSentiment } from "./GuestSentiment";
import { comparables, getSentiment } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { eur, pct0 } from "@/lib/format";
import {
  comparablesRequest,
  rating2,
  signedPct,
  type Benchmark,
  type Comp,
} from "@/lib/market";

export function MarketTab() {
  const navigate = useNavigate();
  const scen = useAppStore((s) => s.activeScenario);
  const prop = useAppStore((s) => s.activeProperty);

  const query = useQuery({
    queryKey: [
      "comparables",
      prop?.city,
      prop?.district,
      prop?.room_type,
      prop?.bedrooms,
      prop?.accommodates,
    ],
    queryFn: () => comparables(comparablesRequest(prop!)),
    enabled: !!prop,
    staleTime: 5 * 60_000,
    retry: false,
  });

  // Real review-text sentiment (madrid/barcelona/malaga). Graceful: on
  // 404/503/error GuestSentiment falls back to the rating-based proxy.
  const sentimentQuery = useQuery({
    queryKey: ["sentiment", prop?.city],
    queryFn: () => getSentiment(prop!.city),
    enabled: !!prop?.city,
    staleTime: Infinity,
    retry: false,
  });

  if (!scen || !prop) {
    return (
      <Card className="grid min-h-[260px] place-items-center text-center">
        <div>
          <p className="text-[15px] font-semibold text-hof">No active analysis</p>
          <p className="mt-1 text-[14px] text-foggy">
            Start one to see how this property sits in the market.
          </p>
          <Button className="mt-4" onClick={() => navigate("/new")}>
            Start new analysis
          </Button>
        </div>
      </Card>
    );
  }

  const comps = (query.data?.comparables ?? []) as unknown as Comp[];
  const benchmark = (query.data?.benchmark ?? {}) as unknown as Benchmark;
  const medianNightly = benchmark.price?.median;
  const medianRevenue = benchmark.estimated_revenue_l365d?.median;
  const medianRating = benchmark.review_scores_rating?.median;

  return (
    <div className="flex flex-col gap-6">
      {/* Top row */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <RecommendationPill recommendation={scen.recommendation} />
          <span className="text-[14px] text-hof-secondary">
            P(Airbnb beats Sell){" "}
            <strong className="text-hof">{pct0(scen.p_airbnb_gt_sell)}</strong>
          </span>
        </div>
        <Button
          variant="secondary"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw
            size={15}
            className={query.isFetching ? "animate-spin" : ""}
          />
          {query.isFetching ? "Refreshing…" : "Regenerate"}
        </Button>
      </div>

      {/* Market KPI cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard
          label="Predicted nightly"
          value={scen.predicted_nightly_eur}
          format={eur}
          delta="LightGBM price model"
        />
        <KpiCard
          label="Median nightly"
          value={medianNightly ?? 0}
          format={eur}
          delta={
            medianNightly
              ? `${signedPct(scen.predicted_nightly_eur, medianNightly)} vs market`
              : query.isLoading
                ? "Loading comps…"
                : "No comps"
          }
        />
        <KpiCard
          label="Median annual revenue"
          value={medianRevenue ?? 0}
          format={eur}
          delta={medianRevenue ? "across comparable listings" : "—"}
        />
        <KpiCard
          label="Median guest rating"
          value={medianRating ?? 0}
          format={(n) => `★ ${rating2(n)}`}
          delta={`${comps.length} comparable ${comps.length === 1 ? "listing" : "listings"}`}
        />
      </div>

      {/* Analyst brief */}
      <AnalystBrief scen={scen} benchmark={benchmark} compCount={comps.length} />

      {/* Comps failure banner (non-blocking) */}
      {query.isError && (
        <div className="rounded-card border border-[#F6E2C6] bg-[#FFF6EC] px-4 py-3 text-[13px] text-hof-secondary">
          Couldn't load comparable listings.{" "}
          <button
            className="font-semibold text-kpmg-cobalt underline"
            onClick={() => query.refetch()}
          >
            Retry
          </button>
          . The recommendation and benchmarks above still reflect your analysis.
        </div>
      )}

      {/* Positioning + drivers */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <MarketPositioning scen={scen} benchmark={benchmark} comps={comps} />
        <RateDrivers scen={scen} comps={comps} benchmark={benchmark} />
      </div>

      {/* Comparable listings */}
      {query.isLoading ? (
        <Card className="grid min-h-[200px] place-items-center text-[14px] text-foggy">
          Loading comparable listings…
        </Card>
      ) : (
        <Comparables comps={comps} benchmark={benchmark} />
      )}

      {/* Guest sentiment — real /sentiment when available, else rating proxy */}
      {(comps.length > 0 || sentimentQuery.data) && (
        <GuestSentiment comps={comps} sentiment={sentimentQuery.data ?? null} />
      )}
    </div>
  );
}
