import { Card } from "@/components/ui/Card";
import { FeatureDrivers } from "@/features/decision/charts/FeatureDrivers";
import { RatingRadar } from "./charts/RatingRadar";
import { hasSubscores, rating2, type Benchmark, type Comp } from "@/lib/market";
import type { ScenarioResponse } from "@/lib/types";

export function RateDrivers({
  scen,
  comps,
  benchmark,
}: {
  scen: ScenarioResponse;
  comps: Comp[];
  benchmark: Benchmark;
}) {
  const subscores = hasSubscores(comps);

  return (
    <Card>
      <p className="text-[15px] font-semibold text-hof">
        What drives the nightly rate
      </p>
      <p className="mb-3 text-[12.5px] text-foggy">
        Top model contributions to this property's price
      </p>

      <FeatureDrivers drivers={scen.feature_drivers} />

      <div className="mt-5 border-t border-hairline pt-4">
        <p className="mb-1 text-[12.5px] font-semibold text-hof">
          Peer guest-rating profile
        </p>
        {subscores ? (
          <RatingRadar comps={comps} />
        ) : (
          <div className="grid min-h-[140px] place-items-center rounded-card bg-track/50 px-4 text-center">
            <div>
              <p className="text-[26px] font-bold leading-none text-kpmg-purple">
                ★ {rating2(benchmark.review_scores_rating?.median)}
              </p>
              <p className="mt-1 text-[12.5px] text-foggy">
                Median peer rating. The per-category radar (cleanliness ·
                location · value · accuracy) activates once those review
                subscores are exposed by the comparables service.
              </p>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
