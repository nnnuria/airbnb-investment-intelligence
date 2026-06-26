import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ShieldCheck, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card, CardLabel } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Segmented";
import { RecommendationPill } from "@/components/ui/RecommendationPill";
import { ConfidenceGauge } from "@/components/ui/ConfidenceGauge";
import { KpiCard } from "./KpiCard";
import { DistributionChart } from "./charts/DistributionChart";
import { SeasonalityRing } from "./charts/SeasonalityRing";
import { CostWaterfall } from "./charts/CostWaterfall";
import { FeatureDrivers } from "./charts/FeatureDrivers";
import { getScenario, regulatoryRisk } from "@/lib/api";
import { useSavedMutations } from "@/features/saved/useSaved";
import { useAppStore } from "@/store/useAppStore";
import { REC_META } from "@/lib/recommendation";
import { eur, eurCompact, pct0, signedEur, years } from "@/lib/format";
import type { ScenarioResponse } from "@/lib/types";

export function DecisionTab() {
  const navigate = useNavigate();
  const scen = useAppStore((s) => s.activeScenario);
  const prop = useAppStore((s) => s.activeProperty);
  const managed = useAppStore((s) => s.managed);
  const setActiveAnalysis = useAppStore((s) => s.setActiveAnalysis);
  const setManaged = useAppStore((s) => s.setManaged);
  const setTab = useAppStore((s) => s.setTab);
  const setCopilotOpen = useAppStore((s) => s.setCopilotOpen);

  const { create: saveAnalysis } = useSavedMutations();

  const recompute = useMutation({
    mutationFn: (nextManaged: boolean) =>
      getScenario({ ...prop!, managed: nextManaged }),
    onSuccess: (scenario, nextManaged) => {
      setActiveAnalysis({ ...prop!, managed: nextManaged }, scenario);
      setManaged(nextManaged);
      saveAnalysis.reset(); // re-enable "Save" for the freshly recomputed scenario
    },
  });

  // Optional, non-blocking regulatory banner (graceful on error/rate-limit).
  const reg = useQuery({
    queryKey: ["regulatory", prop?.city, prop?.district],
    queryFn: () => regulatoryRisk({ city: prop!.city, neighbourhood: prop!.district }),
    enabled: !!prop,
    staleTime: Infinity,
    retry: false,
  });

  if (!scen || !prop) {
    return (
      <Card className="grid min-h-[260px] place-items-center text-center">
        <div>
          <p className="text-[15px] font-semibold text-hof">No active analysis</p>
          <p className="mt-1 text-[14px] text-foggy">
            Start one to see the recommendation here.
          </p>
          <Button className="mt-4" onClick={() => navigate("/new")}>
            Start new analysis
          </Button>
        </div>
      </Card>
    );
  }

  const meta = REC_META[scen.recommendation];
  const rate = Math.round(scen.discount_rate * 100);

  return (
    <div className="flex flex-col gap-6">
      {/* 1. Banners */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <GovernanceBanner governance={scen.governance} />
        {reg.data && reg.data.risk_flag !== "LOW" && (
          <RegulatoryBanner flag={reg.data.risk_flag} reason={reg.data.reason} />
        )}
      </div>

      {/* 2. Hero recommendation */}
      <Card className="rounded-hero bg-gradient-to-b from-[#F0FBF9] to-surface">
        <div className="flex flex-col items-center gap-6 md:flex-row md:items-center">
          <ConfidenceGauge value={scen.confidence} color={meta.hex} />
          <div className="flex-1 text-center md:text-left">
            <RecommendationPill recommendation={scen.recommendation} animate />
            <p className="mt-2 text-[16px] text-hof-secondary">
              {scen.npv_advantage_eur >= 0 ? meta.winsVerb : REC_META.sell.winsVerb}{" "}
              over a {scen.holding_years}-year hold.
            </p>
          </div>
          <div className="flex flex-col items-center gap-2 md:items-end">
            <span className="text-[12px] font-semibold uppercase tracking-[0.04em] text-foggy">
              Management model
            </span>
            <Segmented
              options={[
                { value: "managed", label: "Managed" },
                { value: "self", label: "Self-managed" },
              ]}
              value={managed ? "managed" : "self"}
              onChange={(v) => recompute.mutate(v === "managed")}
            />
            <span className="h-4 text-[11px] text-foggy">
              {recompute.isPending ? "Recomputing…" : ""}
            </span>
          </div>
        </div>
      </Card>

      {/* 3. Actions */}
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="dark"
          disabled={saveAnalysis.isPending || saveAnalysis.isSuccess}
          onClick={() =>
            saveAnalysis.mutate({
              property: prop as unknown as Record<string, unknown>,
              scenario: scen as unknown as Record<string, unknown>,
              nickname: (prop.nickname as string | null) ?? null,
            })
          }
        >
          {saveAnalysis.isSuccess
            ? "Saved ✓"
            : saveAnalysis.isPending
              ? "Saving…"
              : "Save analysis"}
        </Button>
        <Button variant="secondary" onClick={() => setCopilotOpen(true)}>
          Ask a follow-up
        </Button>
        <Button variant="purple-outline" onClick={() => setTab("optimise")}>
          See improvement ideas →
        </Button>
        {saveAnalysis.isError && (
          <span className="text-[13px] text-rausch">
            Couldn't save — is the API running?
          </span>
        )}
      </div>

      {/* 4. KPI cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <KpiCard
          label="Gross annual revenue"
          value={scen.gross_revenue_year_eur}
          format={eur}
          delta={`${eur(scen.predicted_nightly_eur)}/night · ${scen.nights_booked_year} nights`}
        />
        <KpiCard
          label="Net annual profit"
          value={scen.net_revenue_year_eur}
          format={eur}
          featured
          band={{
            p10: scen.net_revenue_p10_eur,
            median: scen.net_revenue_year_eur,
            p90: scen.net_revenue_p90_eur,
          }}
          delta={`P10 ${eurCompact(scen.net_revenue_p10_eur)} / P90 ${eurCompact(scen.net_revenue_p90_eur)}`}
        />
        <KpiCard
          label="Indicative sale value"
          value={scen.sale_price_eur}
          format={eur}
          delta={`${eur(scen.sale_price_per_m2_eur)}/m²`}
        />
        <KpiCard
          label="Payback period"
          value={scen.breakeven_years}
          format={years}
          delta="to recover sale value"
        />
        <KpiCard
          label="Projected occupancy"
          value={scen.occupancy_rate_annual}
          format={pct0}
          delta={`${scen.nights_booked_year} nights/year`}
        />
      </div>

      {/* 5. Investment comparison */}
      <InvestmentComparison scen={scen} rate={rate} />

      {/* 6. Charts grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Net-profit distribution" subtitle="P10 / median / P90 after all costs">
          <DistributionChart
            p10={scen.net_revenue_p10_eur}
            median={scen.net_revenue_year_eur}
            p90={scen.net_revenue_p90_eur}
          />
        </ChartCard>
        <ChartCard title="Seasonality of demand" subtitle="Monthly demand index (baseline 1.0)">
          <SeasonalityRing values={scen.monthly_seasonality} />
        </ChartCard>
        <ChartCard title="Cost waterfall" subtitle="Gross revenue → costs → net profit">
          <CostWaterfall
            costBreakdown={scen.cost_breakdown}
            gross={scen.gross_revenue_year_eur}
            net={scen.net_revenue_year_eur}
          />
        </ChartCard>
        <ChartCard title="Why this recommendation" subtitle="Top feature contributions">
          <FeatureDrivers drivers={scen.feature_drivers} />
        </ChartCard>
      </div>
    </div>
  );
}

// ── sub-components ────────────────────────────────────────────────────────────

function GovernanceBanner({ governance }: { governance: Record<string, unknown> }) {
  const violations = (governance?.violations as { message: string }[]) ?? [];
  const needsReview = Boolean(governance?.human_review_required);
  if (!needsReview && violations.length === 0) {
    return (
      <div className="flex items-center gap-2.5 rounded-card border border-[#D6DEFB] bg-[#EFF3FF] px-4 py-3">
        <ShieldCheck size={18} className="text-kpmg-cobalt" />
        <p className="text-[13px] text-hof-secondary">
          Governance review passed — no policy violations, no human review required.
        </p>
      </div>
    );
  }
  return (
    <div className="rounded-card border border-[#F6E2C6] bg-[#FFF6EC] px-4 py-3">
      <div className="flex items-center gap-2.5">
        <ShieldAlert size={18} className="text-arches" />
        <p className="text-[13px] font-semibold text-hof">Governance review flagged this analysis</p>
      </div>
      {violations.length > 0 && (
        <ul className="mt-1 list-disc pl-9 text-[13px] text-hof-secondary">
          {violations.map((v, i) => (
            <li key={i}>{v.message}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function RegulatoryBanner({ flag, reason }: { flag: string; reason: string }) {
  return (
    <div className="rounded-card border border-[#F6E2C6] bg-[#FFF6EC] px-4 py-3">
      <div className="flex items-center gap-2.5">
        <ShieldAlert size={18} className="text-arches" />
        <p className="text-[13px] font-semibold text-hof">Regulatory risk: {flag}</p>
      </div>
      <p className="mt-1 line-clamp-2 pl-9 text-[13px] text-hof-secondary">{reason}</p>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <p className="text-[15px] font-semibold text-hof">{title}</p>
      <p className="mb-3 text-[12.5px] text-foggy">{subtitle}</p>
      {children}
    </Card>
  );
}

function InvestmentComparison({
  scen,
  rate,
}: {
  scen: ScenarioResponse;
  rate: number;
}) {
  const bars = [
    { label: "Keep & list — NPV after-tax", value: scen.npv_airbnb_p50_eur, color: "#00A699" },
    { label: "Keep & list — NPV pre-tax", value: scen.npv_airbnb_pretax_p50_eur, color: "#7FD4CC" },
    { label: "Sell — net after CGT", value: scen.npv_sell_eur, color: "#1E49E2" },
    { label: "Sell — gross pre-tax", value: scen.npv_sell_pretax_eur, color: "#AFC0FA" },
  ];
  const max = Math.max(1, ...bars.map((b) => Math.abs(b.value)));
  const adv = scen.npv_advantage_eur;
  const airbnbPct = Math.min(
    92,
    Math.max(8, (scen.npv_airbnb_p50_eur / (scen.npv_airbnb_p50_eur + scen.npv_sell_eur || 1)) * 100),
  );

  return (
    <Card>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[15px] font-semibold text-hof">
          Investment comparison ({scen.holding_years}-year horizon)
        </p>
        <p className="text-[13px] text-foggy">
          IRR {scen.irr_airbnb_pct != null ? `${(scen.irr_airbnb_pct * 100).toFixed(1)}%` : "—"} · Airbnb · {rate}% discount
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left: four bars */}
        <div className="flex flex-col gap-3">
          {bars.map((b) => (
            <div key={b.label}>
              <div className="flex justify-between text-[13px]">
                <span className="text-hof-secondary">{b.label}</span>
                <span className="font-semibold text-hof">{eur(b.value)}</span>
              </div>
              <div className="mt-1 h-2 w-full rounded-pill bg-track">
                <div
                  className="h-full rounded-pill transition-all"
                  style={{ width: `${(Math.abs(b.value) / max) * 100}%`, background: b.color }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Right: NPV advantage centrepiece */}
        <div className="flex flex-col justify-center rounded-card bg-track/60 p-5 text-center">
          <CardLabel className="text-center">NPV advantage · Airbnb − Sell</CardLabel>
          <p
            className="text-[40px] font-bold tracking-[-0.02em]"
            style={{ color: adv >= 0 ? "#00A699" : "#FF385C" }}
          >
            {signedEur(adv)}
          </p>
          <p className="text-[14px] font-semibold text-hof-secondary">
            {adv >= 0 ? "Listing wins" : "Selling wins"} over a {scen.holding_years}-year hold
          </p>
          <div className="mt-4 flex h-2.5 overflow-hidden rounded-pill">
            <div style={{ width: `${airbnbPct}%`, background: "#00A699" }} />
            <div style={{ width: `${100 - airbnbPct}%`, background: "#FF385C" }} />
          </div>
          <div className="mt-1 flex justify-between text-[11px] text-foggy">
            <span>Airbnb {eurCompact(scen.npv_airbnb_p50_eur)}</span>
            <span>Sell {eurCompact(scen.npv_sell_eur)}</span>
          </div>
        </div>
      </div>

      <p className="mt-4 text-[12px] leading-relaxed text-foggy">
        Assumes IBI {eur(scen.ibi_eur_used)}/yr, basuras {eur(scen.basuras_eur_used)}/yr,{" "}
        {eur(scen.setup_cost_eur_used)} setup, {rate}% discount rate, {scen.holding_years}-year
        holding. After-tax figures apply your marginal income-tax rate and Spanish CGT on sale.
      </p>
    </Card>
  );
}
