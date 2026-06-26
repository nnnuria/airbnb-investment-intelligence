import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Check, Plus, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { KpiCard } from "@/features/decision/KpiCard";
import { cn } from "@/lib/cn";
import { optimise } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { eur } from "@/lib/format";
import type { OptimiseAction } from "@/lib/types";

/** Months → "8 mo" / "1.4 yr"; non-finite → "—". */
function paybackLabel(months: number): string {
  if (!isFinite(months) || months <= 0) return "—";
  return months >= 12 ? `${(months / 12).toFixed(1)} yr` : `${Math.round(months)} mo`;
}

const CONF: Record<string, { label: string; cls: string }> = {
  high: { label: "High confidence", cls: "bg-rec-list-bg text-babu-dark" },
  medium: { label: "Medium confidence", cls: "bg-rec-marginal-bg text-arches" },
  low: { label: "Low confidence", cls: "bg-track text-foggy" },
};

/** Per-row planner state: included in the plan + the user's editable cost. */
interface RowState {
  included: boolean;
  cost: number;
}

export function OptimiseTab() {
  const navigate = useNavigate();
  const prop = useAppStore((s) => s.activeProperty);

  const query = useQuery({
    queryKey: [
      "optimise",
      prop?.city,
      prop?.district,
      prop?.room_type,
      prop?.bedrooms,
      prop?.accommodates,
      prop?.managed,
    ],
    // extra="allow" on the backend — pass the property through (incl. has_*
    // amenity flags so it won't recommend what's already installed).
    queryFn: () => optimise({ ...prop! }),
    enabled: !!prop,
    staleTime: 5 * 60_000,
    retry: false,
  });

  const actions = query.data?.actions ?? [];

  // Local planner state, re-initialised whenever a fresh plan arrives.
  const [rows, setRows] = useState<RowState[]>([]);
  useEffect(() => {
    setRows(actions.map((a) => ({ included: true, cost: a.investment_eur })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.data]);

  if (!prop) {
    return (
      <Card className="grid min-h-[260px] place-items-center text-center">
        <div>
          <p className="text-[15px] font-semibold text-hof">No active analysis</p>
          <p className="mt-1 text-[14px] text-foggy">
            Start one to plan revenue improvements.
          </p>
          <Button className="mt-4" onClick={() => navigate("/new")}>
            Start new analysis
          </Button>
        </div>
      </Card>
    );
  }

  const setRow = (i: number, patch: Partial<RowState>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));

  const includedCount = rows.filter((r) => r.included).length;
  const totalUplift = rows.reduce(
    (sum, r, i) => sum + (r.included ? actions[i]?.annual_uplift_eur ?? 0 : 0),
    0,
  );
  const totalCost = rows.reduce((sum, r) => sum + (r.included ? r.cost : 0), 0);
  const blendedPayback = totalUplift > 0 ? (totalCost / totalUplift) * 12 : Infinity;

  const d = query.data;

  return (
    <div className="flex flex-col gap-6 pb-24">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[15px] font-semibold text-hof">Revenue improvements</p>
          <p className="text-[12.5px] text-foggy">
            {d ? `Benchmarked against ${d.peer_n} peer listings` : "Building your plan…"}
          </p>
        </div>
        <Button
          variant="secondary"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw size={15} className={query.isFetching ? "animate-spin" : ""} />
          {query.isFetching ? "Refreshing…" : "Regenerate"}
        </Button>
      </div>

      {/* Peer-gap summary */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard
          label="Peer median revenue"
          value={d?.peer_median_revenue_eur ?? 0}
          format={eur}
          delta="across comparable listings"
        />
        <KpiCard
          label="Target revenue"
          value={d?.peer_target_revenue_eur ?? 0}
          format={eur}
          featured
          delta="top-quartile peers"
        />
        <KpiCard
          label="Gap to top quartile"
          value={d?.gap_to_top_quartile_eur ?? 0}
          format={eur}
          delta="closeable with improvements"
        />
        <KpiCard
          label="Projected annual nights"
          value={d?.projected_annual_nights ?? 0}
          format={(n) => `${Math.round(n)}`}
          delta="at current pricing"
        />
      </div>

      {/* Improvement list */}
      {query.isError ? (
        <Card className="border-[#F6E2C6] bg-[#FFF6EC] text-[13px] text-hof-secondary">
          Couldn't build an improvement plan.{" "}
          <button
            className="font-semibold text-kpmg-cobalt underline"
            onClick={() => query.refetch()}
          >
            Retry
          </button>
          .
        </Card>
      ) : query.isLoading ? (
        <Card className="grid min-h-[200px] place-items-center text-[14px] text-foggy">
          Finding the highest-impact improvements…
        </Card>
      ) : actions.length === 0 ? (
        <Card className="grid min-h-[160px] place-items-center text-center text-[14px] text-foggy">
          This property is already near the top of its peer set — no high-impact
          improvements found.
        </Card>
      ) : (
        <div className="flex flex-col gap-3">
          {actions.map((a, i) => (
            <ImprovementRow
              key={`${a.action}-${i}`}
              action={a}
              state={rows[i] ?? { included: true, cost: a.investment_eur }}
              onToggle={() => setRow(i, { included: !rows[i]?.included })}
              onCost={(cost) => setRow(i, { cost })}
            />
          ))}
        </div>
      )}

      {d?.disclaimer && (
        <p className="text-[12px] leading-relaxed text-foggy">{d.disclaimer}</p>
      )}

      {/* Live selection summary (sticky) */}
      {actions.length > 0 && (
        <div className="sticky bottom-4 z-10 flex flex-wrap items-center justify-between gap-4 rounded-card bg-hof px-5 py-4 text-white shadow-lift">
          <div>
            <p className="text-[12px] uppercase tracking-[0.04em] text-white/60">
              Your selection
            </p>
            <p className="text-[15px] font-semibold">
              {includedCount} of {actions.length} improvements
            </p>
          </div>
          <div className="flex flex-wrap gap-6">
            <SummaryStat label="Total annual uplift" value={eur(totalUplift)} accent />
            <SummaryStat label="Total investment" value={eur(totalCost)} />
            <SummaryStat label="Blended payback" value={paybackLabel(blendedPayback)} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── sub-components ────────────────────────────────────────────────────────────

function SummaryStat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-[0.04em] text-white/60">{label}</p>
      <p
        className={cn("text-[18px] font-bold", accent ? "text-babu" : "text-white")}
      >
        {value}
      </p>
    </div>
  );
}

function ImprovementRow({
  action,
  state,
  onToggle,
  onCost,
}: {
  action: OptimiseAction;
  state: RowState;
  onToggle: () => void;
  onCost: (cost: number) => void;
}) {
  const conf = CONF[action.confidence] ?? CONF.low;
  const payback = state.cost > 0 && action.annual_uplift_eur > 0
    ? (state.cost / action.annual_uplift_eur) * 12
    : Infinity;

  return (
    <div
      className={cn(
        "rounded-card border bg-surface p-4 shadow-card transition-colors",
        state.included ? "border-babu" : "border-hairline opacity-80",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <button
            type="button"
            onClick={onToggle}
            className={cn(
              "mt-0.5 flex items-center gap-1.5 rounded-pill px-3 py-1.5 text-[12.5px] font-semibold transition-colors",
              state.included
                ? "bg-babu text-white"
                : "border border-inputborder text-hof hover:border-foggy",
            )}
          >
            {state.included ? <Check size={13} /> : <Plus size={13} />}
            {state.included ? "Included" : "Add"}
          </button>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-[14.5px] font-semibold text-hof">{action.action}</p>
              <span
                className={cn(
                  "rounded-pill px-2 py-0.5 text-[11px] font-semibold",
                  conf.cls,
                )}
              >
                {conf.label}
              </span>
            </div>
            <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-hof-secondary">
              {action.rationale}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-5">
          <div className="text-right">
            <p className="text-[11px] uppercase tracking-[0.04em] text-foggy">
              Annual uplift
            </p>
            <p className="text-[16px] font-bold text-babu">
              +{eur(action.annual_uplift_eur)}
            </p>
          </div>
          <div className="text-right">
            <label className="text-[11px] uppercase tracking-[0.04em] text-foggy">
              Your cost (€)
            </label>
            <input
              type="number"
              min={0}
              step={50}
              value={state.cost}
              onChange={(e) => onCost(Math.max(0, Number(e.target.value) || 0))}
              className="mt-0.5 w-28 rounded-input border border-inputborder px-2.5 py-1.5 text-right text-[14px] text-hof focus:border-foggy focus:outline-none"
            />
          </div>
          <div className="text-right">
            <p className="text-[11px] uppercase tracking-[0.04em] text-foggy">
              Payback
            </p>
            <p className="text-[16px] font-semibold text-hof">
              {paybackLabel(payback)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
