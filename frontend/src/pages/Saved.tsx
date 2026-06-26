import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Segmented";
import { RecommendationPill } from "@/components/ui/RecommendationPill";
import { SavedCard } from "@/features/saved/SavedCard";
import { useSavedList, useSavedMutations } from "@/features/saved/useSaved";
import { useAppStore, type SavedView } from "@/store/useAppStore";
import { eur, pct0, signedEur } from "@/lib/format";
import { recProperty, recScenario, recTitle } from "@/lib/saved";
import type { SavedRecord } from "@/lib/types";

export default function Saved() {
  const savedView = useAppStore((s) => s.savedView);
  const setSavedView = useAppStore((s) => s.setSavedView);
  const navigate = useNavigate();

  const saved = useSavedList();
  const { remove, clear } = useSavedMutations();
  const records = saved.data ?? [];

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-[27px] font-semibold tracking-[-0.01em]">
            Your portfolio
          </h1>
          <p className="mt-1 text-[13px] text-foggy">
            {records.length} saved {records.length === 1 ? "analysis" : "analyses"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {records.length > 1 && (
            <Segmented<SavedView>
              options={[
                { value: "list", label: "List" },
                { value: "compare", label: "Compare" },
              ]}
              value={savedView}
              onChange={setSavedView}
            />
          )}
          {records.length > 0 && (
            <Button
              variant="secondary"
              onClick={() => {
                if (
                  window.confirm(
                    "Delete every saved analysis? This can't be undone.",
                  )
                ) {
                  clear.mutate();
                }
              }}
              disabled={clear.isPending}
            >
              <Trash2 size={15} />
              Clear all
            </Button>
          )}
        </div>
      </header>

      {/* Portfolio summary */}
      {records.length > 0 && <PortfolioSummary records={records} />}

      {/* Body */}
      {saved.isLoading ? (
        <Card className="grid min-h-[200px] place-items-center text-[14px] text-foggy">
          Loading your portfolio…
        </Card>
      ) : saved.isError ? (
        <Card className="grid min-h-[200px] place-items-center text-center">
          <p className="text-[14px] text-foggy">
            Couldn't load saved analyses.{" "}
            <button
              className="font-semibold text-kpmg-cobalt underline"
              onClick={() => saved.refetch()}
            >
              Retry
            </button>
          </p>
        </Card>
      ) : records.length === 0 ? (
        <Card className="grid min-h-[240px] place-items-center text-center">
          <div>
            <p className="text-[15px] font-semibold text-hof">
              Nothing saved yet
            </p>
            <p className="mt-1 text-[14px] text-foggy">
              Run an analysis and hit <strong>Save</strong> to keep it here.
            </p>
            <Button className="mt-4" onClick={() => navigate("/new")}>
              Start new analysis
            </Button>
          </div>
        </Card>
      ) : savedView === "compare" && records.length > 1 ? (
        <CompareTable records={records} />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {records.map((r) => (
            <SavedCard
              key={r.id}
              record={r}
              onDelete={(id) => remove.mutate(id)}
              deleting={remove.isPending && remove.variables === r.id}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** Top-line aggregates across the whole portfolio. */
function PortfolioSummary({ records }: { records: SavedRecord[] }) {
  const scens = records.map(recScenario);
  const totalNet = scens.reduce((s, x) => s + (x.net_revenue_year_eur ?? 0), 0);
  const airbnbCount = scens.filter((x) => x.recommendation === "airbnb").length;
  const sellCount = scens.filter((x) => x.recommendation === "sell").length;

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <SummaryStat label="Saved analyses" value={String(records.length)} />
      <SummaryStat label="Total net annual profit" value={eur(totalNet)} />
      <SummaryStat label="Lean to Airbnb" value={`${airbnbCount}`} />
      <SummaryStat label="Lean to sell" value={`${sellCount}`} />
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <p className="text-[11.5px] font-semibold uppercase tracking-[0.04em] text-foggy">
        {label}
      </p>
      <p className="mt-1 text-[24px] font-bold leading-none tracking-[-0.02em] text-hof">
        {value}
      </p>
    </Card>
  );
}

/** Side-by-side metric comparison; click a column header to reopen. */
function CompareTable({ records }: { records: SavedRecord[] }) {
  const navigate = useNavigate();
  const setActiveAnalysis = useAppStore((s) => s.setActiveAnalysis);

  const reopen = (r: SavedRecord) => {
    setActiveAnalysis(recProperty(r), recScenario(r));
    navigate("/workspace");
  };

  const rows: { label: string; render: (r: SavedRecord) => ReactNode }[] = [
    {
      label: "Recommendation",
      render: (r) => (
        <RecommendationPill recommendation={recScenario(r).recommendation} />
      ),
    },
    { label: "Confidence", render: (r) => pct0(recScenario(r).confidence) },
    {
      label: "Predicted nightly",
      render: (r) => eur(recScenario(r).predicted_nightly_eur),
    },
    {
      label: "Occupancy",
      render: (r) => pct0(recScenario(r).occupancy_rate_annual),
    },
    {
      label: "Gross annual revenue",
      render: (r) => eur(recScenario(r).gross_revenue_year_eur),
    },
    {
      label: "Net annual profit",
      render: (r) => (
        <span className="font-semibold text-hof">
          {eur(recScenario(r).net_revenue_year_eur)}
        </span>
      ),
    },
    {
      label: "Sale price",
      render: (r) => eur(recScenario(r).sale_price_eur),
    },
    {
      label: "NPV advantage (Airbnb − Sell)",
      render: (r) => {
        const v = recScenario(r).npv_advantage_eur;
        return (
          <span className={v >= 0 ? "text-babu-dark" : "text-rausch"}>
            {signedEur(v)}
          </span>
        );
      },
    },
    {
      label: "P(Airbnb beats Sell)",
      render: (r) => pct0(recScenario(r).p_airbnb_gt_sell),
    },
    {
      label: "Break-even",
      render: (r) => {
        const y = recScenario(r).breakeven_years;
        return y >= 50 ? "—" : `${y.toFixed(1)} yrs`;
      },
    },
  ];

  return (
    <Card className="overflow-x-auto p-0">
      <table className="w-full border-collapse text-[13.5px]">
        <thead>
          <tr className="border-b border-hairline">
            <th className="sticky left-0 z-10 bg-surface px-4 py-3 text-left text-[11.5px] font-semibold uppercase tracking-[0.04em] text-foggy">
              Metric
            </th>
            {records.map((r) => (
              <th
                key={r.id}
                className="min-w-[150px] px-4 py-3 text-left align-top"
              >
                <button
                  onClick={() => reopen(r)}
                  className="text-left text-[13.5px] font-semibold text-hof hover:text-kpmg-cobalt hover:underline"
                >
                  {recTitle(r)}
                </button>
                <p className="mt-0.5 text-[11.5px] font-normal capitalize text-foggy">
                  {recProperty(r).city}
                </p>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.label}
              className={i % 2 === 0 ? "" : "bg-track/40"}
            >
              <td className="sticky left-0 z-10 bg-inherit px-4 py-2.5 font-medium text-hof-secondary">
                {row.label}
              </td>
              {records.map((r) => (
                <td key={r.id} className="px-4 py-2.5 text-hof">
                  {row.render(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
