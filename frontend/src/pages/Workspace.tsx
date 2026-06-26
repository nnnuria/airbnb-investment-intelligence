import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";
import { useAppStore, type WorkspaceTab } from "@/store/useAppStore";
import { DecisionTab } from "@/features/decision/DecisionTab";
import { MarketTab } from "@/features/market/MarketTab";
import { OptimiseTab } from "@/features/optimise/OptimiseTab";
import { RegulatoryTab } from "@/features/regulatory/RegulatoryTab";
import { CITY_LABELS, type City } from "@/lib/constants";

const TABS: { value: WorkspaceTab; label: string }[] = [
  { value: "decision", label: "Decision" },
  { value: "market", label: "Market" },
  { value: "optimise", label: "Optimise" },
  { value: "regulatory", label: "Regulatory" },
];

export default function Workspace() {
  const tab = useAppStore((s) => s.tab);
  const setTab = useAppStore((s) => s.setTab);
  const prop = useAppStore((s) => s.activeProperty);

  const title = prop?.nickname || prop?.district || "Active property";
  const subline = prop
    ? `${prop.district}, ${CITY_LABELS[prop.city as City] ?? prop.city} · ${prop.size_m2} m² · ${prop.bedrooms} bed · ${prop.bathrooms} bath · sleeps ${prop.accommodates}`
    : "Run an analysis to populate the workspace.";

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow">Analysis workspace</p>
          <h1 className="mt-1 text-[27px] font-semibold tracking-[-0.01em]">{title}</h1>
          <p className="mt-1 text-[14px] text-foggy">{subline}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">Share</Button>
          <Button variant="dark">Export report</Button>
        </div>
      </header>

      {/* Tabs */}
      <div className="flex gap-6 border-b border-hairline">
        {TABS.map((t) => (
          <button
            key={t.value}
            onClick={() => setTab(t.value)}
            className={cn(
              "-mb-px border-b-2 pb-3 text-[15px] font-semibold transition-colors",
              tab === t.value
                ? "border-hof text-hof"
                : "border-transparent text-foggy hover:text-hof",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab body */}
      <div key={tab} className="animate-tab-enter">
        {tab === "decision" ? (
          <DecisionTab />
        ) : tab === "market" ? (
          <MarketTab />
        ) : tab === "optimise" ? (
          <OptimiseTab />
        ) : (
          <RegulatoryTab />
        )}
      </div>
    </div>
  );
}
