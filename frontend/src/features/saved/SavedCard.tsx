import { useNavigate } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { RecommendationPill } from "@/components/ui/RecommendationPill";
import { useAppStore } from "@/store/useAppStore";
import { eur, signedEur } from "@/lib/format";
import { recDate, recProperty, recScenario, recTitle } from "@/lib/saved";
import type { SavedRecord } from "@/lib/types";

/**
 * One saved analysis. Click to reopen (rehydrates the active analysis →
 * Workspace); the trash icon deletes it (its handler is owned by the parent so
 * the list can manage the mutation + cache invalidation).
 */
export function SavedCard({
  record,
  onDelete,
  deleting = false,
}: {
  record: SavedRecord;
  onDelete?: (id: string) => void;
  deleting?: boolean;
}) {
  const navigate = useNavigate();
  const setActiveAnalysis = useAppStore((s) => s.setActiveAnalysis);
  const setManaged = useAppStore((s) => s.setManaged);

  const scen = recScenario(record);
  const prop = recProperty(record);

  const reopen = () => {
    setActiveAnalysis(prop, scen);
    if (typeof prop.managed === "boolean") setManaged(prop.managed);
    navigate("/workspace");
  };

  return (
    <Card
      hover
      role="button"
      tabIndex={0}
      onClick={reopen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          reopen();
        }
      }}
      className="group relative flex cursor-pointer flex-col gap-3"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-[15px] font-semibold text-hof">
            {recTitle(record)}
          </p>
          <p className="mt-0.5 text-[12.5px] capitalize text-foggy">
            {prop.city}
            {recDate(record) ? ` · saved ${recDate(record)}` : ""}
          </p>
        </div>
        {onDelete && (
          <button
            aria-label="Delete saved analysis"
            disabled={deleting}
            onClick={(e) => {
              e.stopPropagation();
              onDelete(record.id);
            }}
            className="shrink-0 rounded-md p-1.5 text-foggy opacity-0 transition-colors hover:bg-rec-sell-bg hover:text-rausch group-hover:opacity-100 disabled:opacity-40"
          >
            <Trash2 size={15} />
          </button>
        )}
      </div>

      <RecommendationPill recommendation={scen.recommendation} className="self-start" />

      <div className="mt-1 flex items-end justify-between gap-3">
        <div>
          <p className="text-[11.5px] font-semibold uppercase tracking-[0.04em] text-foggy">
            Net annual profit
          </p>
          <p className="text-[22px] font-bold leading-none tracking-[-0.02em] text-hof">
            {eur(scen.net_revenue_year_eur)}
          </p>
        </div>
        <div className="text-right">
          <p className="text-[11.5px] font-semibold uppercase tracking-[0.04em] text-foggy">
            NPV gap
          </p>
          <p
            className={
              "text-[15px] font-semibold " +
              (scen.npv_advantage_eur >= 0 ? "text-babu-dark" : "text-rausch")
            }
          >
            {signedEur(scen.npv_advantage_eur)}
          </p>
        </div>
      </div>
    </Card>
  );
}
