import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  FileText,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";
import { regulatoryRisk } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { CITY_LABELS, type City } from "@/lib/constants";
import type { RiskFlag } from "@/lib/types";

const FLAG: Record<
  RiskFlag,
  { label: string; lead: string; icon: LucideIcon; badge: string; dot: string; ring: string }
> = {
  HIGH: {
    label: "High risk",
    lead: "carries high regulatory risk for short-term letting",
    icon: ShieldAlert,
    badge: "bg-rec-sell-bg text-rausch",
    dot: "bg-rausch",
    ring: "border-[#F3C9D2]",
  },
  MEDIUM: {
    label: "Medium risk",
    lead: "carries moderate regulatory risk for short-term letting",
    icon: ShieldAlert,
    badge: "bg-rec-marginal-bg text-arches",
    dot: "bg-arches",
    ring: "border-[#F6E2C6]",
  },
  LOW: {
    label: "Low risk",
    lead: "carries low regulatory risk for short-term letting",
    icon: ShieldCheck,
    badge: "bg-rec-list-bg text-babu-dark",
    dot: "bg-babu",
    ring: "border-[#BFE9E3]",
  },
  UNKNOWN: {
    label: "Unknown",
    lead: "could not be assessed from the available sources",
    icon: ShieldQuestion,
    badge: "bg-track text-foggy",
    dot: "bg-foggy",
    ring: "border-hairline",
  },
};

/** Split a reason blob into sentence bullets; "" when not worth bulleting. */
function toBullets(reason: string): string[] {
  const parts = reason
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return parts.length >= 2 ? parts.slice(0, 8) : [];
}

/** Prettify a citation: "Barcelona — barcelona_peuat.pdf" → drop ext, underscores. */
function prettySource(src: string): string {
  return src.replace(/\.(pdf|txt|md|docx?)$/i, "").replace(/_/g, " ");
}

export function RegulatoryTab() {
  const navigate = useNavigate();
  const prop = useAppStore((s) => s.activeProperty);

  const query = useQuery({
    queryKey: ["regulatory", prop?.city, prop?.district],
    queryFn: () => regulatoryRisk({ city: prop!.city, neighbourhood: prop!.district }),
    enabled: !!prop,
    staleTime: Infinity,
    retry: false,
  });

  if (!prop) {
    return (
      <Card className="grid min-h-[260px] place-items-center text-center">
        <div>
          <p className="text-[15px] font-semibold text-hof">No active analysis</p>
          <p className="mt-1 text-[14px] text-foggy">
            Start one to assess short-term-let regulatory risk.
          </p>
          <Button className="mt-4" onClick={() => navigate("/new")}>
            Start new analysis
          </Button>
        </div>
      </Card>
    );
  }

  const cityLabel = CITY_LABELS[prop.city as City] ?? prop.city;
  const area = `${prop.district}, ${cityLabel}`;

  if (query.isLoading) {
    return (
      <Card className="grid min-h-[220px] place-items-center text-[14px] text-foggy">
        Checking regulatory sources for {area}…
      </Card>
    );
  }

  if (query.isError || !query.data) {
    return (
      <Card className="border-[#F6E2C6] bg-[#FFF6EC]">
        <div className="flex items-center gap-2.5">
          <ShieldQuestion size={18} className="text-arches" />
          <p className="text-[14px] font-semibold text-hof">
            Regulatory service unavailable
          </p>
        </div>
        <p className="mt-1 text-[13px] text-hof-secondary">
          The regulatory risk service couldn't be reached for {area}. Verify
          short-term-let rules against official municipal and regional sources.
        </p>
        <Button className="mt-4" variant="secondary" onClick={() => query.refetch()}>
          <RefreshCw size={15} /> Retry
        </Button>
      </Card>
    );
  }

  const data = query.data;
  const flag = FLAG[data.risk_flag] ?? FLAG.UNKNOWN;
  const Icon = flag.icon;
  const bullets = toBullets(data.reason);

  return (
    <div className="flex flex-col gap-6">
      {/* Top row */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-[14px] text-foggy">
          Short-term-let regulatory risk for{" "}
          <span className="font-semibold text-hof">{area}</span>
        </p>
        <Button
          variant="secondary"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw size={15} className={query.isFetching ? "animate-spin" : ""} />
          {query.isFetching ? "Re-checking…" : "Re-check"}
        </Button>
      </div>

      {/* Risk badge */}
      <Card className={cn("border", flag.ring)}>
        <div className="flex items-center gap-4">
          <div className={cn("grid h-12 w-12 place-items-center rounded-card", flag.badge)}>
            <Icon size={24} />
          </div>
          <div>
            <span
              className={cn(
                "inline-flex rounded-pill px-3 py-1 text-[13px] font-semibold",
                flag.badge,
              )}
            >
              {flag.label}
            </span>
            <p className="mt-1.5 text-[15px] text-hof-secondary">
              {area} {flag.lead}.
            </p>
          </div>
        </div>
      </Card>

      {/* What drives the rating */}
      <Card>
        <p className="text-[15px] font-semibold text-hof">What drives the rating</p>
        <p className="mb-3 text-[12.5px] text-foggy">
          Assessment from the regulatory knowledge base
        </p>
        {bullets.length > 0 ? (
          <ul className="flex flex-col gap-2.5">
            {bullets.map((b, i) => (
              <li key={i} className="flex gap-3 text-[14px] leading-relaxed text-hof-secondary">
                <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", flag.dot)} />
                <span>{b}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[14px] leading-relaxed text-hof-secondary">{data.reason}</p>
        )}
      </Card>

      {/* Sources */}
      {data.sources.length > 0 && (
        <Card>
          <p className="text-[15px] font-semibold text-hof">Sources</p>
          <p className="mb-3 text-[12.5px] text-foggy">
            Documents cited in this assessment
          </p>
          <div className="flex flex-wrap gap-2">
            {data.sources.map((s, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1.5 rounded-pill border border-[#D6DEFB] bg-[#EFF3FF] px-3 py-1.5 text-[12.5px] font-medium text-kpmg-cobalt"
              >
                <FileText size={13} />
                {prettySource(s)}
              </span>
            ))}
          </div>
        </Card>
      )}

      {/* Not legal advice */}
      <div className="rounded-card border border-[#F4E6C4] bg-[#FFFBF0] px-4 py-3">
        <p className="text-[13px] leading-relaxed text-hof-secondary">
          {data.disclaimer ||
            "This is an automated assessment, not legal advice. Verify short-term-let rules against current official municipal and regional sources before acting."}
        </p>
      </div>
    </div>
  );
}
