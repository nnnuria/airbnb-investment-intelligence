import { cn } from "@/lib/cn";
import type { Recommendation } from "@/lib/types";

const META: Record<
  Recommendation,
  { label: string; bg: string; text: string }
> = {
  airbnb: { label: "List on Airbnb", bg: "bg-rec-list-bg", text: "text-babu-dark" },
  sell: { label: "Sell", bg: "bg-rec-sell-bg", text: "text-rausch" },
  marginal: {
    label: "Marginal",
    bg: "bg-rec-marginal-bg",
    text: "text-arches",
  },
};

export function RecommendationPill({
  recommendation,
  className,
  animate = false,
}: {
  recommendation: Recommendation;
  className?: string;
  animate?: boolean;
}) {
  const m = META[recommendation];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-pill px-3.5 py-1.5",
        "text-[14px] font-semibold",
        m.bg,
        m.text,
        animate && "animate-pop-in",
        className,
      )}
    >
      {m.label}
    </span>
  );
}
