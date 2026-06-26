import { Card, CardLabel, CardValue, CardDelta } from "@/components/ui/Card";
import { useCountUp } from "@/lib/useCountUp";

/** KPI card with a count-up value. `format` renders the animated number. */
export function KpiCard({
  label,
  value,
  format,
  delta,
  featured,
  band,
}: {
  label: string;
  value: number;
  format: (n: number) => string;
  delta?: string;
  featured?: boolean;
  band?: { p10: number; median: number; p90: number };
}) {
  const animated = useCountUp(value);
  return (
    <Card featured={featured} className="flex flex-col">
      <CardLabel>{label}</CardLabel>
      <CardValue>{format(animated)}</CardValue>
      {band && <BandBar {...band} />}
      {delta && <CardDelta>{delta}</CardDelta>}
    </Card>
  );
}

function BandBar({ p10, median, p90 }: { p10: number; median: number; p90: number }) {
  const span = Math.max(p90 - p10, 1);
  const pos = Math.min(100, Math.max(0, ((median - p10) / span) * 100));
  return (
    <div className="mt-3 h-1.5 w-full rounded-pill bg-track">
      <div className="relative h-full">
        <span
          className="absolute top-1/2 h-2.5 w-2.5 -translate-y-1/2 rounded-full bg-babu"
          style={{ left: `calc(${pos}% - 5px)` }}
        />
      </div>
    </div>
  );
}
