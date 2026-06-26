import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
} from "recharts";
import { eurCompact } from "@/lib/format";

/**
 * Net-profit distribution — a shaded curve synthesized from the P10/median/P90
 * the engine returns (assumes a roughly normal shape; P10–P90 ≈ 2.563σ), with
 * dashed percentile markers. KPMG cobalt fill.
 */
export function DistributionChart({
  p10,
  median,
  p90,
}: {
  p10: number;
  median: number;
  p90: number;
}) {
  const sigma = Math.max((p90 - p10) / 2.563, 1);
  const lo = p10 - 1.5 * sigma;
  const hi = p90 + 1.5 * sigma;
  const N = 56;
  const data = Array.from({ length: N }, (_, i) => {
    const x = lo + ((hi - lo) * i) / (N - 1);
    const y = Math.exp(-0.5 * ((x - median) / sigma) ** 2);
    return { x, y };
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="distFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1E49E2" stopOpacity={0.25} />
            <stop offset="100%" stopColor="#1E49E2" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="x"
          type="number"
          domain={[lo, hi]}
          tickFormatter={(v) => eurCompact(v)}
          tick={{ fontSize: 11, fill: "#9A9A9A" }}
          tickLine={false}
          axisLine={{ stroke: "#EBEBEB" }}
        />
        <Area
          type="monotone"
          dataKey="y"
          stroke="#1E49E2"
          strokeWidth={2}
          fill="url(#distFill)"
          isAnimationActive
        />
        <ReferenceLine x={p10} stroke="#9A9A9A" strokeDasharray="3 3" label={{ value: "P10", position: "top", fontSize: 10, fill: "#767676" }} />
        <ReferenceLine x={median} stroke="#1E49E2" strokeDasharray="3 3" label={{ value: "P50", position: "top", fontSize: 10, fill: "#1E49E2" }} />
        <ReferenceLine x={p90} stroke="#9A9A9A" strokeDasharray="3 3" label={{ value: "P90", position: "top", fontSize: 10, fill: "#767676" }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
