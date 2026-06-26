import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import type { CostLine } from "@/lib/types";
import { eurCompact } from "@/lib/format";

/** Cost waterfall: gross (teal) → each cost step (rausch) → net profit (cobalt). */
export function CostWaterfall({
  costBreakdown,
  gross,
  net,
}: {
  costBreakdown: CostLine[];
  gross: number;
  net: number;
}) {
  const GROSS = "#00A699";
  const COST = "#FCA5B5";
  const NET = "#1E49E2";

  let top = gross;
  const data: { name: string; base: number; value: number; fill: string }[] = [
    { name: "Gross", base: 0, value: gross, fill: GROSS },
  ];
  for (const [label, v] of costBreakdown) {
    const bottom = top - v;
    data.push({ name: label, base: bottom, value: v, fill: COST });
    top = bottom;
  }
  data.push({ name: "Net", base: 0, value: net, fill: NET });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
        <XAxis
          dataKey="name"
          tick={{ fontSize: 10, fill: "#767676" }}
          tickLine={false}
          axisLine={{ stroke: "#EBEBEB" }}
          angle={-25}
          textAnchor="end"
          interval={0}
          height={50}
        />
        <YAxis hide />
        <Bar dataKey="base" stackId="w" fill="transparent" isAnimationActive={false} />
        <Bar dataKey="value" stackId="w" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function waterfallLegend(): string {
  return "Gross revenue, less each annual cost, to net profit.";
}

export { eurCompact };
