import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureDriver } from "@/lib/types";

/** Signed feature-driver bars from a centre axis: positive = blue→purple, negative = grey. */
export function FeatureDrivers({ drivers }: { drivers: FeatureDriver[] }) {
  const top = [...drivers]
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8)
    .map(([name, frac]) => ({ name, value: frac * 100 }));

  const max = Math.max(1, ...top.map((d) => Math.abs(d.value)));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={top}
        layout="vertical"
        margin={{ top: 4, right: 16, bottom: 4, left: 8 }}
      >
        <defs>
          <linearGradient id="posDriver" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#1E49E2" />
            <stop offset="100%" stopColor="#6E2BF5" />
          </linearGradient>
        </defs>
        <XAxis type="number" domain={[-max, max]} hide />
        <YAxis
          type="category"
          dataKey="name"
          width={132}
          tick={{ fontSize: 11, fill: "#484848" }}
          tickLine={false}
          axisLine={false}
        />
        <ReferenceLine x={0} stroke="#767676" />
        <Bar dataKey="value" radius={[3, 3, 3, 3]} isAnimationActive>
          {top.map((d, i) => (
            <Cell key={i} fill={d.value >= 0 ? "url(#posDriver)" : "#CBD0D6"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
