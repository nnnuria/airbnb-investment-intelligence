import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { eur, eurCompact } from "@/lib/format";
import type { Comp } from "@/lib/market";

interface Point {
  x: number; // nightly €
  y: number; // annual revenue €
  name: string;
}

/**
 * Peers plotted nightly-price (x) vs annual-revenue (y) as blue dots; the
 * subject property is a larger Rausch dot labelled "You". Only comps that carry
 * both a price and a revenue figure are plotted.
 */
export function PositioningScatter({
  comps,
  subject,
}: {
  comps: Comp[];
  subject: { nightly: number; revenue: number };
}) {
  const peers: Point[] = comps
    .filter((c) => c.price != null && c.estimated_revenue_l365d != null)
    .map((c) => ({
      x: c.price as number,
      y: c.estimated_revenue_l365d as number,
      name: c.name || c.neighbourhood_cleansed || "Comparable",
    }));

  const you: Point[] = [
    { x: subject.nightly, y: subject.revenue, name: "You" },
  ];

  return (
    <ResponsiveContainer width="100%" height={240}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
        <CartesianGrid stroke="#EBEBEB" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          name="Nightly"
          tickFormatter={(v) => eurCompact(v)}
          tick={{ fontSize: 11, fill: "#767676" }}
          tickLine={false}
          axisLine={{ stroke: "#EBEBEB" }}
          label={{
            value: "Nightly rate",
            position: "insideBottom",
            offset: -12,
            fontSize: 11,
            fill: "#767676",
          }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Annual revenue"
          tickFormatter={(v) => eurCompact(v)}
          tick={{ fontSize: 11, fill: "#767676" }}
          tickLine={false}
          axisLine={{ stroke: "#EBEBEB" }}
          width={48}
        />
        <ZAxis range={[60, 60]} />
        <Tooltip
          cursor={{ stroke: "#EBEBEB" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as Point;
            return (
              <div className="rounded-card border border-hairline bg-surface px-3 py-2 text-[12px] shadow-card">
                <p className="font-semibold text-hof">{p.name}</p>
                <p className="text-foggy">
                  {eur(p.x)}/night · {eur(p.y)}/yr
                </p>
              </div>
            );
          }}
        />
        <Scatter data={peers} fill="#1E49E2" fillOpacity={0.55} />
        <Scatter
          data={you}
          fill="#FF385C"
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          shape={(props: any) => (
            <g>
              <circle cx={props.cx} cy={props.cy} r={8} fill="#FF385C" />
              <text
                x={props.cx}
                y={props.cy - 12}
                textAnchor="middle"
                fontSize={10}
                fontWeight={700}
                fill="#FF385C"
              >
                You
              </text>
            </g>
          )}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
