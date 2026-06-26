import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";
import type { Comp } from "@/lib/market";

const AXES: { key: keyof Comp; label: string }[] = [
  { key: "review_scores_cleanliness", label: "Cleanliness" },
  { key: "review_scores_location", label: "Location" },
  { key: "review_scores_value", label: "Value" },
  { key: "review_scores_accuracy", label: "Accuracy" },
];

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Peer guest-rating profile across the four review subscores. Renders only when
 * the comps carry subscore columns; the parent decides the fallback otherwise
 * (these columns aren't in comparables.py DISPLAY_COLUMNS yet).
 */
export function RatingRadar({ comps }: { comps: Comp[] }) {
  const data = AXES.map(({ key, label }) => {
    const vals = comps
      .map((c) => c[key])
      .filter((v): v is number => typeof v === "number");
    return { axis: label, value: mean(vals) ?? 0 };
  });

  return (
    <ResponsiveContainer width="100%" height={240}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="#EBEBEB" />
        <PolarAngleAxis
          dataKey="axis"
          tick={{ fontSize: 11, fill: "#484848" }}
        />
        <PolarRadiusAxis
          domain={[4, 5]}
          tick={{ fontSize: 9, fill: "#767676" }}
          axisLine={false}
        />
        <Radar
          name="Peer average"
          dataKey="value"
          stroke="#6E2BF5"
          fill="#6E2BF5"
          fillOpacity={0.18}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
