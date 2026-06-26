import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

const COLORS = {
  positive: "#00A699",
  neutral: "#FCD9A8",
  negative: "#FF385C",
} as const;

/**
 * Sentiment donut from the rating-based proxy (positive/neutral/negative
 * shares). Centre shows the positive share.
 */
export function SentimentDonut({
  positive,
  neutral,
  negative,
}: {
  positive: number;
  neutral: number;
  negative: number;
}) {
  const data = [
    { name: "Positive", value: positive, key: "positive" as const },
    { name: "Neutral", value: neutral, key: "neutral" as const },
    { name: "Negative", value: negative, key: "negative" as const },
  ].filter((d) => d.value > 0);

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            innerRadius={56}
            outerRadius={78}
            startAngle={90}
            endAngle={-270}
            paddingAngle={2}
            stroke="none"
          >
            {data.map((d) => (
              <Cell key={d.key} fill={COLORS[d.key]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[26px] font-bold leading-none text-babu">
          {positive}%
        </span>
        <span className="text-[11px] text-foggy">positive</span>
      </div>
    </div>
  );
}
