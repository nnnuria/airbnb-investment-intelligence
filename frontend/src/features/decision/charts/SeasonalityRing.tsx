const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}

function arc(cx: number, cy: number, rO: number, rI: number, start: number, end: number) {
  const [x1, y1] = polar(cx, cy, rO, end);
  const [x2, y2] = polar(cx, cy, rO, start);
  const [x3, y3] = polar(cx, cy, rI, start);
  const [x4, y4] = polar(cx, cy, rI, end);
  const large = end - start <= 180 ? 0 : 1;
  return `M${x1} ${y1} A${rO} ${rO} 0 ${large} 0 ${x2} ${y2} L${x3} ${y3} A${rI} ${rI} 0 ${large} 1 ${x4} ${y4} Z`;
}

function hexLerp(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const m = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `#${m.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** Radial 12-month demand heat ring (teal scale; baseline = 1.0). */
export function SeasonalityRing({ values }: { values: number[] }) {
  const size = 230;
  const cx = size / 2;
  const cy = size / 2;
  const rO = 100;
  const rI = 64;
  const gap = 3;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const peakIdx = values.indexOf(max);

  return (
    <div className="flex items-center justify-center">
      <svg width={size} height={size}>
        {values.map((v, i) => {
          const start = i * 30 + gap / 2;
          const end = (i + 1) * 30 - gap / 2;
          const t = max === min ? 0.5 : (v - min) / (max - min);
          const fill = hexLerp("#D7EFEC", "#00A699", t);
          return <path key={i} d={arc(cx, cy, rO, rI, start, end)} fill={fill} />;
        })}
        {/* month ticks */}
        {MONTHS.map((m, i) => {
          const [x, y] = polar(cx, cy, rO + 12, i * 30 + 15);
          return (
            <text key={m} x={x} y={y} fontSize={9} fill="#9A9A9A" textAnchor="middle" dominantBaseline="middle">
              {m[0]}
            </text>
          );
        })}
        <text x={cx} y={cy - 8} fontSize={11} fill="#767676" textAnchor="middle">
          Peak
        </text>
        <text x={cx} y={cy + 12} fontSize={18} fontWeight={700} fill="#00857C" textAnchor="middle">
          {MONTHS[peakIdx]}
        </text>
      </svg>
    </div>
  );
}
