import { useCountUp } from "@/lib/useCountUp";

/** Animated radial confidence gauge — sweeps from empty to the confidence %. */
export function ConfidenceGauge({
  value,
  color,
  size = 118,
}: {
  value: number; // 0..1
  color: string; // hex
  size?: number;
}) {
  const animated = useCountUp(value, 900);
  const stroke = 11;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - animated);

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#E6F4F1"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[24px] font-bold tracking-[-0.02em] text-hof">
          {Math.round(animated * 100)}%
        </span>
        <span className="text-[11px] font-semibold text-foggy">confidence</span>
      </div>
    </div>
  );
}
