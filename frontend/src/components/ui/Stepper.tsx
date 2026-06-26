import { Minus, Plus } from "lucide-react";

/** −  value  +  numeric stepper, per DESIGN.md (32px square buttons). */
export function Stepper({
  label,
  value,
  onChange,
  min = 0,
  max = 99,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  return (
    <div className="flex items-center justify-between">
      <span className="text-[14px] text-hof-secondary">{label}</span>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => onChange(clamp(value - step))}
          disabled={value <= min}
          className="grid h-8 w-8 place-items-center rounded-[9px] border border-inputborder text-hof disabled:opacity-40"
          aria-label={`Decrease ${label}`}
        >
          <Minus size={15} />
        </button>
        <span className="w-8 text-center text-[15px] font-semibold tabular-nums">
          {value}
        </span>
        <button
          type="button"
          onClick={() => onChange(clamp(value + step))}
          disabled={value >= max}
          className="grid h-8 w-8 place-items-center rounded-[9px] border border-inputborder text-hof disabled:opacity-40"
          aria-label={`Increase ${label}`}
        >
          <Plus size={15} />
        </button>
      </div>
    </div>
  );
}
