import { cn } from "@/lib/cn";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

/** Pill-style segmented control (Cards/Map/Table, List/Compare, Managed/Self…). */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  className,
}: {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "inline-flex gap-1 rounded-pill bg-track p-1",
        className,
      )}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={cn(
            "rounded-pill px-3.5 py-1.5 text-[13px] font-semibold transition-colors",
            value === opt.value
              ? "bg-hof text-white"
              : "text-foggy hover:text-hof",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
