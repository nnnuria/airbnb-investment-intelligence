import {
  type InputHTMLAttributes,
  type SelectHTMLAttributes,
  type ReactNode,
} from "react";
import { cn } from "@/lib/cn";

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[12.5px] font-semibold text-hof-secondary">
        {label}
      </span>
      {children}
      {hint && <span className="text-[11.5px] text-foggy">{hint}</span>}
    </label>
  );
}

const FIELD_CLS =
  "w-full rounded-input border border-inputborder bg-surface px-3 py-2 text-[14px] text-hof outline-none focus:border-foggy";

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn(FIELD_CLS, props.className)} />;
}

export function Select({
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...props} className={cn(FIELD_CLS, "appearance-none", props.className)}>
      {children}
    </select>
  );
}

/** Toggleable amenity / bundle chip. */
export function Chip({
  label,
  selected,
  onClick,
  variant = "amenity",
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
  variant?: "amenity" | "bundle";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "whitespace-nowrap rounded-pill border px-3 py-1.5 text-[13px] font-semibold transition-colors",
        variant === "bundle"
          ? selected
            ? "border-hof bg-hof text-white"
            : "border-inputborder bg-surface text-hof hover:border-foggy"
          : selected
            ? "border-rausch bg-rec-sell-bg text-rausch"
            : "border-inputborder bg-surface text-hof-secondary hover:border-foggy",
      )}
    >
      {label}
    </button>
  );
}
