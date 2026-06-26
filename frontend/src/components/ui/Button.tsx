import { type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

type Variant = "primary" | "dark" | "secondary" | "ghost" | "purple-outline";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const VARIANTS: Record<Variant, string> = {
  primary: "bg-rausch text-white shadow-cta hover:bg-rausch-dark",
  dark: "bg-hof text-white hover:bg-black",
  secondary:
    "bg-surface text-hof border border-inputborder hover:border-foggy",
  ghost: "bg-transparent text-hof hover:bg-track",
  "purple-outline":
    "bg-surface text-kpmg-purple border border-kpmg-purple/40 hover:bg-kpmg-purple/5",
};

export function Button({ variant = "primary", className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-input px-5 py-2.5",
        "text-[14px] font-semibold transition-colors duration-[120ms]",
        "disabled:cursor-not-allowed disabled:opacity-50",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
}
