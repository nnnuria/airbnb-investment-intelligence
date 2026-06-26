import { type HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** KPMG-blue / teal featured accent border + lift. */
  featured?: boolean;
  /** Adds hover-lift interaction (translateY + shadow). */
  hover?: boolean;
}

export function Card({ featured, hover, className, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-card border border-hairline bg-surface p-5 shadow-card",
        featured && "border-2 border-babu shadow-teal",
        hover &&
          "transition-[transform,box-shadow] duration-[180ms] ease-out hover:-translate-y-[3px] hover:shadow-lift",
        className,
      )}
      {...props}
    />
  );
}

export function CardLabel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "mb-2 text-[11.5px] font-semibold uppercase tracking-[0.04em] text-foggy",
        className,
      )}
      {...props}
    />
  );
}

export function CardValue({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "text-[28px] font-bold leading-none tracking-[-0.02em] text-hof",
        className,
      )}
      {...props}
    />
  );
}

export function CardDelta({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("mt-2 text-[13px] leading-snug text-foggy", className)}
      {...props}
    />
  );
}
