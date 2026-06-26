import { useEffect, useRef, useState } from "react";

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

/** Animate a number from 0 → target on mount (count-up). Skips on reduced-motion. */
export function useCountUp(target: number, durationMs = 850): number {
  const [value, setValue] = useState(prefersReducedMotion() ? target : 0);
  const raf = useRef<number>();

  useEffect(() => {
    if (prefersReducedMotion()) {
      setValue(target);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      setValue(target * easeOut(t));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [target, durationMs]);

  return value;
}
