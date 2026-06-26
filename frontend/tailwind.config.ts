import type { Config } from "tailwindcss";

/**
 * Design tokens are lifted directly from docs/design/DESIGN.md (the Claude Design
 * handoff spec). Airbnb core palette is primary; KPMG cobalt/deep/purple are
 * data-viz / partner accents only.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ── Airbnb core (primary identity) ──
        rausch: { DEFAULT: "#FF385C", dark: "#E03150" },
        babu: { DEFAULT: "#00A699", dark: "#00857C" },
        arches: "#FC642D",
        hof: { DEFAULT: "#222222", secondary: "#484848" },
        foggy: "#767676",
        faint: "#9A9A9A",
        surface: "#FFFFFF",
        background: "#F7F7F7",
        hairline: "#EBEBEB",
        inputborder: "#E4E4E4",
        track: "#F2F2F2",
        // ── KPMG partner accent (data-viz / links / partner mark ONLY) ──
        kpmg: { cobalt: "#1E49E2", deep: "#00338D", purple: "#6E2BF5" },
        // ── recommendation tints ──
        "rec-list-bg": "#E6F7F4",
        "rec-sell-bg": "#FFF0F3",
        "rec-marginal-bg": "#FFF1EA",
      },
      fontFamily: {
        sans: [
          "Plus Jakarta Sans",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
      },
      borderRadius: {
        card: "16px",
        hero: "24px",
        input: "10px",
        pill: "999px",
      },
      boxShadow: {
        card: "0 4px 16px rgba(0,0,0,0.04)",
        lift: "0 12px 28px rgba(0,0,0,0.10)",
        cta: "0 8px 22px rgba(255,56,92,0.3)",
        teal: "0 8px 30px rgba(0,166,153,0.08)",
      },
      maxWidth: {
        page: "1240px",
        form: "920px",
      },
      backgroundImage: {
        "kpmg-gradient": "linear-gradient(90deg,#1E49E2,#6E2BF5)",
        "brand-logo": "linear-gradient(135deg,#FF385C,#FC642D)",
        "avatar-gradient": "linear-gradient(135deg,#6E2BF5,#1E49E2)",
        "hero-gradient": "linear-gradient(120deg,#FFF1F3,#FFF6F1,#F3F0FF)",
      },
      transitionTimingFunction: {
        "slide-over": "cubic-bezier(.4,0,.2,1)",
      },
      keyframes: {
        "pop-in": {
          "0%": { transform: "scale(.96)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        "tab-enter": {
          "0%": { transform: "translateY(8px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" },
        },
      },
      animation: {
        "pop-in": "pop-in .22s ease both",
        "tab-enter": "tab-enter .28s ease both",
        shimmer: "shimmer 1.4s linear infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
