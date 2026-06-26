import type { Recommendation } from "./types";

/** Single source of truth for recommendation label / hex (SVG) / verdict verb. */
export const REC_META: Record<
  Recommendation,
  { label: string; hex: string; winsVerb: string }
> = {
  airbnb: { label: "List on Airbnb", hex: "#00A699", winsVerb: "Listing wins" },
  sell: { label: "Sell", hex: "#FF385C", winsVerb: "Selling wins" },
  marginal: { label: "Marginal", hex: "#FC642D", winsVerb: "Too close to call" },
};
