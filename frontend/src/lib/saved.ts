/**
 * Helpers for the /saved portfolio. A SavedRecord stores `property` and
 * `scenario` as opaque dicts (the backend persists them as-is); these coerce
 * them back to the typed shapes the UI renders and derive card summaries.
 */
import type {
  SavedRecord,
  ScenarioRequest,
  ScenarioResponse,
} from "./types";

export function recProperty(r: SavedRecord): ScenarioRequest {
  return r.property as unknown as ScenarioRequest;
}

export function recScenario(r: SavedRecord): ScenarioResponse {
  return r.scenario as unknown as ScenarioResponse;
}

/** Card title: explicit nickname, else "District, City". */
export function recTitle(r: SavedRecord): string {
  if (r.nickname && r.nickname.trim()) return r.nickname.trim();
  const p = recProperty(r);
  return [p.district, p.city].filter(Boolean).join(", ") || "Saved analysis";
}

/** Human "saved on" date, resilient to a missing/odd timestamp. */
export function recDate(r: SavedRecord): string {
  const d = new Date(r.saved_at);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
