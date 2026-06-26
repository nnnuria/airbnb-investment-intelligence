/**
 * HTTP client for the FastAPI decision engine — the TS counterpart of
 * app/components/api_client.py. Same endpoints, same payload shapes.
 *
 * Config: VITE_API_BASE_URL (default http://127.0.0.1:8000).
 */
import axios, { AxiosError } from "axios";
import type {
  ChatRequest,
  ChatResponse,
  ComparablesResponse,
  HealthResponse,
  MarketReportResponse,
  OptimiseResponse,
  RegulatoryResponse,
  SavedCreateRequest,
  SavedRecord,
  ScenarioRequest,
  ScenarioResponse,
  SentimentResponse,
} from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// compute_scenario runs SHAP + two Monte-Carlo passes; give it room (45s, 5s connect).
const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45_000,
  headers: { "Content-Type": "application/json" },
});

/** A UI-ready error mirroring api_client.py's APIError messaging. */
export class APIError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "APIError";
    this.status = status;
  }
}

function toApiError(err: unknown, service: string): APIError {
  const ax = err as AxiosError;
  if (ax?.code === "ERR_NETWORK" || ax?.code === "ECONNABORTED") {
    return new APIError(
      "Can't reach the analysis API. Start it with `uvicorn api.main:app` and try again.",
    );
  }
  if (ax?.response) {
    return new APIError(
      `The ${service} returned an error (${ax.response.status}). Check the API logs.`,
      ax.response.status,
    );
  }
  return new APIError(`${service} request failed: ${ax?.message ?? String(err)}`);
}

export async function getScenario(
  payload: ScenarioRequest,
): Promise<ScenarioResponse> {
  try {
    const { data } = await client.post<ScenarioResponse>("/scenario", payload);
    return data;
  } catch (err) {
    throw toApiError(err, "analysis API");
  }
}

export async function chat(payload: ChatRequest): Promise<ChatResponse> {
  try {
    const { data } = await client.post<ChatResponse>("/chat", {
      use_llm: true,
      ...payload,
    });
    return data;
  } catch (err) {
    throw toApiError(err, "chat service");
  }
}

export async function marketReport(payload: {
  property?: Record<string, unknown> | null;
  scenario: Record<string, unknown>;
  use_llm?: boolean;
}): Promise<MarketReportResponse> {
  try {
    const { data } = await client.post<MarketReportResponse>("/market_report", {
      use_llm: true,
      ...payload,
    });
    return data;
  } catch (err) {
    throw toApiError(err, "market-report service");
  }
}

export async function optimise(
  payload: Record<string, unknown>,
): Promise<OptimiseResponse> {
  try {
    const { data } = await client.post<OptimiseResponse>("/optimise", payload);
    return data;
  } catch (err) {
    throw toApiError(err, "optimisation service");
  }
}

export async function comparables(
  payload: Record<string, unknown>,
): Promise<ComparablesResponse> {
  try {
    const { data } = await client.post<ComparablesResponse>(
      "/comparables",
      payload,
    );
    return data;
  } catch (err) {
    throw toApiError(err, "comparables service");
  }
}

export async function regulatoryRisk(payload: {
  city: string;
  neighbourhood?: string | null;
  question?: string | null;
}): Promise<RegulatoryResponse> {
  try {
    const { data } = await client.post<RegulatoryResponse>(
      "/regulatory_risk",
      payload,
    );
    return data;
  } catch (err) {
    throw toApiError(err, "regulatory service");
  }
}

// ── /saved ────────────────────────────────────────────────────────────────────

export async function getSaved(): Promise<SavedRecord[]> {
  try {
    const { data } = await client.get<SavedRecord[]>("/saved");
    return data;
  } catch (err) {
    throw toApiError(err, "saved-properties service");
  }
}

export async function createSaved(
  payload: SavedCreateRequest,
): Promise<SavedRecord> {
  try {
    const { data } = await client.post<SavedRecord>("/saved", payload);
    return data;
  } catch (err) {
    throw toApiError(err, "saved-properties service");
  }
}

export async function deleteSaved(
  id: string,
): Promise<{ ok: boolean; deleted: string }> {
  try {
    const { data } = await client.delete<{ ok: boolean; deleted: string }>(
      `/saved/${encodeURIComponent(id)}`,
    );
    return data;
  } catch (err) {
    throw toApiError(err, "saved-properties service");
  }
}

export async function clearSaved(): Promise<{ ok: boolean }> {
  try {
    const { data } = await client.delete<{ ok: boolean }>("/saved");
    return data;
  } catch (err) {
    throw toApiError(err, "saved-properties service");
  }
}

// ── /sentiment ────────────────────────────────────────────────────────────────

export async function getSentiment(city: string): Promise<SentimentResponse> {
  try {
    const { data } = await client.get<SentimentResponse>("/sentiment", {
      params: { city },
    });
    return data;
  } catch (err) {
    throw toApiError(err, "sentiment service");
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}

export async function apiHealthy(): Promise<boolean> {
  try {
    await client.get("/health", { timeout: 3_000 });
    return true;
  } catch {
    return false;
  }
}
