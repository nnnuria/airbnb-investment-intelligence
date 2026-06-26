import { create } from "zustand";
import type { ScenarioRequest, ScenarioResponse } from "@/lib/types";

/**
 * Global UI + active-analysis state. Replaces Streamlit's st.session_state.
 * Surface/tab navigation is driven by the router; the bits here are the
 * cross-screen state the DESIGN.md spec calls out (managed toggle, co-pilot,
 * market/saved sub-views, and the active property + scenario).
 */

export type WorkspaceTab = "decision" | "market" | "optimise" | "regulatory";
export type ViewState = "ready" | "empty" | "loading" | "error";
export type MarketView = "cards" | "map" | "table";
export type SavedView = "list" | "compare";

interface AppState {
  // active analysis
  activeProperty: ScenarioRequest | null;
  activeScenario: ScenarioResponse | null;
  setActiveAnalysis: (
    property: ScenarioRequest | null,
    scenario: ScenarioResponse | null,
  ) => void;

  // workspace
  tab: WorkspaceTab;
  setTab: (tab: WorkspaceTab) => void;
  view: ViewState;
  setView: (view: ViewState) => void;
  managed: boolean;
  setManaged: (managed: boolean) => void;

  // sub-views
  marketView: MarketView;
  setMarketView: (v: MarketView) => void;
  savedView: SavedView;
  setSavedView: (v: SavedView) => void;

  // co-pilot
  copilotOpen: boolean;
  setCopilotOpen: (open: boolean) => void;
  toggleCopilot: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeProperty: null,
  activeScenario: null,
  setActiveAnalysis: (activeProperty, activeScenario) =>
    set({ activeProperty, activeScenario }),

  tab: "decision",
  setTab: (tab) => set({ tab }),
  view: "empty",
  setView: (view) => set({ view }),
  managed: true,
  setManaged: (managed) => set({ managed }),

  marketView: "cards",
  setMarketView: (marketView) => set({ marketView }),
  savedView: "list",
  setSavedView: (savedView) => set({ savedView }),

  copilotOpen: false,
  setCopilotOpen: (copilotOpen) => set({ copilotOpen }),
  toggleCopilot: () => set((s) => ({ copilotOpen: !s.copilotOpen })),
}));
