# Airbnb Investment Intelligence — Frontend

React SPA for the Airbnb-vs-sell decisioning product. Replaces the Streamlit
`app/`. Talks to the existing FastAPI backend over HTTP (no business logic
lives here). Design spec: [`../docs/design/DESIGN.md`](../docs/design/DESIGN.md).

## Stack
Vite · React 18 · TypeScript · Tailwind CSS · TanStack Query · React Router ·
Zustand · Recharts · Axios. Design tokens (Airbnb palette + KPMG accents,
Plus Jakarta Sans) live in `tailwind.config.ts`.

## Getting started
```bash
cd frontend
npm install
cp .env.example .env        # adjust VITE_API_BASE_URL if the API isn't on :8000
npm run dev                 # http://localhost:5173
```
Run the backend alongside it:
```bash
# from repo root
uvicorn api.main:app --reload   # http://127.0.0.1:8000
```

## Scripts
- `npm run dev` — dev server (HMR)
- `npm run build` — typecheck + production build to `dist/`
- `npm run preview` — serve the production build
- `npm run typecheck` — types only

## Layout
```
src/
  lib/         api.ts (mirrors app/components/api_client.py), types.ts (Pydantic mirror), cn, queryClient
  store/       useAppStore.ts (Zustand — replaces st.session_state)
  components/  layout/ (AppBar, Footer, AppShell) · ui/ (Card, Button, RecommendationPill, Segmented)
  features/    copilot/ (docked slide-over)
  pages/       Dashboard, NewAnalysis, Workspace (Decision/Market/Optimise/Regulatory tabs), Saved
```

## Status
Scaffold: app shell, routing, theme, API client + types, store, and placeholder
screens. Next: Decision tab (the proof page), then Market / Optimise / Regulatory,
the input form, Saved (needs `/saved` API), and the co-pilot wiring.
