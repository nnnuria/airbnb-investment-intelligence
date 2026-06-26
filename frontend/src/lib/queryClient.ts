import { QueryClient } from "@tanstack/react-query";

/**
 * Shared TanStack Query client. Scenario/market/optimise calls are expensive
 * (SHAP + Monte-Carlo on the backend), so cache generously and don't refetch
 * on window focus.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      gcTime: 30 * 60 * 1000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});
