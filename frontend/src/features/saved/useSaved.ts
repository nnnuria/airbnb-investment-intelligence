/**
 * TanStack Query hooks over the /saved CRUD API. One shared cache key so the
 * Dashboard grid, the portfolio page, and the Decision "Save analysis" button
 * all stay in sync — every mutation invalidates the list.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { clearSaved, createSaved, deleteSaved, getSaved } from "@/lib/api";
import type { SavedCreateRequest } from "@/lib/types";

export const SAVED_KEY = ["saved"] as const;

export function useSavedList() {
  return useQuery({ queryKey: SAVED_KEY, queryFn: getSaved, retry: false });
}

export function useSavedMutations() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: SAVED_KEY });

  const create = useMutation({
    mutationFn: (payload: SavedCreateRequest) => createSaved(payload),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (id: string) => deleteSaved(id),
    onSuccess: invalidate,
  });
  const clear = useMutation({
    mutationFn: () => clearSaved(),
    onSuccess: invalidate,
  });

  return { create, remove, clear };
}
