import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { SavedCard } from "@/features/saved/SavedCard";
import { useSavedList, useSavedMutations } from "@/features/saved/useSaved";
import { useAppStore } from "@/store/useAppStore";

/** Lean landing: hero + entry CTAs + saved grid. (Value-props / how-it-works
 *  intentionally omitted — Marco wants the Dashboard lean.) */
export default function Dashboard() {
  const navigate = useNavigate();
  const setCopilotOpen = useAppStore((s) => s.setCopilotOpen);
  const saved = useSavedList();
  const { remove } = useSavedMutations();
  const records = saved.data ?? [];

  return (
    <div className="flex flex-col gap-8">
      {/* Hero */}
      <section className="rounded-hero border border-[#FBE2E8] bg-hero-gradient px-8 py-12">
        <p className="eyebrow">Decide with confidence</p>
        <h1 className="mt-3 max-w-2xl text-[38px] font-semibold leading-[1.1] tracking-[-0.02em]">
          Should you list it on Airbnb — or sell?
        </h1>
        <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-hof-secondary">
          A data-grounded recommendation for any property in Madrid, Barcelona,
          or Málaga — net income vs sale value, with clear uncertainty.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Button onClick={() => navigate("/new")}>Start new analysis</Button>
          <Button variant="secondary" onClick={() => setCopilotOpen(true)}>
            Open the co-pilot
          </Button>
        </div>
      </section>

      {/* Saved grid */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-[20px] font-semibold">Your saved analyses</h2>
          {records.length > 0 && (
            <button
              className="text-[13px] font-semibold text-kpmg-cobalt hover:underline"
              onClick={() => navigate("/saved")}
            >
              View portfolio →
            </button>
          )}
        </div>

        {saved.isLoading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <Card key={i} className="h-36">
                <div className="skeleton h-4 w-2/3" />
                <div className="skeleton mt-3 h-8 w-1/2" />
                <div className="skeleton mt-3 h-3 w-1/3" />
              </Card>
            ))}
          </div>
        ) : saved.isError ? (
          <Card className="grid min-h-[140px] place-items-center text-center">
            <p className="text-[14px] text-foggy">
              Couldn't load saved analyses.{" "}
              <button
                className="font-semibold text-kpmg-cobalt underline"
                onClick={() => saved.refetch()}
              >
                Retry
              </button>
            </p>
          </Card>
        ) : records.length === 0 ? (
          <Card className="grid min-h-[140px] place-items-center text-center">
            <div>
              <p className="text-[14px] text-hof-secondary">
                No saved analyses yet.
              </p>
              <p className="mt-1 text-[13px] text-foggy">
                Run an analysis and hit <strong>Save</strong> to build your
                portfolio.
              </p>
            </div>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {records.slice(0, 6).map((r) => (
              <SavedCard
                key={r.id}
                record={r}
                onDelete={(id) => remove.mutate(id)}
                deleting={remove.isPending && remove.variables === r.id}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
