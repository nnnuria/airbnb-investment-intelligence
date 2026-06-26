import { useState } from "react";
import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import { ImageIcon, MapPin, Star } from "lucide-react";
import { Card, CardLabel } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Segmented";
import { cn } from "@/lib/cn";
import { eur, eurCompact } from "@/lib/format";
import {
  nearest,
  rating2,
  type Benchmark,
  type BenchmarkStat,
  type Comp,
} from "@/lib/market";
import { useAppStore } from "@/store/useAppStore";

/** How many of the nearest comps to surface in the listings display. */
const SHOW_N = 6;

export function Comparables({
  comps,
  benchmark,
}: {
  comps: Comp[];
  benchmark: Benchmark;
}) {
  const view = useAppStore((s) => s.marketView);
  const setView = useAppStore((s) => s.setMarketView);

  // Surface only the nearest few — a focused, scannable peer set.
  const shown = nearest(comps, SHOW_N);

  return (
    <Card>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-[15px] font-semibold text-hof">Comparable listings</p>
          <p className="text-[12.5px] text-foggy">
            {shown.length} nearest {shown.length === 1 ? "listing" : "listings"} by
            size, capacity & location
          </p>
        </div>
        <Segmented
          options={[
            { value: "cards", label: "Cards" },
            { value: "map", label: "Map" },
            { value: "table", label: "Table" },
          ]}
          value={view}
          onChange={setView}
        />
      </div>

      {shown.length === 0 ? (
        <div className="grid min-h-[160px] place-items-center text-center text-[14px] text-foggy">
          No comparable listings found for this property.
        </div>
      ) : view === "cards" ? (
        <CardsView comps={shown} />
      ) : view === "map" ? (
        <MapView comps={shown} />
      ) : (
        <TableView comps={shown} />
      )}

      <Benchmarks benchmark={benchmark} />
    </Card>
  );
}

// ── Cards ─────────────────────────────────────────────────────────────────────

function CardsView({ comps }: { comps: Comp[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {comps.map((c, i) => (
        <ListingCard key={c.id ?? i} comp={c} />
      ))}
    </div>
  );
}

function ListingCard({ comp }: { comp: Comp }) {
  const [loaded, setLoaded] = useState(false);
  const beds = comp.bedrooms ?? 0;

  return (
    <div className="group overflow-hidden rounded-card border border-hairline bg-surface shadow-card transition-[transform,box-shadow] duration-150 hover:-translate-y-[3px] hover:shadow-lift">
      {/* Photo with graceful placeholder behind it (DESIGN.md image handling) */}
      <div className="relative aspect-[4/3] w-full bg-gradient-to-br from-[#F2F2F2] to-[#E4E4E4]">
        <div className="absolute inset-0 grid place-items-center text-foggy/60">
          <ImageIcon size={28} />
        </div>
        {comp.picture_url && (
          <img
            src={comp.picture_url}
            alt={comp.name || "Listing"}
            loading="lazy"
            onLoad={() => setLoaded(true)}
            className={cn(
              "absolute inset-0 h-full w-full object-cover transition-opacity duration-300",
              loaded ? "opacity-100" : "opacity-0",
            )}
          />
        )}
        {comp.review_scores_rating != null && (
          <span className="absolute left-2.5 top-2.5 flex items-center gap-1 rounded-pill bg-surface/95 px-2 py-0.5 text-[12px] font-semibold text-hof shadow-card">
            <Star size={11} className="fill-hof text-hof" />
            {rating2(comp.review_scores_rating)}
          </span>
        )}
      </div>

      <div className="p-3.5">
        <p className="truncate text-[14px] font-semibold text-hof">
          {comp.name || "Comparable listing"}
        </p>
        <p className="mt-0.5 truncate text-[12.5px] text-foggy">
          {comp.neighbourhood_cleansed || comp.city || "—"} · {beds}{" "}
          {beds === 1 ? "bed" : "beds"}
        </p>
        <div className="mt-2.5 flex items-end justify-between">
          <span className="text-[15px] font-semibold text-hof">
            {comp.price != null ? `${eur(comp.price)}` : "—"}
            <span className="text-[12px] font-normal text-foggy">/night</span>
          </span>
          {comp.estimated_revenue_l365d != null && (
            <span className="text-[12.5px] text-babu">
              {eurCompact(comp.estimated_revenue_l365d)}/yr
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Map (real OpenStreetMap tiles) ───────────────────────────────────────────────

function pinColor(price: number | undefined): string {
  if (price == null) return "#6E8BF5";
  if (price >= 180) return "#00338D";
  if (price >= 150) return "#1E49E2";
  return "#6E8BF5";
}

function MapView({ comps }: { comps: Comp[] }) {
  const located = comps.filter(
    (c) => c.latitude != null && c.longitude != null,
  );

  if (located.length === 0) {
    return (
      <div className="grid min-h-[220px] place-items-center rounded-card bg-gradient-to-br from-[#EFF3FF] to-[#F4F1FF] px-6 text-center">
        <div>
          <MapPin size={26} className="mx-auto text-kpmg-cobalt" />
          <p className="mt-2 text-[14px] font-semibold text-hof">
            Map view activates with comp coordinates
          </p>
          <p className="mx-auto mt-1 max-w-sm text-[12.5px] text-foggy">
            These comps don't carry latitude/longitude. Use Cards or Table
            instead.
          </p>
        </div>
      </div>
    );
  }

  const lats = located.map((c) => c.latitude as number);
  const lngs = located.map((c) => c.longitude as number);
  const center: [number, number] = [
    (Math.min(...lats) + Math.max(...lats)) / 2,
    (Math.min(...lngs) + Math.max(...lngs)) / 2,
  ];
  // Two-corner bounds fit the points; a single point falls back to a fixed zoom.
  const bounds: [[number, number], [number, number]] = [
    [Math.min(...lats), Math.min(...lngs)],
    [Math.max(...lats), Math.max(...lngs)],
  ];
  const single = located.length === 1;

  return (
    <div>
      <div className="relative isolate h-[320px] overflow-hidden rounded-card border border-hairline">
        <MapContainer
          {...(single
            ? { center, zoom: 14 }
            : { bounds, boundsOptions: { padding: [36, 36] } })}
          scrollWheelZoom={false}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {located.map((c, i) => (
            <CircleMarker
              key={c.id ?? i}
              center={[c.latitude as number, c.longitude as number]}
              radius={9}
              pathOptions={{
                color: "#ffffff",
                weight: 2,
                fillColor: pinColor(c.price),
                fillOpacity: 1,
              }}
            >
              <Tooltip direction="top" offset={[0, -6]}>
                <span className="text-[12px] font-semibold text-hof">
                  {c.name ?? "Comparable"}
                </span>
                <span className="block text-[11.5px] text-foggy">
                  {c.price != null ? `${eur(c.price)}/night` : "—"}
                  {c.review_scores_rating != null
                    ? ` · ★ ${rating2(c.review_scores_rating)}`
                    : ""}
                </span>
              </Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-[11.5px] text-foggy">
        <Legend color="#00338D" label="€180+" />
        <Legend color="#1E49E2" label="€150–180" />
        <Legend color="#6E8BF5" label="< €150" />
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

// ── Table ──────────────────────────────────────────────────────────────────────

function TableView({ comps }: { comps: Comp[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-hairline text-foggy">
            <th className="py-2 pr-3 font-semibold">Name</th>
            <th className="py-2 pr-3 font-semibold">Neighbourhood</th>
            <th className="py-2 pr-3 text-right font-semibold">Beds</th>
            <th className="py-2 pr-3 text-right font-semibold">Nightly</th>
            <th className="py-2 pr-3 text-right font-semibold">Revenue/yr</th>
            <th className="py-2 text-right font-semibold">Rating</th>
          </tr>
        </thead>
        <tbody>
          {comps.map((c, i) => (
            <tr key={c.id ?? i} className="border-b border-hairline/70">
              <td className="max-w-[200px] truncate py-2 pr-3 text-hof">
                {c.name || "—"}
              </td>
              <td className="py-2 pr-3 text-hof-secondary">
                {c.neighbourhood_cleansed || "—"}
              </td>
              <td className="py-2 pr-3 text-right text-hof-secondary">
                {c.bedrooms ?? "—"}
              </td>
              <td className="py-2 pr-3 text-right text-hof">
                {c.price != null ? eur(c.price) : "—"}
              </td>
              <td className="py-2 pr-3 text-right text-hof-secondary">
                {c.estimated_revenue_l365d != null
                  ? eurCompact(c.estimated_revenue_l365d)
                  : "—"}
              </td>
              <td className="py-2 text-right text-hof-secondary">
                {c.review_scores_rating != null
                  ? `★ ${rating2(c.review_scores_rating)}`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Benchmark cards ─────────────────────────────────────────────────────────────

function Benchmarks({ benchmark }: { benchmark: Benchmark }) {
  const items: { label: string; stat?: BenchmarkStat; fmt: (n: number) => string }[] =
    [
      { label: "Median nightly", stat: benchmark.price, fmt: eur },
      {
        label: "Median annual revenue",
        stat: benchmark.estimated_revenue_l365d,
        fmt: eurCompact,
      },
      {
        label: "Median guest rating",
        stat: benchmark.review_scores_rating,
        fmt: (n) => `★ ${rating2(n)}`,
      },
    ];

  if (items.every((i) => !i.stat)) return null;

  return (
    <div className="mt-5 grid grid-cols-1 gap-3 border-t border-hairline pt-5 sm:grid-cols-3">
      {items.map(
        (i) =>
          i.stat && (
            <div key={i.label} className="rounded-card bg-track/50 px-4 py-3">
              <CardLabel className="mb-1">{i.label}</CardLabel>
              <p className="text-[20px] font-bold tracking-[-0.01em] text-hof">
                {i.fmt(i.stat.median)}
              </p>
              <p className="mt-0.5 text-[12px] text-foggy">
                P25 {i.fmt(i.stat.p25)} · P75 {i.fmt(i.stat.p75)}
              </p>
            </div>
          ),
      )}
    </div>
  );
}
