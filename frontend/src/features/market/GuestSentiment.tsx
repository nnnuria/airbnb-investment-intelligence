import { Card } from "@/components/ui/Card";
import {
  ratingSentiment,
  ratingToScore,
  toPercentShares,
  type Comp,
} from "@/lib/market";
import type { SentimentResponse } from "@/lib/types";
import { SentimentDonut } from "./charts/SentimentDonut";

/**
 * Guest sentiment. When the dedicated /sentiment NLP service has data for the
 * city we show the real thing — review-text label split (donut), language mix,
 * representative quotes, and the per-listing mean-sentiment distribution.
 * Otherwise we degrade to a transparent rating-based proxy over the comps
 * (clearly labelled, so it's never mistaken for review-text sentiment).
 */
export function GuestSentiment({
  comps,
  sentiment,
}: {
  comps: Comp[];
  sentiment?: SentimentResponse | null;
}) {
  if (sentiment && Object.keys(sentiment.label_split).length > 0) {
    return <RealSentiment data={sentiment} />;
  }
  return <ProxySentiment comps={comps} />;
}

// ── real review-text sentiment ────────────────────────────────────────────────

const LANG_NAMES: Record<string, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  nl: "Dutch",
  ca: "Catalan",
};

function RealSentiment({ data }: { data: SentimentResponse }) {
  const labels = Object.fromEntries(
    toPercentShares(data.label_split).map((s) => [s.key.toLowerCase(), s.pct]),
  );
  const positive = labels.positive ?? 0;
  const neutral = labels.neutral ?? 0;
  const negative = labels.negative ?? 0;

  const langs = toPercentShares(data.lang_split).slice(0, 5);
  const maxBucket = Math.max(1, ...data.listing_histogram.map((b) => b.count));
  const posQuotes = data.examples.positive ?? [];
  const negQuotes = data.examples.negative ?? [];

  return (
    <Card>
      <p className="text-[15px] font-semibold text-hof">Guest sentiment</p>
      <p className="mb-4 text-[12.5px] text-foggy">
        Review-text analysis over {data.reviews_scored.toLocaleString("en-US")}{" "}
        scored reviews · {data.n_listings} listings
      </p>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Donut + legend */}
        <div className="flex items-center gap-5">
          <div className="w-[200px] shrink-0">
            <SentimentDonut
              positive={positive}
              neutral={neutral}
              negative={negative}
            />
          </div>
          <div className="flex flex-col gap-2 text-[13px]">
            <LegendRow color="#00A699" label="Positive" value={positive} />
            <LegendRow color="#FCD9A8" label="Neutral" value={neutral} />
            <LegendRow color="#FF385C" label="Negative" value={negative} />
          </div>
        </div>

        {/* Per-listing mean-sentiment histogram */}
        <div>
          <p className="mb-2 text-[12.5px] font-semibold text-hof">
            Per-listing mean sentiment
          </p>
          <div className="flex h-[120px] items-end gap-1">
            {data.listing_histogram.map((b) => {
              const edge = parseFloat(b.label);
              const height = 4 + (b.count / maxBucket) * 96;
              const color =
                edge >= 0.2 ? "#00A699" : edge < -0.2 ? "#FF385C" : "#FCD9A8";
              return (
                <div
                  key={b.label}
                  className="flex-1 rounded-t-sm transition-all"
                  style={{ height: `${height}%`, background: color }}
                  title={`${b.label} · ${b.count} listings`}
                />
              );
            })}
          </div>
          <div className="mt-1 flex justify-between text-[11px] text-foggy">
            <span>Negative</span>
            <span>Positive</span>
          </div>
        </div>
      </div>

      {/* Language split */}
      {langs.length > 0 && (
        <div className="mt-6">
          <p className="mb-2 text-[12.5px] font-semibold text-hof">
            Review languages
          </p>
          <div className="flex flex-col gap-1.5">
            {langs.map((l) => (
              <div key={l.key} className="flex items-center gap-3 text-[12.5px]">
                <span className="w-16 shrink-0 text-hof-secondary">
                  {LANG_NAMES[l.key.toLowerCase()] ?? l.key.toUpperCase()}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-track">
                  <div
                    className="h-full rounded-full bg-kpmg-cobalt"
                    style={{ width: `${l.pct}%` }}
                  />
                </div>
                <span className="w-9 shrink-0 text-right font-semibold text-hof">
                  {l.pct}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Representative quotes */}
      {(posQuotes.length > 0 || negQuotes.length > 0) && (
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
          {posQuotes.length > 0 && (
            <QuoteColumn
              title="What guests love"
              accent="text-babu-dark"
              quotes={posQuotes.slice(0, 2)}
            />
          )}
          {negQuotes.length > 0 && (
            <QuoteColumn
              title="Common complaints"
              accent="text-rausch"
              quotes={negQuotes.slice(0, 2)}
            />
          )}
        </div>
      )}

      {data.disclaimer && (
        <p className="mt-5 text-[11.5px] leading-snug text-foggy">
          {data.disclaimer}
        </p>
      )}
    </Card>
  );
}

function QuoteColumn({
  title,
  accent,
  quotes,
}: {
  title: string;
  accent: string;
  quotes: { text: string; score: number }[];
}) {
  return (
    <div>
      <p className={`mb-2 text-[12.5px] font-semibold ${accent}`}>{title}</p>
      <div className="flex flex-col gap-2">
        {quotes.map((q, i) => (
          <blockquote
            key={i}
            className="rounded-card border border-hairline bg-track/40 px-3 py-2 text-[12.5px] italic leading-snug text-hof-secondary"
          >
            “{q.text.length > 220 ? `${q.text.slice(0, 217)}…` : q.text}”
          </blockquote>
        ))}
      </div>
    </div>
  );
}

// ── rating-based proxy (fallback when /sentiment has no data) ──────────────────

function ProxySentiment({ comps }: { comps: Comp[] }) {
  const s = ratingSentiment(comps);
  const rated = comps.filter((c) => typeof c.review_scores_rating === "number");

  if (s.n === 0) return null;

  return (
    <Card>
      <p className="text-[15px] font-semibold text-hof">Guest sentiment</p>
      <p className="mb-4 text-[12.5px] text-foggy">
        Rating-based proxy across {s.n} comparable{" "}
        {s.n === 1 ? "listing" : "listings"}
      </p>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Donut + legend */}
        <div className="flex items-center gap-5">
          <div className="w-[200px] shrink-0">
            <SentimentDonut
              positive={s.positive}
              neutral={s.neutral}
              negative={s.negative}
            />
          </div>
          <div className="flex flex-col gap-2 text-[13px]">
            <LegendRow color="#00A699" label="Positive" value={s.positive} />
            <LegendRow color="#FCD9A8" label="Neutral" value={s.neutral} />
            <LegendRow color="#FF385C" label="Negative" value={s.negative} />
          </div>
        </div>

        {/* Per-listing mean sentiment distribution */}
        <div>
          <p className="mb-2 text-[12.5px] font-semibold text-hof">
            Per-listing mean sentiment
          </p>
          <div className="flex h-[120px] items-end gap-1">
            {rated.map((c, i) => {
              const score = ratingToScore(c.review_scores_rating as number);
              const height = 12 + ((score + 1) / 2) * 88; // 12–100%
              const color =
                score >= 0.3 ? "#00A699" : score <= -0.3 ? "#FF385C" : "#FCD9A8";
              return (
                <div
                  key={c.id ?? i}
                  className="flex-1 rounded-t-sm transition-all"
                  style={{ height: `${height}%`, background: color }}
                  title={`${c.name ?? "Comp"} · ★ ${(c.review_scores_rating as number).toFixed(2)}`}
                />
              );
            })}
          </div>
          <div className="mt-1 flex justify-between text-[11px] text-foggy">
            <span>Negative</span>
            <span>Positive</span>
          </div>
        </div>
      </div>

      {/* Review-text features pending the NLP service */}
      <div className="mt-5 rounded-card border border-dashed border-hairline bg-track/40 px-4 py-3">
        <p className="text-[12.5px] text-foggy">
          <span className="font-semibold text-hof-secondary">
            Review-text analysis
          </span>{" "}
          — representative quotes and the review-language split activate once the
          sentiment service (<code className="text-[11.5px]">/sentiment</code>)
          has data for this city.
        </p>
      </div>
    </Card>
  );
}

function LegendRow({
  color,
  label,
  value,
}: {
  color: string;
  label: string;
  value: number;
}) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-3 w-3 rounded-sm" style={{ background: color }} />
      <span className="text-hof-secondary">{label}</span>
      <span className="font-semibold text-hof">{value}%</span>
    </span>
  );
}
