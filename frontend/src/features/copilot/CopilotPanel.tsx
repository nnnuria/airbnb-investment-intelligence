import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  BarChart3,
  Scale,
  Send,
  Sparkles,
  Trash2,
  TrendingUp,
  Wrench,
  X,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { chat } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";

const SUGGESTED = [
  "Should I sell?",
  "Why this recommendation?",
  "What if I self-manage?",
];

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: string[];
  intent?: string;
  disclaimer?: string;
  error?: boolean;
}

const GREETING: Message = {
  role: "assistant",
  text: "Hi — ask me anything about this analysis. I cite the model and comparables behind each answer.",
};

/** Map the coordinator's intent class to a citation-chip icon. */
const INTENT_ICON: Record<string, LucideIcon> = {
  decision: BarChart3,
  comparables: TrendingUp,
  market: TrendingUp,
  regulatory: Scale,
  optimisation: Wrench,
};
const intentIcon = (intent?: string): LucideIcon =>
  (intent && INTENT_ICON[intent]) || Sparkles;

/**
 * Docked slide-over co-pilot, reachable from every screen. Wired to /chat:
 * keeps a local message history (the panel is mounted once in AppShell, so it
 * survives navigation), sends the active property + scenario for context, and
 * renders per-message source-citation chips. Honors the DESIGN.md slide-over
 * motion and layout.
 */
export function CopilotPanel() {
  const open = useAppStore((s) => s.copilotOpen);
  const setOpen = useAppStore((s) => s.setCopilotOpen);
  const prop = useAppStore((s) => s.activeProperty);
  const scen = useAppStore((s) => s.activeScenario);

  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const mutation = useMutation({
    mutationFn: (message: string) =>
      chat({
        message,
        property: (prop as Record<string, unknown> | null) ?? null,
        scenario: (scen as Record<string, unknown> | null) ?? null,
      }),
    onSuccess: (res) =>
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          sources: (res.sources as string[]) ?? [],
          intent: res.intent,
          disclaimer: res.disclaimer,
        },
      ]),
    onError: () =>
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "I couldn't reach the analysis service. Make sure the API is running, then try again.",
          error: true,
        },
      ]),
  });

  const send = (text: string) => {
    const q = text.trim();
    if (!q || mutation.isPending) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    mutation.mutate(q);
  };

  // Auto-scroll to the newest message / typing indicator.
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, mutation.isPending]);

  const subline = prop
    ? `Tracking ${prop.district} analysis`
    : "No active analysis yet";

  return (
    <>
      {/* Scrim */}
      <div
        onClick={() => setOpen(false)}
        className={cn(
          "fixed inset-0 z-40 bg-black/20 transition-opacity duration-300",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        aria-hidden
      />
      {/* Panel */}
      <aside
        className={cn(
          "fixed right-0 top-0 z-50 flex h-full w-[400px] max-w-[92vw] flex-col bg-surface shadow-lift",
          "transition-transform duration-[340ms] ease-slide-over",
          open ? "translate-x-0" : "translate-x-[105%]",
        )}
        aria-hidden={!open}
      >
        <header className="flex items-center gap-3 border-b border-hairline p-4">
          <span className="grid h-9 w-9 place-items-center rounded-full bg-avatar-gradient text-[13px] font-bold text-white">
            ✦
          </span>
          <div className="flex-1">
            <p className="text-[14px] font-semibold text-hof">Co-pilot</p>
            <p className="text-[12px] text-foggy">{subline}</p>
          </div>
          {messages.length > 1 && (
            <button
              onClick={() => setMessages([GREETING])}
              className="grid h-8 w-8 place-items-center rounded-full text-foggy hover:bg-track"
              aria-label="Clear conversation"
              title="Clear conversation"
            >
              <Trash2 size={16} />
            </button>
          )}
          <button
            onClick={() => setOpen(false)}
            className="grid h-8 w-8 place-items-center rounded-full text-foggy hover:bg-track"
            aria-label="Close co-pilot"
          >
            <X size={18} />
          </button>
        </header>

        {/* Message history */}
        <div ref={scrollRef} className="flex flex-1 flex-col gap-3 overflow-y-auto p-4">
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
          {mutation.isPending && <TypingBubble />}
        </div>

        {/* Composer */}
        <div className="border-t border-hairline p-4">
          <div className="mb-3 flex flex-wrap gap-2">
            {SUGGESTED.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                disabled={mutation.isPending}
                className="rounded-pill border border-kpmg-cobalt/30 bg-kpmg-cobalt/5 px-3 py-1 text-[12px] font-semibold text-kpmg-cobalt transition-colors hover:bg-kpmg-cobalt/10 disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2 rounded-pill border border-inputborder px-3 py-1.5">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") send(input);
              }}
              placeholder="Ask a question…"
              className="flex-1 bg-transparent text-[14px] outline-none placeholder:text-foggy"
            />
            <button
              onClick={() => send(input)}
              disabled={mutation.isPending || !input.trim()}
              className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-rausch text-white transition-opacity disabled:opacity-40"
              aria-label="Send"
            >
              <Send size={15} />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

// ── sub-components ────────────────────────────────────────────────────────────

function MessageBubble({ message: m }: { message: Message }) {
  const isUser = m.role === "user";
  const Icon = intentIcon(m.intent);

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-card p-3 text-[14px]",
          isUser
            ? "bg-hof text-white"
            : m.error
              ? "bg-rec-sell-bg text-hof-secondary"
              : "bg-[#F4F4F4] text-hof-secondary",
        )}
      >
        <p className="whitespace-pre-wrap leading-relaxed">{m.text}</p>

        {!isUser && m.sources && m.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {m.sources.map((s, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-pill bg-kpmg-cobalt/10 px-2 py-0.5 text-[11px] font-medium text-kpmg-cobalt"
              >
                <Icon size={11} />
                {s}
              </span>
            ))}
          </div>
        )}

        {!isUser && m.disclaimer && (
          <p className="mt-1.5 text-[11px] leading-snug text-foggy">{m.disclaimer}</p>
        )}
      </div>
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex justify-start">
      <div className="flex gap-1 rounded-card bg-[#F4F4F4] px-4 py-3">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-foggy"
            style={{ animationDelay: `${i * 120}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
