import { NavLink, useNavigate } from "react-router-dom";
import { Plus, MessageCircle } from "lucide-react";
import { cn } from "@/lib/cn";
import { useAppStore } from "@/store/useAppStore";

function NavPill({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          "rounded-pill px-3.5 py-1.5 text-[14px] font-semibold transition-colors",
          isActive ? "bg-track text-hof" : "text-foggy hover:text-hof",
        )
      }
    >
      {label}
    </NavLink>
  );
}

export function AppBar() {
  const navigate = useNavigate();
  const toggleCopilot = useAppStore((s) => s.toggleCopilot);
  const activeProperty = useAppStore((s) => s.activeProperty);

  const selectorLabel = activeProperty
    ? `${activeProperty.district}, ${cap(activeProperty.city)}`
    : "No active analysis";

  return (
    <header className="sticky top-0 z-40 border-b border-hairline bg-surface/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-page items-center gap-4 px-7">
        {/* Brand */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2.5"
          aria-label="Go to dashboard"
        >
          <span className="h-8 w-8 rounded-full bg-brand-logo" />
          <span className="hidden flex-col items-start leading-tight sm:flex">
            <span className="text-[14px] font-bold text-hof">
              Airbnb Investment Intelligence
            </span>
            <span className="text-[11px] text-foggy">
              List-or-sell decisioning
            </span>
          </span>
        </button>

        {/* Primary nav */}
        <nav className="ml-2 flex items-center gap-1">
          <NavPill to="/" label="Dashboard" />
          <NavPill to="/saved" label="Saved" />
        </nav>

        {/* Property selector */}
        <button
          onClick={() => navigate("/workspace")}
          className="mx-auto flex items-center gap-3 rounded-pill border border-hairline bg-surface px-4 py-1.5 shadow-card"
        >
          <span className="text-[13px] font-semibold text-hof">
            {selectorLabel}
          </span>
          <span
            className="grid h-6 w-6 place-items-center rounded-full bg-rausch text-white"
            aria-hidden
          >
            <Plus size={14} />
          </span>
        </button>

        {/* Co-pilot + avatar */}
        <div className="flex items-center gap-3">
          <button
            onClick={toggleCopilot}
            className="flex items-center gap-2 rounded-pill border border-hairline bg-surface px-3.5 py-1.5 text-[13px] font-semibold text-hof hover:border-foggy"
          >
            <span className="h-2 w-2 rounded-full bg-babu" />
            <MessageCircle size={15} />
            <span className="hidden md:inline">Co-pilot</span>
          </button>
          <span className="grid h-8 w-8 place-items-center rounded-full bg-avatar-gradient text-[12px] font-bold text-white">
            M
          </span>
        </div>
      </div>
    </header>
  );
}

function cap(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
