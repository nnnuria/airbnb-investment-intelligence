<!--
PROVENANCE
This is the design specification for the React frontend, authored by Claude Design
and exported in its handoff bundle on 2026-06-26.

The full handoff bundle — the interactive `Analysis Workspace.dc.html` prototype and
the `screenshots/` reference images — is intentionally NOT committed to git (it's
~95% minified/binary generated output that goes stale once the React app exists). It
lives outside the repo at:
    ../design_handoff_airbnb_investment_intelligence/   (sibling of this repo)
Open that .dc.html in a browser to interact with the prototype; see the screenshots
there for rendered references. This DESIGN.md is the canonical text spec kept in-repo.

Canonical living reference: the Claude Design shareable project link.
-->

# Handoff: Airbnb Investment Intelligence

## Overview
A decision tool for property investors that answers one question per property: **should the owner list it on Airbnb, or sell it?** The product has three surfaces — a **Dashboard** (landing), an **Analysis Workspace** (the core, four tabs), and a docked **Co-pilot** chat — plus a **New-analysis input form** and a **Saved-analyses** area (list + compare). The design is consumer-friendly and warm, modelled closely on Airbnb's own product language, with a discreet KPMG partner accent used only for data-viz and partner marks.

---

## About the Design Files
The prototype file in the handoff bundle — `Analysis Workspace.dc.html` (kept outside the repo, see provenance note above) — is a **design reference created in HTML**. It is a working prototype that demonstrates the intended look, layout, copy, and interaction behavior. **It is not production code to copy directly.**

> Note on format: the file is a "Design Component" — its markup uses a lightweight templating syntax (`{{ ... }}` holes, `<sc-if>`, `<sc-for>`, `<dc-import>`) and a `class Component` logic block, loaded via `support.js`. **Do not port this runtime.** Read it as a spec: the template shows structure/styling, and the `class Component` block shows state, data, and the financial model. Open it in a browser to interact with it.

The task is to **recreate this design in the target codebase** using its established component library, styling system, routing, and state patterns. For this project that target is **React + TypeScript + Tailwind + shadcn/ui + Recharts + TanStack Query + React Router + Zustand**, in a new `frontend/` directory, wired to the existing FastAPI backend.

This sits in front of an existing backend (a **LightGBM nightly-price model**, a calendar-trained LightGBM occupancy model, a Monte-Carlo revenue simulator, a comparables/market service, a regulatory-scoring service, and a governance check). Wire the UI to those real services; all numbers in the prototype are representative sample data.

---

## Fidelity
**High-fidelity (hifi).** Final colors, typography, spacing, rounding, shadows, and interactions are all intentional and specified below. Recreate the UI faithfully using the codebase's existing primitives. Where the codebase already has equivalent components (cards, pills, toggles, tabs, sliders), prefer those, matched to the tokens below.

---

## Design Tokens

### Colors — Airbnb core (primary identity)
| Token | Hex | Use |
|---|---|---|
| Rausch (primary) | `#FF385C` | Primary CTAs, active states, key highlights, "Sell" recommendation, primary map pin |
| Rausch dark (hover) | `#E03150` | CTA hover (approx.) |
| Babu (teal) | `#00A699` | Positive / "List on Airbnb" recommendation, success, positive deltas |
| Babu dark | `#00857C` | Teal text on light backgrounds |
| Arches | `#FC642D` | Warnings, "Marginal" recommendation, medium-risk |
| Hof (text) | `#222222` | Primary text / near-black UI |
| Hof secondary | `#484848` | Body text |
| Foggy (muted) | `#767676` | Captions, labels, muted text |
| Faint label | `#9A9A9A` | Footnotes, axis labels |
| Surface | `#FFFFFF` | Cards |
| Background | `#F7F7F7` | App background |
| Hairline border | `#EBEBEB` | Card borders, dividers |
| Input border | `#E4E4E4` | Inputs, segmented controls |
| Track / fill grey | `#F2F2F2` | Bar-chart tracks, segmented control bg |

### Colors — KPMG partner accent (sparingly: data-viz, links, partner mark ONLY — never a primary surface)
| Token | Hex | Use |
|---|---|---|
| Cobalt blue | `#1E49E2` | Chart series, links, selected/hover, predicted-nightly accent |
| Deep blue | `#00338D` | "KPMG" mark, darkest chart series, high-price map pins |
| Purple | `#6E2BF5` | Secondary chart series, radar fill, sentiment/positioning gradients |
| Blue→purple gradient | `linear-gradient(90deg,#1E49E2,#6E2BF5)` | Feature-driver bars, nightly-rate driver bars |

### Recommendation tinted backgrounds (pills/cards)
- List/positive: bg `#E6F7F4`, text `#00A699` (or `#00857C`)
- Sell/negative: bg `#FFF0F3`, text `#FF385C`
- Marginal/warning: bg `#FFF1EA`, text `#FC642D` / `#C25719`

### Typography
- **Family:** `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif` (Google Fonts; weights 400/500/600/700/800). This is the Airbnb-Cereal-alike. If the codebase has Airbnb Cereal licensed, use it instead.
- **Modest weights** — headlines are 500–600, not heavy 700+ (except numeric KPI values, which are 700).
- Scale used:
  - Page H1: 27px / 600 / letter-spacing −0.01em
  - Dashboard hero H1: 38px / 600 / −0.02em
  - Section title: 13px / 700
  - KPI value: 28–30px / 700 / −0.02em
  - NPV-advantage centrepiece: 40px / 700 / −0.02em
  - Body: 14–15px / 500 / line-height ~1.5
  - Caption / label: 11–12.5px / 600 / color Foggy
  - Eyebrow (uppercase): 12px / 600–700 / letter-spacing 0.04–0.05em / uppercase

### Spacing, radius, shadow
- Card padding: 20–24px; section gaps: 16–20px; page max-width 1240px (input form 920px), 28px horizontal gutter.
- **Radius:** cards 16px (large cards/hero 18–24px); buttons 10–12px; pills & chips fully rounded (`999px`); icon buttons circular; inputs 10px; small tags 999px. **No hard corners on interactive elements.**
- **Shadows (soft):** card `0 4px 16px rgba(0,0,0,0.04)`; featured/hover-lift `0 12px 28px rgba(0,0,0,0.08–0.10)`; CTA `0 8px 22px rgba(255,56,92,0.3)`; teal feature `0 8px 30px rgba(0,166,153,0.08)`.
- Hover-lift pattern: `transform: translateY(-3px)` + larger shadow, `transition: transform .18s ease, box-shadow .18s ease`.

---

## Global Layout & Chrome

**App bar** (sticky, `rgba(255,255,255,0.92)` + backdrop-blur, bottom hairline):
- Left: circular gradient logo (`135deg, #FF385C → #FC642D`) + product name "Airbnb Investment Intelligence" / subtitle "List-or-sell decisioning". Clicking the logo → Dashboard.
- Nav pills: **Dashboard**, **Saved** (active pill = `#F2F2F2` bg / `#222` text).
- Center: a pill-shaped **property selector** (white, hairline border, soft shadow) showing "Eixample, Barcelona | 2-bed apartment" + a circular `#FF385C` "+" button. Clicking → Workspace.
- Right: **Co-pilot** button (white pill, teal status dot) that opens the slide-over; circular user avatar (gradient `135deg,#6E2BF5→#1E49E2`, initials).

**Footer** (every screen): left — disclaimer "Estimates only — not investment, tax, or legal advice. Figures are modelled from market data and the assumptions you provide, which you can edit at any time." Right — "Modelling powered by **KPMG**" (KPMG in `#00338D`, weight 800).

**Dev affordance to REMOVE in production:** a fixed bottom-center pill with "Go" (Dashboard/Input/Workspace/Saved) and "State" (Ready/Empty/Loading/Error) switchers. This is a review aid only — do not ship it; replace with real routing and real data/loading states.

---

## Screens / Views

### 1. Dashboard (landing)
- **Purpose:** entry point; start a new analysis, open the co-pilot, glance at the in-progress analysis, and browse saved analyses.
- **Layout:** centered column, max-width 1240px.
  - **Hero card** (radius 24, gradient `120deg,#FFF1F3→#FFF6F1→#F3F0FF`, border `#FBE2E8`): eyebrow "DECIDE WITH CONFIDENCE" (Rausch), H1 "Should you list it on Airbnb — or sell?" (38px), subcopy, two buttons — primary **Start new analysis** (Rausch → Input form) and secondary **Open the co-pilot** (white).
  - **Three value-prop cards** (3-col grid): **Decide** (teal check icon), **Optimise** (purple trend icon), **Converse** (blue chat icon), each with title + one-line description.
  - **How it works** (4-col grid): numbered steps 1) Enter property details 2) We run the models 3) Review the recommendation 4) Save & revisit.
  - **Session summary row** (2fr/1fr grid): left — teal-tinted "In progress" card (district·city, recommendation pill, Net annual profit, Break-even, Confidence; clicking → Workspace). Right — dark `#222` card showing saved-count "6" + "View all →".
  - **Saved grid** (3-col): saved-analysis cards (see Saved data shape) with hover-lift; click → Workspace.

### 2. New-analysis Input form
- **Purpose:** capture property + financial inputs; "Run analysis" kicks off modelling.
- **Layout:** centered column, max-width 920px. Eyebrow "NEW ANALYSIS" + H1 "Tell us about the property". Sections are white cards (radius 18).
  - **Location:** City (select), District (select), Nickname (optional text).
  - **Property:** Size m² (text), Listing type (select). Then three **stepper** controls — Bedrooms, Bathrooms, Accommodates — each `−  value  +` with 32px square `−/+` buttons (radius 9, hairline border). Bedrooms default 2, Bathrooms 1, Accommodates 4; floors at 0 (Accommodates at 1).
  - **Amenities:** subtitle "Start from a bundle, then fine-tune." A row of **bundle preset** pills — Essentials, Remote Work Ready, Premium Luxury, Family Friendly, Urban Comfort — selecting one is dark `#222`/white and **adds that bundle's amenities** to the selection. Then 5 **chip groups** (Comfort & Building, Outdoor & Views, Work & Smart Stays, Appliances, Family & Pets); each chip toggles selected (selected = bg `#FFF0F3`, text `#FF385C`, border `#FF385C`; unselected = white, border `#E4E4E4`). Chips are `white-space: nowrap`. Manually toggling a chip clears the active bundle highlight. Free-text "Additional amenities" input at the end.
  - **Financial assumptions:** 3-col grid of inputs — Cadastral value (€), IBI override (€/yr, placeholder "auto"), Airbnb setup cost (€), Purchase price for CGT (€), Annual rent growth (%), Property appreciation (%). Then a top-bordered row with **two controls that drive the model** (see *Financial model* below):
    - **Investment horizon** — range **slider**, min 5, max 30, step 1, default **10**, `accent-color:#FF385C`; the chosen value shows as "{n} yrs" in Rausch next to the label, with "5 yrs / 30 yrs" end labels.
    - **Discount rate** — number input, min 3, max 20, step 0.5, default **7**, suffixed "%", helper "Between 3% and 20%".
  - Two grey toggle rows: **Include income tax in net profit** (pill switch, default ON) and **Professionally managed** vs **Self** (segmented control, default Managed).
  - Buttons: **Run analysis** (Rausch) → Workspace, showing the **loading** state for ~1.6s, then **ready**; **Cancel** → Dashboard.

### 3. Analysis Workspace
- **Header:** eyebrow "ANALYSIS WORKSPACE" + "Updated 2h ago" tag; H1 property name; sub-line "Carrer de Mallorca, Barcelona · 68 m² · 2 bed · 1 bath · sleeps 4"; right-aligned **Share** (white) + **Export report** (`#222`) buttons.
- **Tabs** (underline style, active = `#222` text + 2px bottom border; inactive = Foggy): **Decision · Market · Optimise · Regulatory**.
- The body swaps by **state**: ready (tab content), empty ("No active analysis" + Start CTA), loading (spinner + progress checklist + shimmer skeletons), error ("We couldn't finish this analysis" + Retry/Contact, error ref).

#### 3a. Decision tab (hero tab)
Order, top to bottom:
1. **Two conditional banners** (side by side): **Regulatory risk** (here Medium — amber `#FFF6EC`/`#F6E2C6`, shield icon, reasoning line) and **Governance review** (here passed — blue `#EFF3FF`/`#D6DEFB`, "No policy violations · no human review required"). Banner severity/color is data-driven: HIGH/MEDIUM/LOW/UNKNOWN for regulatory; passed vs human-review-required/violations for governance.
2. **Hero recommendation** (teal-tinted card): left — an **animated radial confidence gauge** (118px; track `#E6F4F1`, progress in recommendation color, sweeps from empty to the confidence % on mount; center shows the % via a count-up + "% confidence"). Middle — recommendation **pill** ("List on Airbnb" teal / "Sell" rausch / "Marginal" arches; white check icon; subtle pop-in on mount) + one-line summary. Right — **Management model** segmented control (Managed / Self-managed) with a contextual note; toggling recomputes results.
3. **Actions row:** **Save analysis** (`#222`), **Ask a follow-up** (opens co-pilot), **See improvement ideas →** (purple-outlined; switches to Optimise).
4. **5 KPI cards** (auto-fit grid, min 196px): Gross annual revenue (€48,200), **Net annual profit** (featured — 2px teal border, larger, with a P10–P90 band bar and "P10 €19k / P90 €34k"), Indicative sale value (€410,000), Payback period (2.4 yrs), Projected occupancy (72%). Numeric values use a **count-up** animation on mount.
5. **Investment comparison ({horizon}-year horizon)** card: title + right-aligned "IRR 14.2% · Airbnb · {rate}% discount". Two columns:
   - Left: four labelled bars — Keep & list NPV after-tax, Keep & list NPV pre-tax, Sell net after CGT, Sell gross pre-tax (each value + an animated grow-in bar).
   - Right (**centrepiece**): "NPV advantage · Airbnb − Sell" with a large signed figure — **teal `+€…` when listing wins, red `−€…` when selling wins** — a verdict line ("Listing wins over a {n}-year hold" / "Selling wins over a {n}-year hold"), a proportional Airbnb-vs-Sell bar, and the two underlying values.
   - Caption: "Assumes IBI €640/yr, basuras €120/yr, €6,500 setup, {rate}% discount rate, {horizon}-year holding. After-tax figures apply your marginal income-tax rate and Spanish CGT on sale."
6. **Charts grid (2-col):**
   - **Net-profit distribution** — shaded distribution curve (KPMG blue) with dashed P10/P50/P90 markers.
   - **Seasonality of demand** — radial 12-month heat ring (teal scale: high `#00A699` / mid `#7FD4CC` / low `#D7EFEC`); each month segment has a hover tooltip; center labels the peak month.
   - **Cost waterfall** — gross revenue (teal) → each cost step (rausch/arches shades) → net profit (blue), with connector lines.
   - **Why this recommendation** — signed feature-driver bars from a center axis (positive = blue→purple gradient to the right; negative = grey to the left).

#### 3b. Market tab
- Top row: recommendation pill + "P(Airbnb beats Sell) 87%"; right — **Regenerate** button (refetch comparables/model).
- **4 market KPI cards:** **Predicted nightly €165** ("LightGBM price model", blue accent), Median nightly €158 (+4% vs market), Median annual revenue €41,700 (top quartile), Median rating 4.88 (84 comps).
- **Analyst brief** — purple-tinted card with a "Strong & stable" headline + a narrative paragraph.
- **Positioning + drivers (2-col):**
  - **Market positioning** — three diverging bars from center (Nightly vs market +4% teal, Annual revenue vs market +16% teal, Peer guest rating blue) + a **positioning scatter** (peers plotted nightly-price x vs annual-revenue y as blue dots; the subject property is a larger Rausch dot labelled "You"; hover tooltips).
  - **What drives the nightly rate** — horizontal driver bars (Location 38%, Bedrooms 22%, Amenities 18%, Seasonal demand 14%, Reviews 8%; blue→purple gradient) + a **peer guest-rating radar** (axes Cleanliness/Location/Value/Accuracy; subject polygon in purple, peer average in grey).
- **Comparable listings** with a **Cards / Map / Table** segmented toggle:
  - **Cards** (3-col): Airbnb-style listing cards — photo, rating badge "★ 4.95", name, "{hood} · {beds} bed", "€/night" + "€/yr revenue". **Image handling:** each card has a gradient + photo-icon **placeholder** behind the image; the photo is loaded via JS after mount and only revealed on successful load (`onload`), and hidden on error (`onerror`) so dead URLs degrade gracefully to the placeholder. Hover-lift on cards.
  - **Map:** a stylised map panel with **price pins** colored by nightly (`#00338D` €180+, `#1E49E2` €150–180, `#6E8BF5` €120–150), the **subject property** highlighted distinctly in Rausch with a pulsing ring; pins **animate in** (pop). Legend on the side. *(In production use the real map library + lat/long; pins sized/colored by price or revenue.)*
  - **Table:** columns Name, Neighbourhood, Beds, Nightly, Revenue/yr, Rating.
  - Below the toggle: **benchmark cards** — Median nightly €158 (P25 €138 · P75 €182), Median revenue €41,700 (P25 €34k · P75 €49k), Median rating 4.88 (P25 4.79 · P75 4.94).
- **Guest sentiment:** a **donut** (Positive 78% teal / Neutral 14% `#FCD9A8` / Negative 8% rausch) + legend; two example review quotes (positive teal-tinted with `+0.92` score, negative rausch-tinted with `−0.41`); a **per-listing mean-sentiment** mini bar distribution (negative→positive); a **review language split** stacked bar (English 54 / Spanish 22 / French 12 / German 7 / Other 5) with legend.

#### 3c. Optimise tab — INTERACTIVE planner (important)
- **Peer-gap summary (4 cards):** Peer median revenue €41,700, Target revenue €59,400 (teal), Gap to top quartile €11,200, Projected annual nights 263.
- **Improvement list** — each row (white card; **teal border when included**): an **include/exclude pill toggle**, name + a **confidence badge** (high teal / medium arches / low grey), rationale, "Annual uplift +€X" (teal), an **editable "Your cost (€)" number input**, and a computed **Payback** ("{cost/uplift×12} mo").
- **Live selection summary** (sticky dark `#222` bar): "Your selection — **{N} of {M}** improvements" + totals that recompute instantly: **Total annual uplift** (teal), **Total investment**, **Blended payback** (total cost ÷ total uplift × 12, shown in months, or years if ≥12). This must be a hands-on planner, not a static ranked list.

#### 3d. Regulatory tab
- **Risk badge** card (HIGH/MEDIUM/LOW/UNKNOWN — here Medium/arches, shield icon) + reasoning paragraph.
- "What drives the rating" — bulleted points (teal dot = positive, arches dot = risk).
- **Source links** as pill chips (blue): "Ajuntament de Barcelona · PEUAT ↗", "Generalitat · Tourism registry ↗".
- **"Not legal advice" note** — amber `#FFFBF0`/`#F4E6C4` callout.

### 4. Co-pilot (docked slide-over)
- Reachable from **every** screen (app-bar button, Decision "Ask a follow-up", Dashboard "Open the co-pilot"). 400px right panel (max 92vw), slides in over a scrim; transform `translateX(105%)`→`0`, `.34s cubic-bezier(.4,0,.2,1)`.
- Header: gradient avatar, "Co-pilot" + "Tracking Eixample analysis", close ✕.
- **Message history:** assistant bubbles (left, `#F4F4F4`) and user bubbles (right, `#222`/white). Assistant messages carry **per-message source-citation chips** (blue pills, e.g. "📊 Decision · NPV model", "📈 Market · 84 comps").
- **Suggested-question chips** (blue pills): "Should I sell?", "Why this recommendation?", "What if I self-manage?".
- **Clear conversation** action; free-text input with a circular Rausch send button.
- Always context-aware of the active analysis.

### 5. Saved analyses
- Header "Your portfolio" + a **List / Compare** segmented toggle.
- **Summary stats (4 cards):** Saved analyses 6, Recommend list 4 (teal), Avg net profit €20,200, Avg payback 3.8 yrs.
- **List view:** "Clear all" action; each saved analysis is a row — district + recommendation pill, city, Net profit, Payback, Occupancy, and **Reopen** (`#222`) + **Delete** (rausch-outline) actions.
- **Compare view** (needs ≥2): a **net-profit comparison bar chart** (bars grow-in, colored by recommendation) + a **comparison table** across Recommendation, Net profit, Payback, Occupancy, Nightly rate for the selected analyses.

---

## Interactions & Behavior
- **Navigation:** logo/Dashboard nav → Dashboard; property selector / "Workspace" → Workspace; "Saved" → Saved; Dashboard CTA / empty-state CTA → Input form; Input "Run analysis" → Workspace with a ~1.6s loading state then ready; saved cards / "Reopen" → Workspace.
- **Tabs** swap workspace content with a subtle entrance (`translateY(8px)`+fade, ~.28s) — keep transitions fast, never flashy.
- **Management toggle** (Managed/Self): recomputes results (managed applies a ~15% fee; self assumes ~6 hrs/week of owner time).
- **Investment horizon slider** and **discount rate input** recompute the NPV model live (see below) and update: the comparison header "({n}-year horizon)", the IRR line "· {rate}% discount", both Keep-&-list NPV values, the NPV-advantage centrepiece (figure, color, verdict, bar), and the assumptions caption.
- **Optimise toggles + cost inputs** recompute the live totals instantly.
- **Amenity chips & bundle presets** toggle selection state; bundles add their amenity set.
- **Co-pilot** opens/closes via slide-over; scrim click closes.
- **Motion (tasteful, fast):** count-up on KPI/NPV/IRR numbers on mount; recommendation-pill pop-in; radial gauge sweep; NPV/feature/comparison bars grow-in; card hover-lift; skeleton shimmer in loading; map-pin pop-in; subject-pin pulsing ring. Respect `prefers-reduced-motion` in the real build.

## State Management
Per active analysis / session:
- `surface`: `dashboard | input | workspace | saved` (→ routes in production).
- `tab`: `decision | market | optimise | regulatory`.
- `view`: `ready | empty | loading | error` (drive from real fetch/model status).
- `managed`: boolean (Managed vs Self).
- `copilot`: open/closed; plus message list + per-message citations.
- `marketView`: `cards | map | table`; `savedView`: `list | compare`; compare selection (≥2 ids).
- Input form: `bedrooms`, `bathrooms`, `accommodates` (steppers); `amenities` set + active `bundle`; `incomeTax` boolean; **`horizon` (default 10)**, **`discountRate` (default 7)**; plus the text/number financial fields.
- Optimise: per-improvement `{ included: boolean, cost: number }`; derived totals (count, total uplift, total cost, blended payback).
- Data to fetch: recommendation + confidence; KPIs; market comparables + benchmarks + sentiment + predicted nightly; improvement actions; regulatory + governance assessments; co-pilot responses with citations; saved analyses.

## Financial model (must match the prototype)
The prototype's NPV is driven by `horizon` (n, years) and `discountRate` (d, %), with rent growth g = 2.5%:
```
factor = Σ_{t=1..n} (1+g)^(t-1) / (1+d/100)^t
NPV_airbnb_after_tax = round_to_1000( 23449 * factor )   // calibrated to €182,000 at n=10, d=7
NPV_airbnb_pre_tax    = round_to_1000( 27572 * factor )   // → €214,000 at n=10, d=7
NPV_sell_after_CGT    = 128000   // sell-and-reinvest treated as roughly horizon-independent
NPV_sell_gross        = 158000
NPV_advantage = NPV_airbnb_after_tax − NPV_sell_after_CGT   // + favours Airbnb (teal), − favours selling (red)
airbnb_bar%   = clamp(8..92, after_tax / (after_tax + sell_after_CGT) * 100)
```
At n=10/d=7 → advantage **+€54,000** ("Listing wins"); at n=6/d=18 → **−€42,000** ("Selling wins"). **This is a representative calibration for the mock** — replace the constants with the real NPV/IRR/CGT engine (the `/scenario` endpoint already returns `npv_airbnb_p50_eur`, `npv_airbnb_pretax_p50_eur`, `npv_sell_eur`, `npv_sell_pretax_eur`, `npv_advantage_eur`, `irr_airbnb_pct`, `holding_years`, `discount_rate`), but preserve the behavior: horizon ↑ raises the Airbnb side, discount ↑ lowers it, and the advantage sign flips the centrepiece color + verdict.

## Sample data shapes (representative)
- **Saved analysis:** `{ district, city, rec: 'List on Airbnb'|'Sell'|'Marginal', recColor, recBg, net, payback, occ, nightly, netNum }`
- **Comparable:** `{ name, hood, beds, nightly, rev, rating, photoUrl, lat/long (→ mapL/mapT), nightlyNum, revNum }`
- **Improvement:** `{ id, name, uplift (€/yr), conf: 'high'|'medium'|'low', rationale, cost (editable), included }`
- **Amenity groups:** 5 groups (Comfort & Building, Outdoor & Views, Work & Smart Stays, Appliances, Family & Pets); bundles: Essentials, Remote Work Ready, Premium Luxury, Family Friendly, Urban Comfort.

## Assets
- **Font:** Plus Jakarta Sans (Google Fonts) — or licensed Airbnb Cereal if available.
- **Icons:** simple inline SVG (checks, shields, search, camera/placeholder, chat, send, plus, refresh). Replace with the codebase's icon set.
- **Listing photos:** the prototype loads stock apartment photos (Unsplash) as stand-ins with graceful placeholder/skeleton fallback. **Swap for the product's real photo URLs** (`picture_url`, available in `Data/processed/listings_segmented.parquet`; needs exposing via the comparables endpoint); keep the load-and-reveal + error-fallback behavior for dead links.
- **No real Airbnb or KPMG brand assets are embedded** — the logo is a generic gradient mark and "KPMG" is set as styled text. Use the official brand systems/marks per each brand's guidelines in the real app.

## Files (in the external handoff bundle, NOT committed to git)
Located at `../design_handoff_airbnb_investment_intelligence/` (sibling of this repo):
- `Analysis Workspace.dc.html` — the full prototype (all surfaces, tabs, states, co-pilot, and the financial model). Open in a browser to interact; read the `class Component` block for state/data/model logic and the template for structure/styling. *(Needs its `support.js` runtime to actually run; treat it primarily as a readable spec. The screenshots show the rendered result.)*
- `screenshots/` — rendered reference images of each screen:
  - `01-dashboard.png` — Dashboard landing (hero, value props, how-it-works)
  - `02-input-form.png` — New-analysis input form (location, property steppers, amenities)
  - `03-decision-hero.png` — Decision: banners, confidence gauge, recommendation pill, actions
  - `04-decision-comparison.png` — Decision: KPI cards + investment comparison with NPV-advantage centrepiece
  - `05-decision-charts.png` — Decision: distribution curve, seasonality ring, cost waterfall, feature drivers
  - `06-market-top.png` — Market: KPIs (predicted nightly) + analyst brief
  - `07-market-positioning.png` — Market: positioning bars, scatter, drivers, rating radar
  - `08-market-cards.png` — Market: Airbnb-style listing cards (Cards/Map/Table toggle)
  - `09-market-sentiment.png` — Market: benchmark cards + guest sentiment (donut, quotes, language split)
  - `10-optimise.png` — Optimise: interactive improvement planner + live totals
  - `11-regulatory.png` — Regulatory: risk badge, drivers, source links, not-legal-advice note
  - `12-saved-compare.png` — Saved: Compare view (bar chart + comparison table)
  - `13-copilot.png` — Co-pilot docked slide-over
