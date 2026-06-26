import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Field, Select, TextInput, Chip } from "@/components/ui/Field";
import { Stepper } from "@/components/ui/Stepper";
import { Toggle } from "@/components/ui/Toggle";
import { Segmented } from "@/components/ui/Segmented";
import { getScenario, APIError } from "@/lib/api";
import type { ScenarioRequest } from "@/lib/types";
import { useAppStore } from "@/store/useAppStore";
import {
  CITIES,
  CITY_LABELS,
  ROOM_TYPES,
  DISTRICTS,
  AMENITY_CATEGORIES,
  AMENITY_BUNDLES,
  ALL_AMENITY_KEYS,
  DEFAULT_ON_AMENITIES,
  type City,
} from "@/lib/constants";

function initialAmenities(): Record<string, boolean> {
  return Object.fromEntries(
    ALL_AMENITY_KEYS.map((k) => [k, DEFAULT_ON_AMENITIES.includes(k)]),
  );
}

export default function NewAnalysis() {
  const navigate = useNavigate();
  const setActiveAnalysis = useAppStore((s) => s.setActiveAnalysis);
  const setManagedStore = useAppStore((s) => s.setManaged);
  const setView = useAppStore((s) => s.setView);

  // location
  const [city, setCity] = useState<City>("madrid");
  const [district, setDistrict] = useState<string>(DISTRICTS.madrid[0]);
  const [nickname, setNickname] = useState("");

  // property
  const [sizeM2, setSizeM2] = useState(85);
  const [bedrooms, setBedrooms] = useState(2);
  const [bathrooms, setBathrooms] = useState(1);
  const [accommodates, setAccommodates] = useState(4);
  const [roomType, setRoomType] = useState<string>(ROOM_TYPES[0]);

  // amenities
  const [amenities, setAmenities] = useState<Record<string, boolean>>(
    initialAmenities,
  );
  const [activeBundle, setActiveBundle] = useState<string | null>(null);
  const [extraText, setExtraText] = useState("");

  // financial assumptions
  const [cadastralValue, setCadastralValue] = useState(0);
  const [ibiOverride, setIbiOverride] = useState(0);
  const [setupCost, setSetupCost] = useState(1500);
  const [purchasePrice, setPurchasePrice] = useState(0);
  const [rentGrowthPct, setRentGrowthPct] = useState(2.5);
  const [appreciationPct, setAppreciationPct] = useState(3.0);
  const [includeTax, setIncludeTax] = useState(true);
  const [holdingYears, setHoldingYears] = useState(10);
  const [discountRatePct, setDiscountRatePct] = useState(7);
  const [managed, setManaged] = useState(true);

  function onCityChange(next: City) {
    setCity(next);
    setDistrict(DISTRICTS[next][0]);
  }

  function applyBundle(name: string) {
    const set = new Set(AMENITY_BUNDLES[name] ?? []);
    setAmenities(
      Object.fromEntries(ALL_AMENITY_KEYS.map((k) => [k, set.has(k)])),
    );
    setActiveBundle(name);
  }

  function toggleAmenity(key: string) {
    setAmenities((a) => ({ ...a, [key]: !a[key] }));
    setActiveBundle(null); // manual edit clears the active bundle highlight
  }

  const mutation = useMutation({
    mutationFn: (payload: ScenarioRequest) => getScenario(payload),
    onSuccess: (scenario, payload) => {
      setActiveAnalysis(payload, scenario);
      setManagedStore(managed);
      setView("ready");
      navigate("/workspace");
    },
  });

  function handleSubmit() {
    const payload: Record<string, unknown> = {
      city,
      district,
      size_m2: sizeM2,
      bedrooms,
      bathrooms,
      accommodates,
      room_type: roomType,
      managed,
      nickname: nickname || null,
      extra_amenities: extraText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      setup_cost_eur: setupCost,
      noi_growth_rate: rentGrowthPct / 100,
      property_appreciation_rate: appreciationPct / 100,
      include_income_tax: includeTax,
      holding_years: holdingYears,
      discount_rate: discountRatePct / 100,
      ...amenities,
    };
    if (cadastralValue > 0) payload.cadastral_value = cadastralValue;
    if (ibiOverride > 0) payload.ibi_eur = ibiOverride;
    if (purchasePrice > 0) payload.purchase_price = purchasePrice;
    mutation.mutate(payload as unknown as ScenarioRequest);
  }

  const errorMsg =
    mutation.error instanceof APIError
      ? mutation.error.message
      : mutation.error
        ? "Something went wrong running the analysis."
        : null;

  return (
    <div className="mx-auto max-w-form">
      <p className="eyebrow">New analysis</p>
      <h1 className="mt-2 text-[27px] font-semibold tracking-[-0.01em]">
        Tell us about the property
      </h1>

      <div className="mt-6 flex flex-col gap-5">
        {/* Location */}
        <Card className="rounded-[18px] p-6">
          <h2 className="mb-4 text-[15px] font-semibold">Location</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Field label="City">
              <Select
                value={city}
                onChange={(e) => onCityChange(e.target.value as City)}
              >
                {CITIES.map((c) => (
                  <option key={c} value={c}>
                    {CITY_LABELS[c]}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="District">
              <Select
                value={district}
                onChange={(e) => setDistrict(e.target.value)}
              >
                {DISTRICTS[city].map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Nickname (optional)">
              <TextInput
                value={nickname}
                onChange={(e) => setNickname(e.target.value)}
                placeholder="e.g. Salamanca pied-à-terre"
              />
            </Field>
          </div>
        </Card>

        {/* Property */}
        <Card className="rounded-[18px] p-6">
          <h2 className="mb-4 text-[15px] font-semibold">Property</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field label="Size (m²)">
              <TextInput
                type="number"
                min={20}
                max={400}
                value={sizeM2}
                onChange={(e) => setSizeM2(Number(e.target.value))}
              />
            </Field>
            <Field label="Listing type">
              <Select
                value={roomType}
                onChange={(e) => setRoomType(e.target.value)}
              >
                {ROOM_TYPES.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Stepper label="Bedrooms" value={bedrooms} onChange={setBedrooms} min={0} max={8} />
            <Stepper label="Bathrooms" value={bathrooms} onChange={setBathrooms} min={1} max={6} step={0.5} />
            <Stepper label="Accommodates" value={accommodates} onChange={setAccommodates} min={1} max={16} />
          </div>
        </Card>

        {/* Amenities */}
        <Card className="rounded-[18px] p-6">
          <h2 className="text-[15px] font-semibold">Amenities</h2>
          <p className="mb-3 mt-1 text-[13px] text-foggy">
            Start from a bundle, then fine-tune.
          </p>
          <div className="mb-5 flex flex-wrap gap-2">
            {Object.keys(AMENITY_BUNDLES).map((name) => (
              <Chip
                key={name}
                label={name}
                variant="bundle"
                selected={activeBundle === name}
                onClick={() => applyBundle(name)}
              />
            ))}
          </div>
          <div className="flex flex-col gap-4">
            {AMENITY_CATEGORIES.map((cat) => (
              <div key={cat.name}>
                <p className="mb-2 text-[12px] font-semibold uppercase tracking-[0.04em] text-foggy">
                  {cat.name}
                </p>
                <div className="flex flex-wrap gap-2">
                  {cat.items.map((it) => (
                    <Chip
                      key={it.key}
                      label={it.label}
                      selected={amenities[it.key]}
                      onClick={() => toggleAmenity(it.key)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5">
            <Field
              label="Additional amenities (one per line)"
              hint="Free-text extras count toward the amenity score used to find comparables."
            >
              <textarea
                value={extraText}
                onChange={(e) => setExtraText(e.target.value)}
                rows={3}
                placeholder={"e.g.\nRooftop terrace\nSauna"}
                className="w-full rounded-input border border-inputborder bg-surface px-3 py-2 text-[14px] outline-none focus:border-foggy"
              />
            </Field>
          </div>
        </Card>

        {/* Financial assumptions */}
        <Card className="rounded-[18px] p-6">
          <h2 className="mb-4 text-[15px] font-semibold">Financial assumptions</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Field label="Cadastral value (€)" hint="For IBI — leave 0 to estimate">
              <TextInput type="number" min={0} value={cadastralValue} onChange={(e) => setCadastralValue(Number(e.target.value))} />
            </Field>
            <Field label="IBI override (€/yr)" hint="Leave 0 for auto">
              <TextInput type="number" min={0} value={ibiOverride} onChange={(e) => setIbiOverride(Number(e.target.value))} />
            </Field>
            <Field label="Airbnb setup cost (€)">
              <TextInput type="number" min={0} value={setupCost} onChange={(e) => setSetupCost(Number(e.target.value))} />
            </Field>
            <Field label="Purchase price (€)" hint="For CGT — leave 0 for default">
              <TextInput type="number" min={0} value={purchasePrice} onChange={(e) => setPurchasePrice(Number(e.target.value))} />
            </Field>
            <Field label="Annual rent growth (%)">
              <TextInput type="number" min={0} max={8} step={0.5} value={rentGrowthPct} onChange={(e) => setRentGrowthPct(Number(e.target.value))} />
            </Field>
            <Field label="Property appreciation (%)">
              <TextInput type="number" min={0} max={8} step={0.5} value={appreciationPct} onChange={(e) => setAppreciationPct(Number(e.target.value))} />
            </Field>
          </div>

          {/* Horizon + discount rate (PR #38) */}
          <div className="mt-5 grid grid-cols-1 gap-5 border-t border-hairline pt-5 sm:grid-cols-2">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-[12.5px] font-semibold text-hof-secondary">
                  Investment horizon
                </span>
                <span className="text-[14px] font-bold text-rausch">
                  {holdingYears} yrs
                </span>
              </div>
              <input
                type="range"
                min={5}
                max={30}
                step={1}
                value={holdingYears}
                onChange={(e) => setHoldingYears(Number(e.target.value))}
                className="mt-2 w-full accent-rausch"
              />
              <div className="mt-1 flex justify-between text-[11px] text-foggy">
                <span>5 yrs</span>
                <span>30 yrs</span>
              </div>
            </div>
            <Field label="Discount rate (%)" hint="Between 3% and 20%">
              <TextInput type="number" min={3} max={20} step={0.5} value={discountRatePct} onChange={(e) => setDiscountRatePct(Number(e.target.value))} />
            </Field>
          </div>

          {/* Toggles */}
          <div className="mt-5 flex flex-col gap-3 border-t border-hairline pt-5">
            <div className="flex items-center justify-between rounded-input bg-track px-4 py-3">
              <span className="text-[14px] text-hof-secondary">
                Include income tax in net profit
              </span>
              <Toggle checked={includeTax} onChange={setIncludeTax} label="Include income tax" />
            </div>
            <div className="flex items-center justify-between rounded-input bg-track px-4 py-3">
              <span className="text-[14px] text-hof-secondary">Management</span>
              <Segmented
                options={[
                  { value: "managed", label: "Professionally managed" },
                  { value: "self", label: "Self" },
                ]}
                value={managed ? "managed" : "self"}
                onChange={(v) => setManaged(v === "managed")}
              />
            </div>
          </div>
        </Card>

        {errorMsg && (
          <div className="rounded-input border border-rausch/30 bg-rec-sell-bg px-4 py-3 text-[14px] text-rausch">
            {errorMsg}
          </div>
        )}

        <div className="flex gap-3">
          <Button onClick={handleSubmit} disabled={mutation.isPending}>
            {mutation.isPending ? "Running analysis…" : "Run analysis"}
          </Button>
          <Button variant="secondary" onClick={() => navigate("/")} disabled={mutation.isPending}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}
