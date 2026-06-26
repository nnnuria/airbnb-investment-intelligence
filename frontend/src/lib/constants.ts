/**
 * Static option data mirrored from airbnb_iip.decision.engine so the form's
 * dropdowns match what /scenario accepts. Districts use the engine's
 * _FALLBACK_DISTRICTS (a curated subset); a future /meta endpoint could serve
 * the full dynamic list. Amenity groups/bundles mirror app/pages/1_New_Analysis.py.
 */

export const CITIES = ["madrid", "barcelona", "malaga"] as const;
export type City = (typeof CITIES)[number];

export const CITY_LABELS: Record<City, string> = {
  madrid: "Madrid",
  barcelona: "Barcelona",
  malaga: "Málaga",
};

export const ROOM_TYPES = [
  "Entire home/apt",
  "Private room",
  "Hotel room",
  "Shared room",
] as const;

export const DISTRICTS: Record<City, string[]> = {
  madrid: [
    "Arganzuela", "Barajas", "Carabanchel", "Centro", "Chamartín",
    "Chamberí", "Ciudad Lineal", "Fuencarral - El Pardo", "Hortaleza",
    "Latina", "Moncloa-Aravaca", "Moratalaz", "Puente de Vallecas",
    "Retiro", "Salamanca", "San Blas - Canillejas", "Tetuán",
    "Usera", "Villa de Vallecas", "Villaverde",
  ],
  barcelona: [
    "Ciutat Vella", "Eixample", "Gràcia", "Horta-Guinardó",
    "Les Corts", "Nou Barris", "Sant Andreu", "Sant Martí",
    "Sants-Montjuïc", "Sarrià-Sant Gervasi",
  ],
  malaga: [
    "Bailén - Miraflores", "Carretera de Cádiz", "Centro",
    "Ciudad Jardín", "Cruz de Humilladero", "Este",
    "Puerto de la Torre", "Teatinos",
  ],
};

export interface AmenityCategory {
  name: string;
  items: { key: string; label: string }[];
}

export const AMENITY_CATEGORIES: AmenityCategory[] = [
  {
    name: "Comfort & Building",
    items: [
      { key: "has_ac", label: "Air conditioning" },
      { key: "has_elevator", label: "Elevator" },
      { key: "has_bathtub", label: "Bathtub" },
      { key: "has_hot_tub", label: "Hot tub / Jacuzzi" },
      { key: "has_gym", label: "Gym / Fitness" },
    ],
  },
  {
    name: "Outdoor & Views",
    items: [
      { key: "has_balcony", label: "Balcony or terrace" },
      { key: "has_outdoor_space", label: "Outdoor space / Garden" },
      { key: "has_view", label: "View / Skyline" },
      { key: "has_beach", label: "Beach access" },
      { key: "has_pool", label: "Pool" },
    ],
  },
  {
    name: "Work & Smart Stays",
    items: [
      { key: "has_workspace", label: "Dedicated workspace" },
      { key: "has_self_checkin", label: "Self check-in" },
      { key: "has_long_term_ok", label: "Long-term stays OK" },
      { key: "has_ev_charger", label: "EV charger" },
      { key: "has_private_entrance", label: "Private entrance" },
    ],
  },
  {
    name: "Appliances",
    items: [
      { key: "has_washer", label: "Washer" },
      { key: "has_dryer", label: "Dryer" },
      { key: "has_dishwasher", label: "Dishwasher" },
      { key: "has_cleaning_service", label: "Cleaning service" },
    ],
  },
  {
    name: "Family & Pets",
    items: [
      { key: "has_crib", label: "Crib / Baby bed" },
      { key: "has_pets", label: "Pets allowed" },
      { key: "has_parking", label: "Parking" },
    ],
  },
];

export const ALL_AMENITY_KEYS: string[] = AMENITY_CATEGORIES.flatMap((c) =>
  c.items.map((i) => i.key),
);

export const DEFAULT_ON_AMENITIES = ["has_ac", "has_elevator"];

export const AMENITY_BUNDLES: Record<string, string[]> = {
  Essentials: ["has_ac", "has_elevator", "has_washer", "has_dryer"],
  "Remote Work Ready": [
    "has_ac", "has_elevator", "has_workspace", "has_self_checkin",
    "has_long_term_ok", "has_dishwasher",
  ],
  "Premium Luxury": [
    "has_ac", "has_elevator", "has_pool", "has_hot_tub", "has_gym",
    "has_view", "has_balcony", "has_dishwasher",
  ],
  "Family Friendly": [
    "has_ac", "has_washer", "has_dryer", "has_crib", "has_pets", "has_dishwasher",
  ],
  "Urban Comfort": [
    "has_ac", "has_elevator", "has_parking", "has_dishwasher",
    "has_washer", "has_dryer",
  ],
};
