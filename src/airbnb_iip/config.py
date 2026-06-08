# src/airbnb_iip/config.py
PATHS = {
    "raw": "data/raw",
    "processed": "data/processed",
    "external": "data/external",
}
CITIES = ["madrid", "barcelona", "malaga"]

OCCUPANCY = {"review_rate": 0.50, "avg_length_of_stay": 3, "max_occupancy": 0.70}
FINANCE = {
    "cleaning_fee_per_stay_eur": 45,
    "management_pct": 0.20,
    "platform_fee_pct": 0.03,
    "income_tax_pct": 0.19,
}
RANDOM_STATE = 42
