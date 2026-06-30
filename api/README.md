# API service

FastAPI layer behind the Airbnb-vs-sell decision and optimisation flows.
Agents and the Streamlit UI call these endpoints — they never run notebooks.

## Run locally

```bash
pip install -r requirements-dev.txt
pip install -e .
uvicorn api.main:app --reload
```

Then open the interactive docs at <http://127.0.0.1:8000/docs>.

## Endpoints

| Method | Path | Status | Backed by |
|---|---|---|---|
| GET  | `/health` | live | — |
| POST | `/predict_price` | **live** | LightGBM price model (`models/price_*.pkl`) |
| POST | `/estimate_occupancy` | **live** | `airbnb_iip.data.occupancy` (SF model) |
| POST | `/estimate_revenue` | **live** | `airbnb_iip.finance.costs` |
| POST | `/airbnb_vs_sell` | **live** | `airbnb_iip.finance.scenarios` + price/sale models |
| POST | `/optimise` | **live** | `airbnb_iip.agents.optimisation` (counterfactual + residual + Apriori) |

All endpoints are live and backed by the real models/finance code.

## Example

```bash
curl -X POST http://127.0.0.1:8000/predict_price \
  -H "Content-Type: application/json" \
  -d '{"city":"Madrid","property_type_std":"Entire place","accommodates":4,
       "bedrooms":2,"bathrooms_number":1,"neighbourhood_cleansed":"Salamanca"}'
# -> {"price_per_night": 106.99, "currency": "EUR", "city": "Madrid", "model": "LightGBM"}
```

All property fields are optional — missing ones are imputed with training-set
medians, so a minimal spec (e.g. just `{"city": "Madrid"}`) still returns a
number. Accuracy improves as you supply more fields.
