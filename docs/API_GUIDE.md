# Prediction API — Improved V2

POST `/api/predict` with JSON.

Recommended fields:
`location`, `locality_hint`, `society`, `property_type`, `bhk`, `area_sqft`, `current_floor`, `total_floors`, `bathrooms`, `balconies`, `parking`, `transaction`, `furnishing`, `facing`, `ownership`, `overlooking`.

Required: `location`, `bhk`, `area_sqft`.

Example:
```json
{
  "location": "thane",
  "locality_hint": "pokhran road",
  "society": "dosti vihar",
  "property_type": "apartment",
  "bhk": 2,
  "area_sqft": 1200,
  "current_floor": 5,
  "total_floors": 14,
  "bathrooms": 2,
  "balconies": 1,
  "parking": 1,
  "transaction": "resale",
  "furnishing": "semi-furnished",
  "facing": "east",
  "ownership": "freehold",
  "overlooking": "garden/park"
}
```
