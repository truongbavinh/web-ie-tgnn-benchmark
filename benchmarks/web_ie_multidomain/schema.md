# Web IE (Multi-domain) — Schema & Matching Rules

This document defines the **gold/predictions schema** and the **matching rules** used by the evaluation scripts.  
All examples are in JSON. Files are **JSON Lines** (`.jsonl`), one JSON object per line.

## 1) Record structure (per line)

Each line represents **one page/item**.

```json
{
  "id": "fashion-0001",
  "domain": "fashion",
  "url": "https://zara.com/some-page",
  "attributes": {
    "name": "Striped Shorts Limited Edition",
    "price": { "value": 99.9, "currency": "USD" },
    "material": ["cotton"],
    "color": "Ecru / Red",
    "size": ["XS", "S", "M", "L", "XL"]
  }
}
```

**Required top-level keys**
- `id` (string): unique identifier (e.g., `domain-####`).
- `domain` (string): must be one of:  
  `tourist`, `hotel`, `realestate`, `flights`, `fashion`, `events`, `app`, `course`, `scholarships`, `cooking`.
- `attributes` (object): a dictionary of attributes for this domain, described below.

**Optional**
- `url` (string): source page (helpful for auditing, not used in scoring).

---

## 2) Common attribute types

### 2.1 Primitive
- **string**  
  - Normalize using Unicode **NFKC**, trim whitespace, and compare **case-insensitively**.
- **number** (float)  
  - Standard float parsing. Examples: `rating`, `price.value`.
- **integer**  
  - Non-negative where applicable, e.g., `stops`, `bedrooms`, `bathrooms`.
- **list[string]**  
  - Unordered set equality after per-item normalization; duplicates removed.

### 2.2 Money-like objects
For `price`, `fees`, `amount` use:

```json
{ "value": 199.90, "currency": "USD" }
```

- `value` (number): numeric value.
- `currency` (string): preferred ISO-4217 (e.g., `USD`, `EUR`, `VND`, `GBP`, `JPY`, `KRW`, `CNY`).
- **Matching**:
  - Currency compared case-insensitively (exact string match).
  - Numeric tolerance (absolute) for equality checks: **±0.01**.
  - Error metrics such as **MAE/MAPE** are computed on `value` when both sides present.

### 2.3 Area object
```json
{ "value": 120.0, "unit": "m2" }
```
- `unit`: `"m2"` or `"sqft"` recommended.  
- No automatic unit conversion in baseline scoring (treat as string equality).

### 2.4 Datetime & date
- **Datetime** fields (`date_time`, `departure_time`, `arrival_time`) should be **ISO-8601**:  
  `YYYY-MM-DDTHH:MM:SSZ` or with timezone offset, e.g., `"2025-10-01T08:30:00+07:00"`.
- **Date** fields (`deadline`) use ISO date: `YYYY-MM-DD`.

---

## 3) Domain specifications

Each domain has **required** attributes (must appear in gold for that domain and are counted in exact-match-record) and **optional** attributes (scored if present but not required for exact-match).

> Attribute names are case-sensitive in JSON keys (use exactly as shown).

### 3.1 tourist
- **Required**: `name` (string), `location` (string), `rating` (number ∈ [0,5]), `price` (money), `duration` (string)
- **Optional**: —
```json
{"attributes":{"name":"Ha Long Bay Cruise","location":"Quang Ninh, Vietnam","rating":4.7,"price":{"value":159.0,"currency":"USD"},"duration":"2 days 1 night"}}
```

### 3.2 hotel
- **Required**: `name` (string), `location` (string), `price` (money), `rating` (number ∈ [0,5]), `amenities` (list[string])
- **Optional**: —
```json
{"attributes":{"name":"Sunrise Riverside Hotel","location":"Ho Chi Minh City, Vietnam","price":{"value":68.0,"currency":"USD"},"rating":4.3,"amenities":["wifi","pool","gym"]}}
```

### 3.3 realestate
- **Required**: `title` (string), `location` (string), `price` (money), `area` (area object), `bedrooms` (int ≥0), `bathrooms` (int ≥0)
- **Optional**: —
```json
{"attributes":{"title":"3BR House in District 7","location":"Ho Chi Minh City","price":{"value":220000.0,"currency":"USD"},"area":{"value":120.0,"unit":"m2"},"bedrooms":3,"bathrooms":2}}
```

### 3.4 flights
- **Required**: `name` (string), `duration` (string), `stops` (int ≥0), `price` (money), `departure_time` (datetime), `arrival_time` (datetime), `airline` (string)
- **Optional**: —
```json
{"attributes":{"name":"SGN → NRT","duration":"6h 10m","stops":0,"price":{"value":520.0,"currency":"USD"},"departure_time":"2025-10-01T08:30:00+07:00","arrival_time":"2025-10-01T16:40:00+09:00","airline":"Vietnam Airlines"}}
```

### 3.5 fashion
- **Required**: `name` (string), `price` (money)
- **Optional**: `material` (list[string]), `color` (string), `size` (list[string])
```json
{"attributes":{"name":"Striped Shorts Limited Edition","price":{"value":99.9,"currency":"USD"},"material":["cotton"],"color":"Ecru / Red","size":["XS","S","M","L","XL"]}}
```

### 3.6 events
- **Required**: `name` (string), `venue` (string), `date_time` (datetime), `artists` (list[string])
- **Optional**: —
```json
{"attributes":{"name":"Lo-fi Beats Live","venue":"Youth Theatre","date_time":"2025-11-05T19:30:00+07:00","artists":["DJ Cloud","Mimi"]}}
```

### 3.7 app
- **Required**: `name` (string), `rating` (number ∈ [0,5]), `category` (string), `developer` (string), `os` (list[string])
- **Optional**: —
```json
{"attributes":{"name":"PhotoFix Pro","rating":4.6,"category":"Photography","developer":"PixLab Inc.","os":["Android","iOS"]}}
```

### 3.8 course
- **Required**: `title` (string), `subject` (string), `fees` (money), `duration` (string), `instructor` (string)
- **Optional**: —
```json
{"attributes":{"title":"Graph Neural Networks","subject":"Machine Learning","fees":{"value":299.0,"currency":"USD"},"duration":"8 weeks","instructor":"Dr. Linh Nguyen"}}
```

### 3.9 scholarships
- **Required**: `title` (string), `provider` (string), `amount` (money), `deadline` (date), `award` (string)
- **Optional**: —
```json
{"attributes":{"title":"STEM Excellence Scholarship","provider":"ACME Foundation","amount":{"value":5000.0,"currency":"USD"},"deadline":"2025-12-15","award":"Tuition support"}}
```

### 3.10 cooking
- **Required**: `name` (string), `rating` (number ∈ [0,5]), `author` (string), `time` (string), `type` (string)
- **Optional**: —
```json
{"attributes":{"name":"Bun Bo Hue","rating":4.8,"author":"Chef Tran","time":"90 min","type":"Soup"}}
```

---

## 4) Normalization rules (applied before comparison)

- **Strings**  
  - Unicode **NFKC** → trim → lowercase for comparison.  
  - Collapsed internal whitespace recommended (optional).
- **Lists**  
  - Normalize each item (as string rules), **deduplicate**, compare as **unordered sets**.
- **Numbers**  
  - Standard float parsing. If a site shows `4.7/5`, store **4.7**.
- **Money**  
  - Currency: case-insensitive equality.  
  - `value`: numeric tolerance **±0.01** for equality; MAE/MAPE computed when both present.
- **Datetime/Date**  
  - Compare strings after normalization. Keep timezone offsets; no conversion at scoring time.

---

## 5) Matching logic & metrics

- **Slot-level matching (F1)**  
  - Evaluate per `(attribute, value)` instance. For lists, each normalized item is an instance.  
  - Compute **micro-F1** across all instances; **macro-F1** averages per attribute.
- **Exact-match record**  
  - True iff **all required attributes** for that domain match exactly (after normalization).  
  - Optional attributes do **not** affect exact-match, but they **do** count in F1 if present in gold.
- **Price/Fee/Amount error**  
  - If present in both gold & pred: compute **MAE** and **MAPE** on `.value`.  
  - Currency mismatch counts as mismatch; the value is not scored for MAE/MAPE.

> The concrete metric names in `tasks.yaml`:  
> `f1_slot_micro`, `f1_slot_macro`, `exact_match_record`, `price_mae`, `price_mape`.

---

## 6) Data authoring guidelines

- If a page **does not expose** an optional attribute, **omit** the key entirely. Do **not** emit `null` or empty strings.
- Keep `id` stable and unique. A recommended pattern is `<domain>-<4-digit index>`, e.g., `hotel-0003`.
- For **lists**, prefer concise canonical values, e.g., `["wifi","pool","gym"]` (not full sentences).
- For **money**, always include both `value` and `currency`.
- For **datetime**, prefer site’s local time **with timezone offset** if available.

---

## 7) Validation checklist (before publishing gold)

- [ ] All lines parse as JSON.  
- [ ] `id`, `domain`, `attributes` present on every line.  
- [ ] `domain` ∈ allowed set (10 domains).  
- [ ] For each domain, **all required attributes** exist in gold with correct types.  
- [ ] Money objects: `{value:number, currency:string}`.  
- [ ] Datetimes are ISO-8601 strings; dates are `YYYY-MM-DD`.  
- [ ] Integers non-negative as applicable (`stops`, `bedrooms`, `bathrooms`).  
- [ ] Lists contain strings (after normalization).  
- [ ] No PII or copyrighted raw content is embedded.

---

## 8) Example predictions (for participants)

A valid prediction line (must reuse the same `id` and `domain` as gold):
```json
{"id":"hotel-0001","domain":"hotel","attributes":{"name":"Sunrise Riverside Hotel","location":"Ho Chi Minh City, Vietnam","price":{"value":68.0,"currency":"USD"},"rating":4.3,"amenities":["wifi","pool","gym"]}}
```

Participants should produce `results/<team_or_method>/predictions.jsonl`.

---

*End of schema.*
