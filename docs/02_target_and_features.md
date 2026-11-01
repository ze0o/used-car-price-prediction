# 02 · Target and Feature Selection

## Target

**`price`**: the asking price of the car in US dollars. It is what every buyer and seller wants to know, and the
model's estimate can be compared directly with the price on any listing.

| Candidate target | Type | Why chosen or not |
| --- | --- | --- |
| **price** | continuous, skew 2.07 (−0.22 after log) | **Chosen**: the question the audience asks; drives a clear decision (list / buy at this price) |
| condition | 6 ordered classes, 40% missing | Seller's own opinion, missing for 40% of listings; no clear action follows |
| title_status | 6 classes, 94% "clean" | Very imbalanced and better as an input: a salvage title lowers the price |
| odometer | continuous | Easy to read off the car; nobody needs a model to predict it |

The target is modeled both as `price` and as `log(price)` (lesson 06), because the log removes the skew and turns
the curved age–price relation into a nearly straight one. Errors are always reported back in dollars.

## Row decisions (fixed rules in `src/clean.py`)

| Rule | Rows removed | Reason |
| --- | --- | --- |
| Empty listings (all core fields missing) | 68 | Only region, price and state filled in |
| **Reposts of the same car** | **158,258** | Same VIN + price + odometer (or same details in the same region) posted several times; copies in train and test would inflate the score |
| Price outside $1,000–$150,000 | 28,838 | $0 is "call for price" (7.7% of the raw file); below $1,000 are placeholders; above $150,000 are mostly typos |
| Missing year or odometer | 2,293 | Under 1% of rows and both are key features (lesson 05: drop rows when very few are missing) |
| Odometer above 500,000 miles | 979 | Not believable (many are 999,999 or 10,000,000) |
| Model year before 1990 | 10,741 | Collector market: price rises with age, the opposite of everyday cars |
| **Clean data** | **225,703 rows** | 52.9% of the raw file |

## Feature decisions

| Feature | Decision | Reason |
| --- | --- | --- |
| year → **car_age** | Keep `car_age` = 2021 − year, drop `year` | Same information (r = −1); age is easier to explain. Strongest driver (Spearman 0.70) |
| odometer | Keep, fix the low end | Second driver (Spearman −0.59). Under 100 miles on cars older than 1 year (1,832 rows) are placeholders → set to missing, impute median |
| manufacturer | Keep | Widest price spread ($3,995 saturn → $41,900 tesla); 3.3% missing → "unknown" |
| condition | Keep | Follows its order (salvage $2,800 → new $23,700); 39.7% missing but informative → "unknown" level |
| cylinders | Keep, as a number | 4 → $8,995, 6 → $10,995, 8 → $15,900; 36.2% missing → median (or "unknown" category, tested) |
| fuel | Keep | Diesel $27,999 vs gas $10,995 |
| type | Keep | Pickup/truck $22–23k vs sedan $9k; 26.7% missing → "unknown" |
| drive | Keep | 4wd $16,995 vs fwd $8,900; 28.7% missing → "unknown" |
| title_status | Keep | Salvage / missing / parts-only titles are much cheaper |
| transmission | Keep | "other" (10,605 rows) is a real, expensive group ($23,990), not an error |
| state | Keep | $7,300 (DE) to $22,500 (AK); partly the local mix of vehicles |
| paint_color | Candidate | Weak spread ($6,995 green → $15,550 white); test with and without |
| model | Candidate, reduced | 24,166 free-text spellings; test top-N models + "other" |
| size | **Drop** | 64.2% missing (above the 30% guideline) and the weakest spread ($2,905) |
| region | **Drop** | 404 levels nested inside `state`; would add hundreds of one-hot columns |
| lat, long | **Drop** | Weak link to price (|r| ≤ 0.17), 80 rows outside the US; `state` covers location |
| id, VIN, county, posting_date | **Drop** | Identifiers, 100% empty, or one month of dates (only used to compute age) |
| description, url, region_url, image_url | **Drop** (before loading) | Text and links; NLP is not allowed in the capstone |

## Feature groups

| Group | Features |
| --- | --- |
| Age and use | car_age, odometer |
| What the car is | manufacturer, (model), type, cylinders, fuel, drive, transmission, (paint_color) |
| Its state | condition, title_status |
| Where it is sold | state |

That is **12 features kept + 2 candidates** for the first models.
