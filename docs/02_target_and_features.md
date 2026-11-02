# 02 · Target and Feature Selection

## Target

**`price`**: the asking price of the car in US dollars. It is what every buyer and seller wants to know, and the
model's estimate can be compared directly with the price on any listing.

| Candidate target | Type | Why chosen or not |
| --- | --- | --- |
| **price** | continuous, skew 2.12 (−0.21 after log) | **Chosen**: the question the audience asks; drives a clear decision (list / buy at this price) |
| condition | 6 ordered classes, 40% missing | Seller's own opinion, missing for 40% of listings; no clear action follows |
| title_status | 6 classes, 94% "clean" | Very imbalanced and better as an input: a salvage title lowers the price |
| odometer | continuous | Easy to read off the car; nobody needs a model to predict it |

The target is modeled both as `price` and as `log(price)` (lesson 06), because the log removes the skew and turns
the curved age–price relation into a nearly straight one. Errors are always reported back in dollars.

## Row decisions (fixed rules in `src/clean.py`)

| Rule | Rows removed | Reason |
| --- | --- | --- |
| Empty listings (all core fields missing) | 68 | Only region, price and state filled in |
| **Reposts of the same car** | **180,118** | Same VIN + price + odometer, or the same price, year, make, model and exact odometer in any region (private sellers post in neighbouring cities, usually without a VIN); copies in train and test would inflate the score |
| Price outside $1,000–$150,000 | 25,832 | $0 is "call for price" (7.7% of the raw file); below $1,000 are placeholders; above $150,000 are mostly typos |
| Missing year or odometer | 2,292 | Under 1% of rows and both are key features (lesson 05: drop rows when very few are missing) |
| Odometer above 500,000 miles | 906 | Not believable (many are 999,999 or 10,000,000) |
| Model year before 1990 | 10,020 | Collector market: price rises with age, the opposite of everyday cars |
| **Clean data** | **207,644 rows** | 48.6% of the raw file |

## Feature decisions

| Feature | Decision | Reason |
| --- | --- | --- |
| year → **car_age** | Keep `car_age` = 2021 − year, drop `year` | Same information (r = −1); age is easier to explain. Strongest driver (Spearman 0.71) |
| odometer | Keep, fix the low end | Second driver (Spearman −0.60). Under 100 miles on cars older than 1 year (1,474 rows) are placeholders → set to missing, impute median |
| manufacturer | Keep | Widest price spread ($3,995 saturn → $41,800 tesla); 3.2% missing → "unknown" |
| condition | Keep | Follows its order (salvage $2,800 → new $22,943); 37.9% missing but informative → "unknown" level |
| cylinders | Keep, as a number | 4 → $8,995, 6 → $10,900, 8 → $15,000; 34.8% missing → median (or "unknown" category, tested) |
| fuel | Keep | Diesel $25,900 vs gas $10,950 |
| type | Keep | Pickup/truck $21–23k vs sedan $9k; 24.4% missing → "unknown" |
| drive | Keep | 4wd $16,800 vs fwd $8,895; 27.3% missing → "unknown" |
| title_status | Keep | Salvage / missing / parts-only titles are much cheaper |
| transmission | Keep | "other" (10,459 rows) is a real, expensive group ($24,000), not an error |
| state | Keep | $7,300 (DE) to $22,500 (AK); partly the local mix of vehicles |
| paint_color | Candidate | Weak spread ($6,995 green → $15,393 white); test with and without |
| model | Candidate, reduced | 24,166 free-text spellings; test top-N models + "other" |
| size | **Drop** | 63.4% missing (above the 30% guideline) and the weakest spread ($2,504) |
| region | **Drop** | 404 levels nested inside `state`; would add hundreds of one-hot columns |
| lat, long | **Drop** | Weak link to price (|r| ≤ 0.17), 79 rows outside the US; `state` covers location |
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
