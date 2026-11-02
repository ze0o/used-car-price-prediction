# 05 · Results and Recommendations

## Conclusion

A **Random Forest** trained on `log(price)` estimates a used car's asking price with a typical error of
**$3,231 on the test set, about $3,500 for a genuinely new listing**, against $9,675 for guessing the median price.
It is within 20% of the asking price for 6 cars in 10, and its **80% fair price range (estimate −30% to +43%)
contains exactly 80.0% of test asking prices**. Age and mileage set most of the price; diesel, 4wd, pickup bodies
and bigger engines add large premiums.

## Final model

| Item | Choice |
| --- | --- |
| Model | `RandomForestRegressor` (100 trees, `max_features=0.3`, `min_samples_leaf=1`) |
| Target | `log(price)`; predictions turned back into dollars with `exp` before scoring |
| Features | 12 inputs: car age, odometer, cylinders, manufacturer, condition, fuel, title status, transmission, drive, body type, state, and the 300 most common model names (+ "other"); 439 columns after one-hot encoding |
| Preprocessing | Median imputation for numbers; missing categories kept as an "unknown" level (missingness carries price information) |
| Validation | Random 80/20 split (166,115 / 41,529 cars) after removing reposts; `GridSearchCV` (3-fold) on a 40,000-car training sample; test set used once |
| Price range | 10th and 90th percentiles of the log errors on a validation split of the training set |

## Model comparison (test set, 41,529 cars)

| Model | MAE | RMSE | R² | Median error | Within 20% of price |
| --- | --- | --- | --- | --- | --- |
| Baseline (median price) | $9,675 | $14,553 | −0.087 | $6,650 | 16.3% |
| Linear Regression | $4,151 | $7,886 | 0.681 | $2,151 | 50.7% |
| Ridge | $4,150 | $7,863 | 0.683 | $2,150 | – |
| Lasso | $4,151 | $7,830 | 0.686 | $2,150 | – |
| ElasticNet | $4,149 | $7,768 | 0.690 | $2,152 | – |
| Decision Tree | $4,196 | $7,433 | 0.717 | $2,354 | – |
| Gradient Boosting | $3,303 | $6,348 | 0.793 | $1,761 | 59.2% |
| **Random Forest** | **$3,231** | **$6,184** | **0.804** | **$1,746** | **59.5%** |

- **The ensembles win clearly:** Random Forest's MAE is 22% lower than the best linear model and 67% lower than the baseline.
- **`log(price)` beats raw price** for the linear model in cross-validation (MAE $4,018 vs $4,853), because prices
  are right-skewed and depreciation works in percentages.
- **Regularisation does not help:** with 166,000 training rows the linear model is not overfitting, so Ridge, Lasso
  and ElasticNet match plain Linear Regression.
- **One tree is not enough:** the tuned Decision Tree is no better than Linear Regression on typical error.

## How honest is the test score?

The same car is often re-posted with an edited price or mileage, which an exact-match rule cannot catch.
**25.7% of test cars have a "near twin" in training** (same make, model and model year, mileage within 1%).

| Model | All test cars | With near twin | No near twin |
| --- | --- | --- | --- |
| Linear Regression | $4,151 | $3,314 | $4,440 |
| Gradient Boosting | $3,303 | $2,600 | $3,546 |
| Random Forest | $3,231 | $2,383 | $3,524 |

All models do better on cars with a twin (they are common, easy-to-price cars), but Random Forest gains the most.
On cars without a twin, **Random Forest and Gradient Boosting are tied** ($3,524 vs $3,546). The fair figure to
quote for a new listing is **about $3,500, or 29% of the median price**. Random Forest is kept because it is never
worse and trains in 1.5 minutes instead of 42.

## Pricing tool performance

| Check (test set) | Result |
| --- | --- |
| Estimate within 10% / 20% / 30% of the asking price | 34.5% / 59.5% / 75.1% |
| Median error | 15.8% of the price (17.0% on new listings) |
| Asking prices inside the 80% range | **80.0%** (77.8% on new listings) |
| By price band, inside the range | under $5k 62% · $5–10k 83% · $10–20k 85% · $20–35k 87% · over $35k 80% |
| Flagged below / above the range | 9.9% / 10.1% |

**What the flags catch:** listings below the range are mostly cheap cars with twice the share of branded titles
(7.4% vs 3.7%) and four times the share of "fair" or "salvage" condition. The extreme cases are nearly new trucks
listed around $1,000 (down payments or "call for price" placeholders). Listings above the range include fake prices
such as $111,111 and $123,456, and 1990s collector cars.

## What drives a used car's price

Evidence from feature importance, partial dependence and SHAP on the Random Forest, and the Linear Regression
coefficients:

| Driver | Evidence |
| --- | --- |
| **Age** | Biggest driver (33.8% importance; moves a car's price by 44% on average in SHAP). −5 to −10% a year between about 5 and 15 years; flat after about 20 years. Linear Regression: −6.7% a year |
| **Mileage** | Second (22.2% importance; 22% SHAP). −4 to −9% per 10,000 miles between 40,000 and 200,000 |
| **Fuel** | Diesel +65% over gas, all else unchanged: the largest category effect |
| **Drive and engine** | 4wd +38% over front-wheel drive; 8 cylinders about +60% over 4 |
| **Body type** | Pickup +35% over sedan |
| **Condition** | "Fair" is the cheapest level (−41% vs the top levels); listings with no condition are priced like "excellent" (often dealers) |
| **Title** | Salvage −13% (Random Forest) to −25% (Linear Regression) vs clean; few cars have one, so the forest uses it little |
| Little effect | State, model name, transmission (each 2% or less once everything else is known) |

## Goals check

| Goal ([01](01_problem_statement.md)) | Result |
| --- | --- |
| 1. Estimate price from what a seller knows when posting | Done: 12 features, all filled in at posting time |
| 2. Beat the median baseline | MAE $3,231 vs $9,675 (−67%); R² 0.80 vs −0.09 |
| 3. Find the drivers and their size | Table above: age, mileage, then the "truck" features |
| 4. Compare the course models, with and without a log target | 8 models compared; log target better |
| 5. Plain-language error and a price range | "About $3,500 typical error; 80% range −30% to +43% of the estimate" |

## Recommendations

1. **Show a range, not a single price.** A single estimate is off by about $3,500 on a new listing; the 80% range is
   honest (80.0% coverage) and easy to explain: "most similar cars are listed between $X and $Y".
2. **Sellers: price inside the range,** close to the estimate for a typical car. The estimate drops about 30% between
   90,000 and 150,000 miles, so mileage should be stated exactly.
3. **Buyers: treat "below range" as "find out why".** Below-range listings are much more likely to have a salvage or
   rebuilt title, a "fair" condition, or a price that is not the full price.
4. **Marketplaces: use the flags as a data-quality check.** Prices like $1,000 on a 2020 truck or $123,456 on a 2013
   Buick can be caught automatically and sent back to the seller.
5. **Make the key fields required.** Blank fields push estimates towards dealer prices (a 2010 Civic with 160,000 miles
   and no other details is valued like a 2015 Camry). Condition, fuel, drive and body type should be required before a
   price is shown.
6. **Use the right model for the job:** Random Forest to price, Linear Regression to explain ("each year costs about
   6.7%").

## Limitations

- **Asking price, not selling price:** the model learns what sellers ask, which can sit above what cars sell for.
- **One month in 2021:** listings are from April–May 2021, when used-car prices were rising fast (chip shortage); the
  model needs fresh data before real use.
- **Near-identical listings:** a quarter of test cars have a near twin in training, so test scores are a little
  optimistic; $3,500 is the fairer figure.
- **Missing information:** no trim, options or accident history, which is why the price range is wide.
- **Seller-typed data:** odometer typos (1,152 cars aged 8+ show 1,000 miles or less, median price $5,499) teach the
  model that almost no miles means a cheap car; rare levels (salvage condition, "new") get little weight.
- **Scope:** cars from model year 1990 on, priced $1,000–$150,000. Cheap cars under $5,000 are priced worst
  (median error 28%), and collector cars are out of scope.

## Next steps

- Add a fixed rule that treats very low odometer readings on old cars as missing, and re-run notebooks 03–07.
- Re-validate on a later scrape of Craigslist to measure how much prices have moved since 2021.
- Try grouping near-identical listings so that one car can never sit in both training and test.
- Build the presentation: problem, data, EDA, models, honest error, pricing tool, drivers, recommendations, limitations.
