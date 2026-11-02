# Used-Car Price Prediction: a "Zillow for Used Cars"

Capstone project (General Assembly Data Science, Bahrain). Used-car prices on Craigslist are set by guesswork:
7.7% of listings say **$0** and the same car is often posted dozens of times. This project builds a regression
model that estimates a car's **fair asking price** from its age, mileage, make, condition and other details, so
sellers can price realistically and buyers can spot bad deals.

**Status:** stage 13 of 14 complete (Results & recommendations). Best model: **Random Forest, typical error $3,231 on the test set
(R² 0.80)**, about **$3,500 on genuinely new listings**, against $9,675 for a "predict the median" baseline.

## Problem statement

> Predict the fair asking price of a used car from its details (age, mileage, make, condition, fuel, body type,
> title and location), so sellers can set a realistic price and buyers can spot over- and under-priced listings.

- **Type:** supervised regression
- **Target:** `price` (US dollars)
- **Audience:** private sellers, buyers, small dealers, online marketplaces
- **Metrics:** MAE (main, in dollars), RMSE, R², compared with a "predict the median price" baseline

Full details: [docs/01_problem_statement.md](docs/01_problem_statement.md) · Final results and recommendations: [docs/05_results_and_recommendations.md](docs/05_results_and_recommendations.md)

## Dataset

[Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (Kaggle),
scraped from every US Craigslist region, 4 April to 5 May 2021. **426,880 rows × 26 columns (1.45 GB)**.
The text and link columns (`description`, `url`, `region_url`, `image_url`) are removed, because NLP is not allowed;
this leaves a 16.6 MB file in the repo. After removing reposts and impossible values: **207,644 listings, 18 features**.
Details: [data/README.md](data/README.md)

## Key EDA findings

| Finding | Number |
| --- | --- |
| Reposts of the same car | 180,118 rows (42% of the file); one car posted 261 times |
| Placeholder prices | 32,895 listings at $0 (7.7%) |
| Median / mean price (clean) | $11,950 / $15,957 (skew 2.12; −0.21 after log) |
| Car age vs price | Spearman −0.71: $36,947 at 0–1 years → $14,800 at 6–8 → ~$5,000 after 15 |
| Odometer vs price | Spearman −0.60: $27,973 under 25k miles → $5,000 above 200k |
| Condition | salvage $2,800 → good $8,500 → like new $14,650 → new $22,943 |
| Fuel / body type | diesel $25,900 vs gas $10,950; pickup $22,900 vs sedan $8,995 |
| Missing values | size 63.4% (drop); condition 37.9%, cylinders 34.8%: missing listings are pricier → "unknown" level |
| Price outliers | 4.3% by IQR, 0.4% on the log scale (real cars, kept) |

Details: [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb)

## Model results (test set, 41,529 cars)

All models predict `log(price)` and are scored in dollars after turning the prediction back with `exp`.

| Model | MAE | RMSE | R² | Median error |
| --- | --- | --- | --- | --- |
| **Random Forest** | **$3,231** | $6,184 | **0.804** | $1,746 |
| Gradient Boosting | $3,303 | $6,348 | 0.793 | $1,761 |
| ElasticNet | $4,149 | $7,768 | 0.690 | $2,152 |
| Ridge | $4,150 | $7,863 | 0.683 | $2,150 |
| Linear Regression | $4,151 | $7,886 | 0.681 | $2,151 |
| Lasso | $4,151 | $7,830 | 0.686 | $2,150 |
| Decision Tree | $4,196 | $7,433 | 0.717 | $2,354 |
| Baseline (median price) | $9,675 | $14,553 | −0.087 | $6,650 |

- The ensembles cut the typical error by 22% compared with the best linear model, and by 67% compared with the baseline.
- Linear Regression explains the drivers in plain numbers: each extra year of age lowers the price by about 6.7%,
  holding the other features fixed ([results/linear_coefficients.csv](results/linear_coefficients.csv)).
- Weakest spot: cars under $5,000, where the error is 75% of the price on average (junk and parts cars).
- **Honest error:** 25.7% of test cars have a near-identical listing in training (same make, model, year, mileage
  within 1%). On the other cars Random Forest ($3,524) and Gradient Boosting ($3,546) are tied, so about $3,500 is
  the realistic error for a new listing ([results/near_twin_check.csv](results/near_twin_check.csv)).

Details: [04_linear_models.ipynb](notebooks/04_linear_models.ipynb), [05_tree_models.ipynb](notebooks/05_tree_models.ipynb),
[results/metrics.csv](results/metrics.csv)

## What drives the price (Random Forest, feature importance + partial dependence + SHAP)

| Rank | Driver | Effect, everything else unchanged |
| --- | --- | --- |
| 1 | Age | −5 to −10% a year between about 5 and 15 years; flat after about 20 (old cars keep their value) |
| 2 | Mileage | −4 to −9% per 10,000 miles between 40k and 200k |
| 3 | "Truck" features | diesel +65% vs gas, 4wd +38% vs front-wheel drive, pickup +35% vs sedan, 8 cylinders ≈ +60% vs 4 |
| 4 | Make and condition | about 6% average move each; "fair" condition is the cheapest |
| – | State, model name, transmission, title status | 2% or less once the rest is known |

Details: [06_explainability.ipynb](notebooks/06_explainability.ipynb), [results/feature_importance.csv](results/feature_importance.csv)

## Pricing tool

The model gives an estimate plus an **80% fair price range: estimate −30% to +43%**, set on a validation split of the
training data.

| Check (test set) | Result |
| --- | --- |
| Estimate within 20% of the asking price | 59.5% of cars (Linear Regression 50.7%, median guess 16.3%) |
| Asking prices inside the 80% range | **80.0%** (85–87% for cars between $10k and $35k; 62% under $5k) |
| Flagged below / above the range | 9.9% / 10.1%: damaged cars, down-payment and placeholder prices ($1,000, $123,456), collector cars |

Details: [07_pricing_tool.ipynb](notebooks/07_pricing_tool.ipynb), [results/pricing_accuracy.csv](results/pricing_accuracy.csv)

## Repository structure

```text
used-car-price-prediction/
├── data/
│   ├── vehicles_slim.csv.gz
│   └── README.md
├── docs/
│   ├── 01_problem_statement.md
│   ├── 02_target_and_features.md
│   ├── 03_preprocessing_plan.md
│   ├── 04_course_fit_and_limitations.md
│   └── 05_results_and_recommendations.md
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_and_baseline.ipynb
│   ├── 04_linear_models.ipynb
│   ├── 05_tree_models.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_pricing_tool.ipynb
├── results/
│   ├── metrics.csv
│   ├── best_params.json
│   ├── linear_coefficients.csv
│   ├── test_predictions.csv
│   ├── near_twin_check.csv
│   ├── feature_importance.csv
│   ├── pricing_accuracy.csv
│   └── price_range.json
├── src/
│   ├── make_slim.py
│   ├── clean.py
│   └── prep.py
├── requirements.txt
└── README.md
```

## Current updates

| Stage | Date | What was added |
| --- | --- | --- |
| 13. Results & recommendations | 2026-11-03 | `docs/05_results_and_recommendations.md`: final model, model comparison, honest error, pricing tool, price drivers, goals check, recommendations, limitations, next steps |
| 12. Pricing tool | 2026-11-02 | `notebooks/07_pricing_tool.ipynb`: share of estimates within 10/20/30%, 80% fair price range from a validation split (80.0% test coverage), deal flags and what they catch, `price_car()` demo; `near_twin_mask()` moved into `src/prep.py` |
| 11. Explainability | 2026-11-02 | `notebooks/06_explainability.ipynb`: near-twin check of the test set, feature importance, partial dependence (numeric and categorical, in dollars, vs Linear Regression), SHAP with `KernelExplainer` on the 12 original features |
| 10. Tree models | 2026-11-02 | `notebooks/05_tree_models.ipynb`: Decision Tree, Random Forest and Gradient Boosting tuned with `GridSearchCV`; overfitting check, all-model comparison, error by price band, `results/test_predictions.csv` |
| 9. Linear models | 2026-11-02 | `notebooks/04_linear_models.ipynb`: Linear Regression, Ridge, Lasso, ElasticNet; log vs raw price target, residuals, coefficients (`results/linear_coefficients.csv`) |
| 8. Preprocessing & baseline | 2026-11-02 | `src/prep.py` and `notebooks/03_preprocessing_and_baseline.ipynb`: split, leakage check, top-300 models, linear and tree preprocessors, median baseline. Repost rule in `src/clean.py` now matches copies across regions (test cars with an exact twin in training: 9.5% → 0) |
| 7. Course fit & limitations | 2026-11-01 | `docs/04_course_fit_and_limitations.md`: Lightning Talk checklist, scope vs the course, limitations, open questions |
| 6. Preprocessing plan | 2026-11-01 | `docs/03_preprocessing_plan.md`: log target, imputing, encoding, scaling, models to compare |
| 5. Target & features | 2026-11-01 | `docs/02_target_and_features.md`: target choice, row rules, feature decisions |
| 4. EDA | 2026-11-01 | `notebooks/02_eda.ipynb`: the six Lightning Talk EDA questions, plus reposts, scope and location |
| 3. Data overview | 2026-11-01 | `notebooks/01_data_overview.ipynb`: data dictionary, types, nulls, reposts, placeholder values |
| 2. Problem statement | 2026-11-01 | `docs/01_problem_statement.md`: business problem, audience, goals, success metrics |
| 1. Project setup | 2026-11-01 | Slim dataset (`data/vehicles_slim.csv.gz`, `src/make_slim.py`), cleaning rules (`src/clean.py`), README, `requirements.txt`, `.gitignore` |

**Next:** stage 14: presentation slides.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order. All of them read `data/vehicles_slim.csv.gz` and apply the cleaning rules in `src/clean.py`;
notebooks 03–05 also use `src/prep.py`, so every model gets the same split and features. Notebook 05 takes about
70 minutes (mostly Gradient Boosting, which runs on one CPU core) and notebook 06 about 7 minutes and notebook 07 about 3 minutes; the others take a
few minutes each.
