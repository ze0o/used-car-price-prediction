# Used-Car Price Prediction: a "Zillow for Used Cars"

Capstone project (General Assembly Data Science, Bahrain). Used-car prices on Craigslist are set by guesswork:
7.7% of listings say **$0** and the same car is often posted dozens of times. This project builds a regression
model that estimates a car's **fair asking price** from its age, mileage, make, condition and other details, so
sellers can price realistically and buyers can spot bad deals.

**Status:** stage 7 of 14 complete (Course fit & limitations). Lightning Talk 1 material ready; modeling is next.

## Problem statement

> Predict the fair asking price of a used car from its details (age, mileage, make, condition, fuel, body type,
> title and location), so sellers can set a realistic price and buyers can spot over- and under-priced listings.

- **Type:** supervised regression
- **Target:** `price` (US dollars)
- **Audience:** private sellers, buyers, small dealers, online marketplaces
- **Metrics:** MAE (main, in dollars), RMSE, R², compared with a "predict the median price" baseline

Full details: [docs/01_problem_statement.md](docs/01_problem_statement.md)

## Dataset

[Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (Kaggle),
scraped from every US Craigslist region, 4 April to 5 May 2021. **426,880 rows × 26 columns (1.45 GB)**.
The text and link columns (`description`, `url`, `region_url`, `image_url`) are removed, because NLP is not allowed;
this leaves a 16.6 MB file in the repo. After removing reposts and impossible values: **225,703 listings, 18 features**.
Details: [data/README.md](data/README.md)

## Key EDA findings

| Finding | Number |
| --- | --- |
| Reposts of the same car | 158,258 rows (37% of the file); one car posted 261 times |
| Placeholder prices | 32,895 listings at $0 (7.7%) |
| Median / mean price (clean) | $11,999 / $16,334 (skew 2.07; −0.22 after log) |
| Car age vs price | Spearman −0.70: $37,859 at 0–1 years → $14,995 at 6–8 → ~$5,000 after 15 |
| Odometer vs price | Spearman −0.59: $27,995 under 25k miles → $5,500 above 200k |
| Condition | salvage $2,800 → good $8,500 → like new $14,995 → new $23,700 |
| Fuel / body type | diesel $27,999 vs gas $10,995; pickup $22,995 vs sedan $8,999 |
| Missing values | size 64.2% (drop); condition 39.7%, cylinders 36.2%: missing listings are pricier → "unknown" level |
| Price outliers | 4.2% by IQR, 0.4% on the log scale (real cars, kept) |

Details: [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb)

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
│   └── 04_course_fit_and_limitations.md
├── notebooks/
│   ├── 01_data_overview.ipynb
│   └── 02_eda.ipynb
├── src/
│   ├── make_slim.py
│   └── clean.py
├── requirements.txt
└── README.md
```

## Current updates

| Stage | Date | What was added |
| --- | --- | --- |
| 7. Course fit & limitations | 2026-11-01 | `docs/04_course_fit_and_limitations.md`: Lightning Talk checklist, scope vs the course, limitations, open questions |
| 6. Preprocessing plan | 2026-11-01 | `docs/03_preprocessing_plan.md`: log target, imputing, encoding, scaling, models to compare |
| 5. Target & features | 2026-11-01 | `docs/02_target_and_features.md`: target choice, row rules, feature decisions |
| 4. EDA | 2026-11-01 | `notebooks/02_eda.ipynb`: the six Lightning Talk EDA questions, plus reposts, scope and location |
| 3. Data overview | 2026-11-01 | `notebooks/01_data_overview.ipynb`: data dictionary, types, nulls, reposts, placeholder values |
| 2. Problem statement | 2026-11-01 | `docs/01_problem_statement.md`: business problem, audience, goals, success metrics |
| 1. Project setup | 2026-11-01 | Slim dataset (`data/vehicles_slim.csv.gz`, `src/make_slim.py`), cleaning rules (`src/clean.py`), README, `requirements.txt`, `.gitignore` |

**Next:** Lightning Talk slides, then stage 8: preprocessing pipeline and baseline model.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order. Both read `data/vehicles_slim.csv.gz`; `02_eda.ipynb` applies the cleaning rules in `src/clean.py`.
