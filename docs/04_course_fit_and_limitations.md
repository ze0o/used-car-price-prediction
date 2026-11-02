# 04 · Course Fit and Limitations

## Lightning Talk 1 checklist

| Requirement (guidelines + PDF) | Where it is covered |
| --- | --- |
| Problem statement | [01_problem_statement.md](01_problem_statement.md) |
| Potential audience | [01_problem_statement.md](01_problem_statement.md) |
| Goals | [01_problem_statement.md](01_problem_statement.md) |
| Proposed solution (regression / classification) | Supervised regression, see [01](01_problem_statement.md) and [03](03_preprocessing_plan.md) |
| Data source, how it was collected, size, features, target | [data/README.md](../data/README.md), [01_data_overview.ipynb](../notebooks/01_data_overview.ipynb) |
| Data dictionary | [01_data_overview.ipynb](../notebooks/01_data_overview.ipynb) |
| Success metrics | MAE, RMSE, R² vs a median baseline ([01](01_problem_statement.md)) |
| 15k+ rows, 10+ columns | 426,880 × 26 raw; 207,644 × 18 features after cleaning |
| No NLP / text data | `description` and the link columns are removed before loading |
| EDA: target and what it represents | [02_eda.ipynb](../notebooks/02_eda.ipynb) §2 |
| EDA: categorical or continuous, balanced, distribution plot | §3 |
| EDA: feature types and counts | §4 |
| EDA: nulls per feature | §5 |
| EDA: correlations with the target and with each other | §6 |
| EDA: outliers | §7 |

## Fit with the course

Everything in this project comes from the supervised ML lessons (01–17).

| Area | Course lessons | Used for |
| --- | --- | --- |
| EDA: distributions, IQR outliers, missing-value rules, Pearson / Spearman | 05 | Notebook 02, cleaning rules |
| Encoding, binning, log / Yeo-Johnson, scaling | 06 | Preprocessing plan |
| Linear Regression, coefficients | 02 | First model, readable drivers |
| MAE, RMSE, R² | 03 | Success metrics |
| Train/test split, k-fold cross-validation | 04, 07 | Validation |
| Ridge, Lasso, ElasticNet | 12 | Regularized linear models |
| Pipelines, ColumnTransformer, SimpleImputer, GridSearchCV | 13 | Leak-free preprocessing and tuning |
| KNN, Decision Trees, Bagging, Random Forest, Gradient Boosting | 08, 14, 15 | Non-linear models |
| Feature importance, partial dependence, SHAP (`KernelExplainer`) | 16 | Explaining the price drivers (notebook 06) |

**Note on regression versions:** the course notebooks show KNN, trees and ensembles on classification problems.
This project uses the regression versions of the same algorithms (`KNeighborsRegressor`, `DecisionTreeRegressor`,
`RandomForestRegressor`, `GradientBoostingRegressor`). They work the same way and only predict a number instead
of a class; lesson 14 itself is CART, *Classification and Regression* Trees. Likewise the baseline is the
regression twin of the `DummyClassifier` from lesson 08.

Left out because the course does not teach them: text / NLP on `description`, target encoding for `model`
(replaced by top-N + "other"), ANOVA and Cramér's V (replaced by median-price-per-category tables), and
`TransformedTargetRegressor` (replaced by taking the log of `y` by hand and `np.exp` on the predictions).
Support Vector Machines (lesson 17) are not planned: the course shows them for classification, and kernel SVMs
are too slow for about 166,000 training rows.

## Limitations

- **Asking price, not selling price:** the data shows what sellers *ask*. The model learns the market's asking
  prices, which can sit above what cars actually sell for.
- **One month, one year:** all listings are from April–May 2021, when used-car prices were rising fast (chip
  shortage). Prices today may be different; the model would need fresh data to be used now.
- **Seller-typed fields:** `condition`, `model`, `cylinders` and `type` are typed or chosen by the seller and are
  often missing (27–40%) or inconsistent. `model` has 24,166 spellings.
- **Reposts found by rule:** reposts are removed by matching VIN or price + year + make + model + exact mileage.
  After the split no test car has an exact twin in training, but a copy re-posted with a slightly changed price
  or mileage would not be caught.
- **Placeholders left in:** charm prices ($x,995) and round odometers (100,000) are real seller habits and are kept;
  a few remaining placeholder values may add noise.
- **Near-identical listings:** 25.7% of test cars have a training listing with the same make, model, year and
  mileage within 1%. Scores on the full test set are therefore a little optimistic; the error on cars without such a
  twin ($3,524 for Random Forest vs $3,231 overall) is the fairer number.
- **Odometer typos:** very low readings on old cars teach the model that almost no miles means a cheap car.
- **No trim level, options or accident history**, which explain part of the price difference between two cars of
  the same make, model and age.

## Open questions

- [ ] Confirm the dataset is approved by the IAs.
- [ ] Should model years before 1990 be a separate (collector) model later, or stay out of scope?
- [ ] Keep `paint_color` and a reduced `model` feature, or leave them out for simplicity?
- [x] Which models go in the final deck? Random Forest (best: MAE $3,231, R² 0.80), Gradient Boosting (runner-up,
  $3,303), Linear Regression (readable coefficients) and the median baseline. See [05_tree_models.ipynb](../notebooks/05_tree_models.ipynb).
- [x] Part of Random Forest's lead may come from near-identical reposts. Checked in stage 11: on test cars without a
  near twin in training, Random Forest ($3,524) and Gradient Boosting ($3,546) are tied; about $3,500 is the realistic
  error for a new listing.
- [ ] Odometer typos: 1,152 cars aged 8+ show 1,000 miles or less (median price $5,499), likely "150" typed for
  150,000. Add a fixed rule that sets these readings to missing in a later version?
