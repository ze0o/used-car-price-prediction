# 03 · Preprocessing Plan

Every method below is taught in the course (lesson numbers in the last column). The fixed rules in `src/clean.py`
run first (see `02_target_and_features.md`); everything here **learns from the data**, so it goes inside a
`Pipeline` and is fit on the training data only.

| Feature | Treatment | Why | Lesson |
| --- | --- | --- | --- |
| price (target) | Compare raw `price` vs `np.log(price)`; convert predictions back with `np.exp` before scoring | Skew 2.12 → −0.21; IQR outliers 4.3% → 0.4% | 06 |
| car_age | `StandardScaler`; optionally bin with `pd.cut` into 0–1 / 2–3 / 4–5 / 6–8 / 9–11 / 12–15 / 16–20 / 21+ | Value drops fast in the first years then flattens; bins let linear models follow the curve | 06 |
| odometer | Set < 100 miles on cars older than 1 year to missing → `SimpleImputer(strategy='median')` → `PowerTransformer` (Yeo-Johnson) or `StandardScaler` | Placeholders at 0 / 1; skew 0.60; median because of the right tail | 05, 06, 13 |
| condition | One-hot with "unknown" as its own level; test `OrdinalEncoder` (salvage < fair < good < excellent < like new < new, unknown → middle rank) as the alternative | Clear order, but missingness carries information (dealer listings: $14,900 vs $9,999 median), which one-hot keeps | 05, 06 |
| cylinders | Text → number ("6 cylinders" → 6, "other" → missing) → `SimpleImputer(strategy='median')`; test keeping it as a category with "unknown" instead | It is a count stored as text; price rises with it; 36% missing | 05, 06 |
| manufacturer, fuel, title_status, transmission, drive, type, state, (paint_color) | `SimpleImputer(strategy='constant', fill_value='unknown')` → `OneHotEncoder(handle_unknown='ignore')` | Nominal; "unknown" kept as a level because missingness is informative | 05, 06, 13 |
| model (candidate) | Keep the most frequent N models, rest → "other", then one-hot | 24,166 spellings; one-hot on all is impossible | 06 |

## Scaling

`StandardScaler` (or `RobustScaler` if outliers remain) is needed for Linear Regression, Ridge, Lasso and
ElasticNet, whose coefficient sizes depend on the feature scale, and for KNN, which uses distances.
Decision Tree, Random Forest and Gradient Boosting do not need scaling or transforms.

## Splitting and validation

- **Random 80/20 train/test split** (`random_state=42`) after the fixed cleaning, so reposted copies of one car
  cannot land on both sides. No stratification is needed for a continuous target.
- Tune with `GridSearchCV` / `cross_val_score` using 5-fold cross-validation on the training set only; evaluate once on the
  untouched test set (lessons 04, 13).
- All postings come from one month (April–May 2021), so a time-based split is not needed.

## Models to compare (in order)

1. **Baseline**: predict the training median price for every car.
2. **Linear Regression** on `price` and on `log(price)`: readable coefficients (lesson 02).
3. **Ridge, Lasso, ElasticNet**: handle the overlap between age and mileage and the many one-hot columns (lesson 12).
4. **KNN** (regression version): distance-based, needs scaling (lesson 08).
5. **Decision Tree** (CART regression tree), tuned on `max_depth`, `min_samples_leaf` (lesson 14).
6. **Random Forest / Bagging**, then **Gradient Boosting** (lesson 15), with feature importance, partial
   dependence and SHAP for explanation (lesson 16).

The training set has about 166,000 rows, so KNN and slow grid searches can be tuned on a random sample first.

## Order of operations

1. Load and apply the fixed rules (`src/clean.py`).
2. Fix the cylinders text and odometer placeholders (fixed rules, no learning).
3. Train/test split 80/20.
4. Build the `ColumnTransformer` (impute, encode, scale) and put it in a `Pipeline` with the model.
5. Tune with cross-validation on the training set; evaluate once on the test set in dollars (MAE, RMSE, R²).
