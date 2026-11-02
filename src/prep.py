"""Shared feature preparation, split, preprocessing and evaluation for the model notebooks.

Every model notebook imports from here, so the train/test split, the features and the
preprocessing are identical for all models.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from clean import load_clean

RANDOM_STATE = 42
TOP_MODELS = 300            # most frequent model spellings kept; the rest become "other"

NUM_COLS = ['car_age', 'odometer', 'cylinders_num']
CAT_COLS = ['manufacturer', 'condition', 'fuel', 'title_status', 'transmission',
            'drive', 'type', 'state']
MODEL_COL = 'model_top'


def load_data(path='../data/vehicles_slim.csv.gz'):
    """Clean data (src/clean.py) plus two more fixed rules that learn nothing from the data.

    - odometer under 100 miles on a car older than 1 year is a placeholder -> missing
    - cylinders is a count stored as text ("6 cylinders") -> number; "other" -> missing
    """
    df = load_clean(path, verbose=False)
    placeholder = (df['odometer'] < 100) & (df['car_age'] > 1)
    df.loc[placeholder, 'odometer'] = np.nan
    df['cylinders_num'] = pd.to_numeric(df['cylinders'].str.extract(r'(\d+)')[0], errors='coerce')
    return df


def split(df, test_size=0.2):
    """Random 80/20 split. Reposts were removed before this point, so copies of one car
    cannot end up on both sides."""
    X = df.drop(columns='price')
    y = df['price']
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def add_model_top(X_train, X_test, n=TOP_MODELS):
    """Keep the n most frequent `model` spellings (counted on the TRAINING set only) and
    group the rest as "other". Returns copies of both sets with a new `model_top` column."""
    top = X_train['model'].value_counts().head(n).index
    out = []
    for X in (X_train, X_test):
        X = X.copy()
        X[MODEL_COL] = X['model'].where(X['model'].isin(top), 'other')
        out.append(X)
    return out


def _categorical(cols, drop):
    return (make_pipeline(SimpleImputer(strategy='constant', fill_value='unknown'),
                          OneHotEncoder(drop=drop, handle_unknown='ignore', sparse_output=False)),
            cols)


def linear_preprocessor(use_model=True):
    """For Linear Regression / Ridge / Lasso / ElasticNet / KNN: impute, transform, scale, one-hot.

    - car_age: StandardScaler (no skew problem)
    - odometer: median imputation (right-skewed) then Yeo-Johnson
    - cylinders_num: median imputation then StandardScaler
    - categories: missing -> "unknown" level (missingness is informative), one-hot with
      drop='first' so the dummies are not perfectly collinear
    """
    cats = CAT_COLS + ([MODEL_COL] if use_model else [])
    pipe, cols = _categorical(cats, drop='first')
    return ColumnTransformer([
        ('age', StandardScaler(), ['car_age']),
        ('odometer', make_pipeline(SimpleImputer(strategy='median'),
                                   PowerTransformer(method='yeo-johnson')), ['odometer']),
        ('cylinders', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), ['cylinders_num']),
        ('onehot', pipe, cols),
    ], verbose_feature_names_out=False)


def tree_preprocessor(use_model=True):
    """For Decision Tree / Random Forest / Gradient Boosting: median-impute numbers, one-hot categories.

    Trees split on thresholds, so they need no scaling or transforms.
    """
    cats = CAT_COLS + ([MODEL_COL] if use_model else [])
    pipe, cols = _categorical(cats, drop=None)
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUM_COLS),
        ('onehot', pipe, cols),
    ], verbose_feature_names_out=False)


def evaluate(name, y_true, y_pred, target='log(price)'):
    """Return the project's metrics for one model as a dict (all in dollars)."""
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'model': name,
        'target': target,
        'mae': mae,
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mae_pct_of_median': mae / np.median(y_true) * 100,
        'median_abs_error': np.median(np.abs(y_true - y_pred)),
    }


def cv_dollars(pipe, X, y, log_target=True, cv=5):
    """k-fold cross-validation on the TRAINING set, scored in dollars.

    cross_val_score cannot undo the log, so the loop is written out: fit on log(price)
    (or price), turn the predictions back with np.exp, then score in dollars.
    Returns the mean MAE, RMSE and R2 over the folds.
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    rows = []
    for tr, va in KFold(cv, shuffle=True, random_state=RANDOM_STATE).split(X):
        model = clone(pipe)
        y_fit = np.log(y.iloc[tr]) if log_target else y.iloc[tr]
        model.fit(X.iloc[tr], y_fit)
        pred = model.predict(X.iloc[va])
        pred = np.exp(pred) if log_target else pred
        rows.append(evaluate('', y.iloc[va], pred))
    res = pd.DataFrame(rows)[['mae', 'rmse', 'r2']].mean()
    return res.round(3).to_dict()


def save_result(row, path='../results/metrics.csv'):
    """Append (or replace) one model's test metrics in the shared results table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old['model'] != row['model']]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values('mae', ascending=False).round(4)
    new.to_csv(path, index=False)
    return new.round(3)


def save_params(name, params, path='../results/best_params.json'):
    """Store the tuned hyper-parameters of one model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[name] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()}
    json.dump(data, open(path, 'w'), indent=2)
