"""Fixed-rule cleaning shared by every notebook.

Each rule removes rows that are impossible, empty or repeated. None of them learns
anything from the data (no means, medians or fitted values), so they are safe to
apply before the train/test split. Learned steps (imputing, scaling, encoding)
happen later inside a Pipeline, on the training data only.

The reasons and the numbers behind every rule are in notebooks/02_eda.ipynb and
docs/02_target_and_features.md.
"""

import pandas as pd

DATA_PATH = '../data/vehicles_slim.csv.gz'
POSTING_YEAR = 2021          # every listing was posted between 4 Apr and 5 May 2021
PRICE_MIN, PRICE_MAX = 1_000, 150_000
ODOMETER_MAX = 500_000
YEAR_MIN = 1990              # older cars are a collector market: their price rises with age

CORE_COLS = ['year', 'manufacturer', 'model', 'odometer', 'fuel', 'transmission']
DROP_COLS = ['county',       # 100% empty
             'id', 'VIN',    # identifiers (VIN is only used to find reposts)
             'posting_date'] # one month of postings: only used to compute car age


def load_raw(path=DATA_PATH):
    """Load the slim file (raw data minus text and links, see src/make_slim.py)."""
    return pd.read_csv(path, low_memory=False)


def repost_mask(df):
    """True for rows that repeat a listing already seen earlier in the file.

    Dealers post the same car in several nearby regions. If copies of one car land in
    both the train and test sets, the test score is inflated, so only the first copy is kept.
    - with a VIN: same VIN, price, odometer, year and model
    - without a VIN: same price, year, manufacturer, model, odometer and region
    """
    with_vin = df['VIN'].notna() & df.duplicated(['VIN', 'price', 'odometer', 'year', 'model'])
    no_vin = (df['VIN'].isna() & df['odometer'].notna() & df['model'].notna()
              & df.duplicated(['price', 'year', 'manufacturer', 'model', 'odometer', 'region']))
    return with_vin | no_vin


def clean(df, verbose=True):
    """Apply the fixed rules in order and report how many rows each one removes."""
    log = [('raw file', len(df))]

    def step(name, keep):
        nonlocal df
        df = df[keep]
        log.append((name, len(df)))

    step('drop empty listings (all core fields missing)', ~df[CORE_COLS].isna().all(axis=1))
    step('drop reposts of the same car', ~repost_mask(df))
    step(f'keep price ${PRICE_MIN:,}-${PRICE_MAX:,}', df['price'].between(PRICE_MIN, PRICE_MAX))
    step('drop missing year or odometer', df['year'].notna() & df['odometer'].notna())
    step(f'keep odometer <= {ODOMETER_MAX:,} miles', df['odometer'] <= ODOMETER_MAX)
    step(f'keep model year >= {YEAR_MIN}', df['year'] >= YEAR_MIN)

    df = df.drop(columns=DROP_COLS).copy()
    df['year'] = df['year'].astype(int)
    df['car_age'] = POSTING_YEAR - df['year']

    if verbose:
        report = pd.DataFrame(log, columns=['step', 'rows_left'])
        report['removed'] = (-report['rows_left'].diff()).fillna(0).astype(int)
        report['share_of_raw_%'] = (report['rows_left'] / log[0][1] * 100).round(1)
        print(report.to_string(index=False))
    return df.reset_index(drop=True)


def load_clean(path=DATA_PATH, verbose=True):
    return clean(load_raw(path), verbose=verbose)
