"""Build the slim copy of the dataset that the notebooks use.

The raw Kaggle file `vehicles.csv` is 1.45 GB, too big for GitHub (100 MB limit).
Almost all of that size is free text and links: `description`, `url`,
`region_url` and `image_url`. Text / NLP is outside the capstone rules and the course,
so those columns are removed here. Nothing else is changed: no rows are dropped,
no values are cleaned. All cleaning happens in the notebooks.

Usage (from the repo root):
    python src/make_slim.py path/to/vehicles.csv
writes data/vehicles_slim.csv.gz (pandas reads .csv.gz directly).
"""

import sys

import pandas as pd

TEXT_AND_LINK_COLS = ['url', 'region_url', 'image_url', 'description']


def main(raw_path, out_path='data/vehicles_slim.csv.gz'):
    header = pd.read_csv(raw_path, nrows=0).columns
    keep = [c for c in header if c not in TEXT_AND_LINK_COLS]
    df = pd.read_csv(raw_path, usecols=keep, low_memory=False)
    df.to_csv(out_path, index=False, compression='gzip')
    print(f'{df.shape[0]:,} rows x {df.shape[1]} columns -> {out_path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/vehicles.csv')
