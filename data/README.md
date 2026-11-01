# Data

`vehicles_slim.csv.gz`: 426,880 rows × 22 columns, gzip-compressed CSV (16.6 MB). pandas reads it directly:
`pd.read_csv('data/vehicles_slim.csv.gz')`.

**Source:** [Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data),
Austin Reese, Kaggle. Collected by scraping used-car listings from every Craigslist region in the US.
Postings run from 4 April to 5 May 2021. Each row is one listing; the target `price` is the seller's asking price
in US dollars.

## Why a slim copy?

The original `vehicles.csv` is **1.45 GB (426,880 rows × 26 columns)**, above GitHub's 100 MB file limit. Most of
that size is free text and links. `src/make_slim.py` removes only these 4 columns and changes nothing else:

| Removed column | Why |
| --- | --- |
| `description` | Free text written by the seller; NLP is not allowed in the capstone and not taught in the course |
| `url`, `region_url`, `image_url` | Links to Craigslist pages and photos |

To rebuild it from the original file:

```bash
python src/make_slim.py path/to/vehicles.csv
```

See `notebooks/01_data_overview.ipynb` for the full data dictionary and data-quality checks.
