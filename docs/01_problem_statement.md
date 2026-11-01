# 01 · Problem Statement

## Business problem

Pricing a used car is guesswork for most people. On Craigslist, sellers type in any price they like: the raw
file has 32,895 listings at **$0** ("call for price") and a long tail of prices that are far too high or too low.
A private seller who prices too high waits weeks without a buyer; one who prices too low loses money. Buyers have
the opposite problem: they cannot tell whether an asking price is fair. Dealers also repost the same car across nearby regions (one car appears 261 times), which makes the market look
bigger and noisier than it is.

## Problem statement

> **Predict the fair asking price of a used car from its details (age, mileage, make, condition, fuel, body type,
> title and location), so sellers can set a realistic price and buyers can spot over- and under-priced listings.**

Business problem → why it matters → what ML will do:
used-car prices are set by guesswork → sellers lose time or money and buyers overpay → a regression model learns
from about 225,000 real listings what similar cars sell for and gives every car a data-based price estimate.

The pitch, in the Lightning Talk's own words, is a **"Zillow for used cars"**.

## Type of problem

**Supervised regression.** Target: `price` (US dollars, continuous).

## Audience

- **Private sellers**: set a listing price that sells without leaving money on the table.
- **Buyers**: check whether a listing is fairly priced before contacting the seller.
- **Small dealers**: price trade-ins and stock consistently instead of by feel.
- **Marketplaces (Craigslist-style sites)**: show a "fair price" badge and flag suspicious listings
  (for example a 2019 pickup listed at $1,500).

## Goals

1. Build a regression model that estimates a car's price from information a seller knows when posting the ad.
2. Beat the naive baseline: predicting the median price ($11,999) for every car.
3. Find which features drive price (age, mileage, make, condition, fuel, body type, title) and by how much.
4. Compare models from the course: Linear Regression → Ridge / Lasso → Decision Tree → Random Forest → Gradient
   Boosting, with and without a log transform of the target.
5. Give the result in plain terms: the typical error in dollars, and a price range for a given car.

## Success metrics

| Metric | Why |
| --- | --- |
| **MAE** (mean absolute error, $) | Main metric: the typical error in dollars, easy to explain, and less sensitive to the expensive outliers (lesson 05) |
| **RMSE** ($) | Punishes large errors more; shows whether the model makes big mistakes on some cars |
| **R²** | Share of the price variation the model explains; easy to compare between models |
| MAE relative to the median price | Puts the error in context: a $3,000 error matters more on a $6,000 car than on a $40,000 one |

Every model is compared on the same test set against a **baseline that predicts the median price for every car**.
A concrete success target (for example MAE below a set dollar amount) will be set after the baseline and the
first Linear Regression are run.

## Data

[Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
(Austin Reese, Kaggle). The data was collected by **scraping** Craigslist listings across the US between
4 April and 5 May 2021. Raw size: **426,880 rows × 26 columns (1.45 GB)**. After removing text, reposts and
impossible values: **225,703 listings** with 18 usable features. Details: [data/README.md](../data/README.md).
