# Somelier

This program is an example for learning the Machine Learning.

It uses a dataset (example included) about wine in CSV format with 11 columns:
- id (Unnamed: 0)
- country
- description
- designation
- points
- price
- province
- region_1
- region_2
- variety
- winery

The goal is to predict if points are higher than 90.

## Setup

1. Clone this repo.
2. Launch setup.sh:
```bash
./setup.sh
```

## Start

```bash
./start.sh path/to/dataset.csv
# or
./start.sh http://example.com/data/dataset.csv
```
