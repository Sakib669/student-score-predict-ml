# House Price Prediction

This project explores a 5,000-row house sales dataset and develops a machine
learning model to estimate sale prices.

## Environment

The project uses `uv` to manage its Python 3.14 environment and dependencies:

```bash
uv sync
uv run jupyter lab
```

Open `index.ipynb` in Jupyter or VS Code to inspect the data and follow the
feature-preparation work.

## Dataset

`house_price_prediction_dataset.csv` contains property details such as size,
bedrooms, condition, heating, and location, with `sale_price_usd` as the target
to predict.