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

## Train the model

Install the project and train a model with:

```bash
uv sync
uv run house-price-predict-ml train
```

By default, the command reads the project CSV and saves the fitted preprocessing
and prediction pipeline to `models/house_price_model.joblib`. It reports mean
absolute error (MAE, in dollars) and the R² score on a held-out test split.

To use a different CSV or model destination:

```bash
uv run house-price-predict-ml train \
  --data path/to/houses.csv \
  --model-output path/to/model.joblib
```

The CSV must contain the same feature and target columns as the included data.
Missing feature values are imputed by the training pipeline; missing sale prices
are rejected by the dataset loader.

## Dataset

`house_price_prediction_dataset.csv` contains property details such as size,
bedrooms, condition, heating, and location, with `sale_price_usd` as the target
to predict.