"""Dataset loading and schema validation for house-price modeling."""

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = (
    "property_type",
    "neighborhood",
    "square_feet",
    "lot_square_feet",
    "bedrooms",
    "bathrooms",
    "year_built",
    "condition",
    "heating_type",
    "parking_spaces",
    "has_garden",
    "renovated",
    "distance_to_city_center_km",
)
TARGET_COLUMN = "sale_price_usd"
DATASET_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "house_price_prediction_dataset.csv"
)


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a loaded dataset and return only the modeling columns."""
    missing_columns = [column for column in DATASET_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )
    if data.empty:
        raise ValueError("Dataset contains no rows")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values")
    if not pd.api.types.is_numeric_dtype(data[TARGET_COLUMN]):
        raise ValueError(f"Target column {TARGET_COLUMN!r} must be numeric")

    return data.loc[:, DATASET_COLUMNS].copy()


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the project CSV and reject missing schema or target values."""
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    return validate_dataset(pd.read_csv(dataset_path))
