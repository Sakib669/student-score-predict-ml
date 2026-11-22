"""Preprocessing and training pipeline for house-sale price prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from house_price_predict_ml.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataset,
)

NUMERIC_FEATURES = (
    "square_feet",
    "lot_square_feet",
    "bedrooms",
    "bathrooms",
    "year_built",
    "parking_spaces",
    "distance_to_city_center_km",
)
NOMINAL_FEATURES = (
    "property_type",
    "neighborhood",
    "heating_type",
    "has_garden",
    "renovated",
)
CONDITION_LEVELS = ("Poor", "Fair", "Good", "Very Good", "Excellent")


def build_pipeline() -> Pipeline:
    """Build a model that imputes missing values and encodes categories."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    condition_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[list(CONDITION_LEVELS)],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("nominal", nominal_pipeline, list(NOMINAL_FEATURES)),
            ("condition", condition_pipeline, ["condition"]),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    data: pd.DataFrame,
    model_path: str | Path | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit the pipeline, report holdout metrics, and optionally save it."""
    validated_data = validate_dataset(data)
    features = validated_data.loc[:, FEATURE_COLUMNS]
    target = validated_data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    if model_path is not None:
        artifact_path = Path(model_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_path)

    return model, metrics
