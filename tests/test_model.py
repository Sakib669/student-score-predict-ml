import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from house_price_predict_ml.data import (
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    load_dataset,
    validate_dataset,
)
from house_price_predict_ml.model import train_model


def make_sample_data(rows: int = 20) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "property_type": ["House" if index % 2 else "Condo" for index in range(rows)],
            "neighborhood": ["North" if index % 2 else "South" for index in range(rows)],
            "square_feet": [900 + index * 30 for index in range(rows)],
            "lot_square_feet": [1500 + index * 20 for index in range(rows)],
            "bedrooms": [2 + index % 4 for index in range(rows)],
            "bathrooms": [1 + index % 3 for index in range(rows)],
            "year_built": [1980 + index for index in range(rows)],
            "condition": [
                ("Poor", "Fair", "Good", "Very Good", "Excellent")[index % 5]
                for index in range(rows)
            ],
            "heating_type": ["Gas" if index % 2 else "Electric" for index in range(rows)],
            "parking_spaces": [index % 3 for index in range(rows)],
            "has_garden": ["Yes" if index % 2 else "No" for index in range(rows)],
            "renovated": ["No" if index % 2 else "Yes" for index in range(rows)],
            "distance_to_city_center_km": [1.0 + index / 2 for index in range(rows)],
            "sale_price_usd": [180_000 + index * 12_000 for index in range(rows)],
        }
    )
    data.loc[0, "square_feet"] = np.nan
    data.loc[1, "neighborhood"] = None
    data.loc[2, "condition"] = None
    return data


class DatasetTests(unittest.TestCase):
    def test_loads_project_dataset_with_expected_shape(self) -> None:
        data = load_dataset()

        self.assertEqual(data.shape, (5000, len(DATASET_COLUMNS)))
        self.assertEqual(tuple(data.columns), DATASET_COLUMNS)

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_dataset(pd.DataFrame({"sale_price_usd": [100_000]}))

    def test_rejects_missing_target_values(self) -> None:
        data = make_sample_data()
        data.loc[0, "sale_price_usd"] = np.nan

        with self.assertRaisesRegex(ValueError, "contains missing values"):
            validate_dataset(data)


class ModelTests(unittest.TestCase):
    def test_training_saves_pipeline_and_predicts_unseen_categories(self) -> None:
        data = make_sample_data()
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "nested" / "model.joblib"
            model, metrics = train_model(data, model_path)

            self.assertTrue(model_path.is_file())
            self.assertTrue(np.isfinite(metrics["mae"]))
            self.assertTrue(np.isfinite(metrics["r2"]))
            prediction = model.predict(
                pd.DataFrame(
                    [
                        {
                            **data.loc[3, FEATURE_COLUMNS].to_dict(),
                            "property_type": "Castle",
                            "neighborhood": "Unknown",
                            "heating_type": "Geothermal",
                            "condition": "Exceptional",
                        }
                    ]
                )
            )
            self.assertEqual(prediction.shape, (1,))
            self.assertTrue(np.isfinite(prediction[0]))


if __name__ == "__main__":
    unittest.main()
