import math
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()
model = None
model_error = None
model_path = Path(__file__).with_name("house_price_model.pkl")

try:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
except (FileNotFoundError, OSError, EOFError, AttributeError, ModuleNotFoundError) as exc:
    model_error = str(exc)

# Define the shape of the data the user will send
class HouseFeatures(BaseModel):
    property_type: str = Field(min_length=1)
    neighborhood: str = Field(min_length=1)
    square_feet: float | None = Field(default=None, ge=0)
    lot_square_feet: float | None = Field(default=None, ge=0)
    bedrooms: float | None = Field(default=None, ge=0)
    bathrooms: float | None = Field(default=None, ge=0)
    condition: Literal["Poor", "Fair", "Good", "Very Good", "Excellent"] | None = None
    heating_type: str | None = None
    parking_spaces: float | None = Field(default=None, ge=0)
    has_garden: Literal["Yes", "No"] | None = None
    renovated: Literal["Yes", "No"] | None = None
    distance_to_city_center_km: float | None = Field(default=None, ge=0)
    house_age: float | None = Field(default=None, ge=0)

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable",
        )
    return {"status": "ok"}


@app.post("/predict")
def predict_price(house: HouseFeatures):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable",
        )

    try:
        payload = house.model_dump()
        numeric_fields = (
            "square_feet", "lot_square_feet", "bedrooms", "bathrooms",
            "parking_spaces", "distance_to_city_center_km", "house_age",
        )
        for field_name in numeric_fields:
            value = payload[field_name]
            if value is not None and not math.isfinite(value):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"{field_name} must be a finite number",
                )

        condition_map = {
            "Poor": 1,
            "Fair": 2,
            "Good": 3,
            "Very Good": 4,
            "Excellent": 5,
        }
        condition = payload["condition"]
        if condition is not None:
            if condition not in condition_map:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="condition must be Poor, Fair, Good, Very Good, or Excellent",
                )
            payload["condition"] = condition_map[condition]

        yes_no_map = {"Yes": 1, "No": 0}
        for field_name in ("has_garden", "renovated"):
            value = payload[field_name]
            if value is not None:
                payload[field_name] = yes_no_map[value]

        input_data = pd.DataFrame([payload])
        prediction = model.predict(input_data)[0]

        if pd.isna(prediction):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The model could not produce a valid prediction",
            )

        return {"predicted_price": round(float(prediction), 2)}
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid house features for prediction",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction",
        ) from exc