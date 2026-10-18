from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# global variable for the model

ml_model = None


# load the model  at the startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    try:
        ml_model = joblib.load("student_score_model.joblib")
        print("model loaded into memory")
    except Exception as e:
        print(f"x failed to load model: {e}")
    yield
    ml_model = None


app = FastAPI(title="Student Score Predictor API", lifespan=lifespan)


# define the input schema
# these must match the exact column names from your csv
class StudentData(BaseModel):
    study_hours: float = Field(..., example=6.3)
    sleep_hours: float = Field(..., example=7.1)
    attendance_pct: float = Field(..., example=92.3)
    previous_score: float = Field(..., example=89.8)


#  Define the Output Schema
class PredictionResponse(BaseModel):
    predicted_exam_score: float


# the prediction enpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_score(data: StudentData):
    if ml_model is None:
        raise HTTPException(status_code=503, details="MOdel not found")

    # conver pydantic object to a dictionary, then to a pandas datafram
    # we use a dataframe because our model was trained on a dataframe
    input_df = pd.DataFrame([data.model_dump()])

    # make prediction
    prediction = ml_model.predict(input_df)[0]

    # return the result (rounded to 2 decimal places)
    return {"predicted_exam_score": round(float(prediction), 2)}


@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}
