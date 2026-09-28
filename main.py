import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mental Health Signal API")

# Safe absolute path modeling load sequence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Mental_Health_Model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(" SUCCESS: Model loaded perfectly!")
except Exception as e:
    model = None
    print(f" ERROR: Failed to load model file: {e}")

# This data model maps directly to your Android fields
class MentalHealthPayload(BaseModel):
    age: int
    gender: str
    country: str
    academic_level: str
    platform: str
    screen_time: float
    unlocks: int
    study_hours: float
    stress_level: str

# THIS FIXES THE "DETAIL NOT FOUND" ERROR FOR THE RAW PRIMARY URL
@app.get("/")
def read_root():
    return {"message": "The Mental Health AI Backend Server is live and running!"}

@app.post("/predict")
def get_prediction(data: MentalHealthPayload):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Machine learning model is offline."
        )

    try:
        # We fill the remaining columns with standard neutral placeholder values (e.g., 0.0 or 5.0)
        # to ensure the ColumnTransformer layout sees all 12 expected training fields!
        raw_data_dict = {
            "Age": [data.age],
            "Gender": [data.gender],
            "Country": [data.country],
            "Academic_Level": [data.academic_level],
            "Platform": [data.platform],
            "Screen_Time": [data.screen_time],
            "Unlocks": [data.unlocks],
            "Study_Hours": [data.study_hours],
            "Stress_Level": [data.stress_level],
            # 👇 ADD THESE THREE FILLED LINES BELOW TO FIX THE 400 ERROR
            "Sleep_Duration": [7.0],  # Dummy standard sleep hours
            "Social_Media_Hours": [data.screen_time],  # Maps screen time directly here
            "Work_Study_Balance": [5.0],  # Neutral scale placeholder (1-10)
        }

        # Convert the dictionary map array into a Pandas DataFrame table structure
        input_dataframe = pd.DataFrame(raw_data_dict)

        # Run prediction calculations against the DataFrame schema
        prediction_output = model.predict(input_dataframe)
        return {"status": "success", "prediction": int(prediction_output[0])}

    except Exception as err:
        print("\n=================== 🚨 MODEL CRASH DETAILS 🚨 ===================")
        print(f"ERROR: {str(err)}")
        print("==================================================================\n")
        raise HTTPException(
            status_code=400,
            detail=f"Model Processing Failed: {str(err)}",
        )
