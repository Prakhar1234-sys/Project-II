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
        # Standardize capitalization to match typical training datasets
        gender_clean = data.gender.strip()
        academic_clean = data.academic_level.strip()
        platform_clean = data.platform.strip()

        # Map dropdown selections to common dataset values if needed
        if platform_clean == "Select Platform":
            platform_clean = "Instagram"  # Fallback baseline

        cleaned_stress_level = data.stress_level
        if cleaned_stress_level == "Med":
            cleaned_stress_level = "Medium"
        elif cleaned_stress_level == "V.High":
            cleaned_stress_level = "Very High"

        raw_data_dict = {
            "Age": [data.age],
            "Gender": [gender_clean],
            "Grouped_country": [data.country.strip()],
            "Academic_Level": [academic_clean],
            "Most_Used_Platform": [platform_clean],
            "Avg_Daily_Usage_Hours": [data.screen_time],
            "Daily_Unlocks": [data.unlocks],
            "Study_Hours": [data.study_hours],
            "Stress_Level": [cleaned_stress_level],
            # Hardcoded placeholders (If your model relies heavily on these columns,
            # try changing them to see how the model reacts!)
            "Sleep_Hours_Per_Night": [5.0],  # Lower this to test high-stress triggers
            "Purpose_Of_Use": ["Social Media"],
            "Physical_Activity_Hours": [0.5],
        }


        input_dataframe = pd.DataFrame(raw_data_dict)
        prediction_output = model.predict(input_dataframe)
        return {"status": "success", "prediction": int(prediction_output[0])}

    except Exception as err:
        print("\n=================== 🚨 MODEL CRASH DETAILS 🚨 ===================")
        print(f"ERROR: {str(err)}")
        print("==================================================================\n")
        raise HTTPException(
            status_code=400, detail=f"Model Processing Failed: {str(err)}"
        )
