from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Make sure the field names match the Android variables EXACTLY (case-sensitive)
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

@app.post("/predict")
def get_prediction(data: MentalHealthPayload):
    try:
        # ⚠️ CRITICAL STEP: Your pre-trained .pkl model only understands numbers.
        # If your model was trained ONLY on numerical columns (e.g., age, screen_time, unlocks, study_hours),
        # you must list only those specific numbers in the exact order your model expects them!
        
        # Example: if your model takes: [Age, Screen Time, Unlocks, Study Hours]
        features = [[
            data.age,
            data.screen_time,
            data.unlocks,
            data.study_hours
        ]]
        
        # Run calculation
        prediction_output = model.predict(features)
        return {"status": "success", "prediction": int(prediction_output[0])}

    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
