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
        # Array shape containing the 4 numerical columns your app sends
        features = [[
            data.age,
            data.screen_time,
            data.unlocks,
            data.study_hours
        ]]
        
        prediction_output = model.predict(features)
        return {"status": "success", "prediction": int(prediction_output[0])}

    except Exception as err:
        #  THIS LINE WILL FORCE RENDER LOGS TO PRINT THE REAL MACHINE LEARNING ERROR:
        print(" MODEL CALCULATE CRASH:", str(err))
        raise HTTPException(status_code=400, detail=str(err))

