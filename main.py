import os
import joblib
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
        raise HTTPException(status_code=500, detail="Machine learning model is completely offline.")
    
    try:
        # Currently tracking 4 parameters from your original form layout
        features = [[
            data.age,
            data.screen_time,
            data.unlocks,
            data.study_hours
        ]]
        
        # Run standard prediction matrix calculation
        prediction_output = model.predict(features)
        return {"status": "success", "prediction": int(prediction_output[0])}

    except Exception as err:
        #  THIS WILL PRINT THE EXACT COLUMN NUMBER ERROR ON YOUR RENDER LOG VIEW SCREEN:
        print("\n=================== MODEL CRASH DETECTED  ===================")
        print(f"ERROR DETAILS: {str(err)}")
        print("==================================================================\n")
        raise HTTPException(status_code=400, detail=f"Model Processing Failed: {str(err)}")

