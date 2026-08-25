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
        raise HTTPException(status_code=500, detail="Machine learning model is offline.")
    
    try:
        # Let's inspect the hidden transformer structure before it runs
        if hasattr(model, 'feature_names_in_'):
            print("📋 EXACT MODEL COLUMNS EXPECTED:", list(model.feature_names_in_))
        elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            # If it's a pipeline, get the column layout names from the preprocessor step
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'feature_names_in_'):
                print("📋 PIPELINE COLUMNS EXPECTED:", list(preprocessor.feature_names_in_))
        
        # Temporary 12-slot array to avoid hard crashes
        features = [[
            data.age, data.screen_time, data.unlocks, data.study_hours,
            0, 0, 0, 0, 0, 0, 0, 0
        ]]
        
        prediction_output = model.predict(features)
        return {"status": "success", "prediction": int(prediction_output)}

    except Exception as err:
        # If the ColumnTransformer rejects the array shape, print out its structural requirements
        print("\n=================== 🚨 COLUMN TRANSFORMER BLOCK 🚨 ===================")
        print(f"CRASH REASON: {str(err)}")
        if hasattr(err, 'args'):
            print(f"ARGUMENTS: {err.args}")
        print("====================================================================\n")
        raise HTTPException(status_code=400, detail=f"Transformer Match Failure: {str(err)}")
