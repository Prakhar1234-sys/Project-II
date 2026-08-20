import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize our FastAPI application
app = FastAPI(title="Mental Health Predictor API")

# 2. Get the current directory path and build a safe link to your model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Mental_Health_Model.pkl")

# 3. Securely load your machine learning model file
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(" SUCCESS: Machine learning model loaded perfectly!")
    else:
        model = None
        print(" WARNING: 'Mental_Health_Model.pkl' file was not found in this folder.")
except Exception as e:
    model = None
    print(f" ERROR: Failed to load the model file. Reason: {e}")


# 4. Define the structure of incoming data coming from the user/Android
class HealthDataInput(BaseModel):
    feature_one: float
    feature_two: float


# 5. Create a default "Home" web page route just to check if the server is awake
@app.get("/")
def home():
    return {"message": "The Mental Health AI Backend Server is live and running!"}


# 6. Create the endpoint route where the Android app sends data for a prediction
@app.post("/predict")
def get_prediction(data: HealthDataInput):
    # Safeguard check to make sure the model is loaded before trying to use it
    if model is None:
        raise HTTPException(
            status_code=500, detail="Machine learning model is not loaded on this server."
        )

    try:
        # Structure the individual values into a 2D matrix array shape [[val1, val2]]
        formatted_features = [[data.feature_one, data.feature_two]]

        # Run your metrics through your trained scikit-learn model
        prediction_output = model.predict(formatted_features)

        # Send the final prediction back as a clean structured JSON package
        return {"status": "success", "prediction": int(prediction_output[0])}

    except Exception as err:
        raise HTTPException(
            status_code=400, detail=f"Failed to process model inputs: {str(err)}"
        )
