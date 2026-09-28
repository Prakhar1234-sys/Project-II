import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

# Safely resolve path to the model file in the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Mental_Health_Model.pkl')
model = joblib.load(MODEL_PATH)

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI(title="Mental Health Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Pydantic Model matching your UI payload data
class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal['Male', 'Female']
    country: str
    academic_level: Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform: str  # Kept flexible for handling Android dropdown titles safely
    purpose_of_use: str = "Entertainment" # Default placeholder if not provided by UI
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = 1.5  # Standard placeholder baseline
    sleep_hours_per_night: float = 7.0   # Standard placeholder baseline
    stress_level: str

# Output Response Model matching the original model metrics format
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float

@app.get('/')
def greet():
    return {"message": "The Mental Health AI Backend Server is live and running!"}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    # Normalize short Android UI labels to the full words the model expects
    stress_clean = data.stress_level.strip()
    if stress_clean == "Med": stress_clean = "Medium"
    elif stress_clean == "V.High": stress_clean = "Very High"
    elif stress_clean not in ['Medium', 'Low', 'Very High', 'High']: stress_clean = "Medium"

    platform_clean = data.most_used_platform.strip()
    if "Select" in platform_clean: platform_clean = "Instagram" # Fallback

    gender_clean = data.gender.strip().capitalize()
    if gender_clean not in ['Male', 'Female']: gender_clean = "Female"

    academic_clean = data.academic_level.strip()
    if "Undergraduate" in academic_clean: academic_clean = "Undergraduate"

    # Normalize Country name casing to match the top countries list comparison
    country_input = data.country.strip().capitalize()
    if country_input == "India": country_input = "India"
    
    country_group = country_input if country_input in top_countries else "Other"

    # 🌟 The Exact 13-Column Dataframe Layout required by your ColumnTransformer
    input_row = pd.DataFrame([{
        'Age': data.age,
        'Gender': gender_clean,
        'Country': country_input,
        'Academic_Level': academic_clean,
        'Most_Used_Platform': platform_clean,
        'Purpose_Of_Use': data.purpose_of_use,
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
        'Daily_Unlocks': data.daily_unlocks,
        'Study_Hours': data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
        'Stress_Level': stress_clean,
        'Grouped_country': country_group
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))
