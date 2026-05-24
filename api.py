from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')

class UserInput(BaseModel):
    Gender: int
    Age: int
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

@app.post("/predict")
def predict_calories(data: UserInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)
    
    return {"calories_burned": float(prediction[0])}