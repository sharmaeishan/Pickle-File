# main.py inside app folder
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Define the input data model
class PredictionInput(BaseModel):
    tv_ad_spend: float

# Load model
model_filename = "Linear_Regression.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# POST request
@app.post("/predict/")
def predict(input_data: PredictionInput):
    tv_ad_spend = np.array([[input_data.tv_ad_spend]])
    predicted_sales = model.predict(tv_ad_spend)
    return {"tv_ad_spend": input_data.tv_ad_spend, "predicted_sales": predicted_sales[0]}
