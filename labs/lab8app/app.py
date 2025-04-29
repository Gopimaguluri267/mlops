from fastapi import FastAPI
from mlflow.pyfunc import load_model
from pydantic import BaseModel
import uvicorn
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Define input data model
class PredictionRequest(BaseModel):
    data: list

# Load MLflow model (update model path accordingly)
PREPROCESSOR_PATH = "/Users/gopimaguluri/Documents/msds/spring_2025_mod2/MLOps/mlops/models/preprocessor.pkl"
with open(PREPROCESSOR_PATH, 'rb') as file:
    preprocessor = pickle.load(file)

SELECTOR_PATH = "/Users/gopimaguluri/Documents/msds/spring_2025_mod2/MLOps/mlops/models/selector.pkl"
with open(SELECTOR_PATH, 'rb') as file:
    selector = pickle.load(file)

try:
    # MODEL_PATH = "/Users/gopimaguluri/Documents/msds/spring_2025_mod2/MLOps/mlops/labs/mlruns/1/b80774dc58784f55af124f49bec0c8ae/artifacts/better_models"
    MODEL_PATH = "/Users/gopimaguluri/Documents/msds/spring_2025_mod2/MLOps/mlops/labs/mlruns/6/909cb8a20e4a4ce59e5c7de340e7c430/artifacts/final_best_model"
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

app = FastAPI()

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input data to DataFrame
        if len(request.data) == 0:
            raise HTTPException(status_code=400, detail="Empty input data")

        # Check type of first element to decide how to convert
        first_elem = request.data[0]
        if isinstance(first_elem, dict):
            input_df = pd.DataFrame(request.data)
        elif isinstance(first_elem, list):
            input_df = pd.DataFrame(request.data, columns=[
                "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
                "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
            ])
        else:
            raise HTTPException(status_code=400, detail="Invalid input data format")

        # Predict
        input_df_processed = preprocessor.transform(input_df)
        selected_feature_indices = selector.get_support(indices=True)
        feature_names = preprocessor.get_feature_names_out()
        selected_feature_names = feature_names[selected_feature_indices]
        input_df_to_pass = pd.DataFrame(selector.transform(input_df_processed), columns=selected_feature_names)

        prediction = model.predict(input_df_to_pass)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
