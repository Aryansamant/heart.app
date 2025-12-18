from fastapi import FastAPI
from api.schemas import HeartFeatures, PredictionOut
from api.model_loader import get_model
import numpy as np

app = FastAPI(title="Heart Disease Classifier API", version="1.0")

@app.get("/")
def root():
    return {"message": "API is running. Go to /docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: HeartFeatures):
    model = get_model()

    x = np.array([[
        payload.age, payload.sex, payload.cp, payload.trestbps, payload.chol,
        payload.fbs, payload.restecg, payload.thalach, payload.exang,
        payload.oldpeak, payload.slope, payload.ca, payload.thal
    ]], dtype=float)

    pred = int(model.predict(x)[0])

    proba = 0.0
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0][1])

    return PredictionOut(prediction=pred, probability=proba)
