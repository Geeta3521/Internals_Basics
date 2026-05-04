import os, json, joblib
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "meta.json")
LOG_FILE   = os.path.join(BASE_DIR, "logs", "predictions.jsonl")
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

model = joblib.load(MODEL_PATH)
model_name = "Lasso"
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        model_name = json.load(f).get("best_model_name", model_name)

app = FastAPI(title="OrbitCalc Signal Delay API", version="1.0")

class SignalInput(BaseModel):
    orbit_altitude_km:    float = Field(..., ge=200.0,  le=36000.0)
    ground_station_count: int   = Field(..., ge=1,      le=10)
    atmospheric_index:    float = Field(..., ge=1.0,    le=5.0)
    is_polar_orbit:       int   = Field(..., ge=0,      le=1)

    @validator("is_polar_orbit")
    def must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("is_polar_orbit must be 0 or 1")
        return v

@app.get("/ping")
def ping():
    return {"status": "running", "model": model_name, "version": "1.0"}

@app.post("/score")
def score(payload: SignalInput):
    features = np.array([[payload.orbit_altitude_km, payload.ground_station_count,
                          payload.atmospheric_index, payload.is_polar_orbit]])
    pred = round(float(model.predict(features)[0]), 4)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": payload.dict(),
        "prediction": pred
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"prediction": pred, "model": model_name, "version": "1.0"}