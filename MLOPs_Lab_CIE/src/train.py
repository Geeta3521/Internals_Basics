import os, json, joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

EXPERIMENT_NAME = "orbitcalc-signal-delay-ms"
FEATURES = ["orbit_altitude_km","ground_station_count","atmospheric_index","is_polar_orbit"]
TARGET   = "signal_delay_ms"

df = pd.read_csv(DATA_PATH)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment(EXPERIMENT_NAME)

model_configs = [
    {
        "name": "Lasso",
        "model": Lasso(alpha=1.0, max_iter=10000, random_state=42),
        "params": {"alpha": 1.0, "max_iter": 10000, "random_state": 42},
    },
    {
        "name": "GradientBoosting",
        "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    },
]

results_models = []
for cfg in model_configs:
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.set_tag("team", "ml_engineering")
        mlflow.log_params(cfg["params"])
        cfg["model"].fit(X_train, y_train)
        preds = cfg["model"].predict(X_test)
        mae  = round(float(mean_absolute_error(y_test, preds)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(cfg["model"], artifact_path="model")
        results_models.append({"name": cfg["name"], "mae": mae, "rmse": rmse})
        print(f"[{cfg['name']}] MAE={mae} RMSE={rmse}")

best = min(results_models, key=lambda x: x["mae"])
best_cfg = next(c for c in model_configs if c["name"] == best["name"])
joblib.dump(best_cfg["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump({"best_model_name": best["name"]}, f)

step1 = {
    "experiment_name": EXPERIMENT_NAME,
    "models": results_models,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}
with open(os.path.join(RESULT_DIR, "step1_s1.json"), "w") as f:
    json.dump(step1, f, indent=2)
print(json.dumps(step1, indent=2))