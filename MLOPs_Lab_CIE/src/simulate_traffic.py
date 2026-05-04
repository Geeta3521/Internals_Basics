import time, json, os, csv, random, datetime
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "predictions.jsonl")
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

try:
    import requests as req_lib
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

API_URL = "http://localhost:9000/score"

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "orbit_altitude_km":    float(row["orbit_altitude_km"]),
                "ground_station_count": int(row["ground_station_count"]),
                "atmospheric_index":    float(row["atmospheric_index"]),
                "is_polar_orbit":       int(row["is_polar_orbit"]),
            })
    return rows

train_rows = load_csv(os.path.join(BASE_DIR, "data", "training_data.csv"))
new_rows   = load_csv(os.path.join(BASE_DIR, "data", "new_data.csv"))

random.seed(42)
normal_payloads  = random.choices(train_rows, k=30)
drifted_payloads = random.choices(new_rows, k=20)

model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))

def local_predict(row):
    x = np.array([[row["orbit_altitude_km"], row["ground_station_count"],
                   row["atmospheric_index"], row["is_polar_orbit"]]])
    return round(float(model.predict(x)[0]), 4)

def write_log(input_data, prediction):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": input_data,
        "prediction": prediction,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

open(LOG_FILE, "w").close()  # clear log

print("Sending 30 NORMAL requests...")
for i, payload in enumerate(normal_payloads, 1):
    sent = False
    if HAS_REQUESTS:
        try:
            resp = req_lib.post(API_URL, json=payload, timeout=3)
            if resp.status_code == 200:
                write_log(payload, resp.json()["prediction"])
                sent = True
        except Exception:
            pass
    if not sent:
        write_log(payload, local_predict(payload))
    print(f"  [N{i:02d}] alt={payload['orbit_altitude_km']:.0f}")
    time.sleep(0.02)

print("\nWriting 20 DRIFTED records...")
for i, payload in enumerate(drifted_payloads, 1):
    pred = local_predict(payload)
    write_log(payload, pred)
    print(f"  [D{i:02d}] alt={payload['orbit_altitude_km']:.0f}  pred={pred:.2f}")

print("\nDone -- 50/50 records written.")