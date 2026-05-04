import json, os, csv

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE  = os.path.join(BASE_DIR, "logs", "predictions.jsonl")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "training_data.csv")
OUT_FILE  = os.path.join(BASE_DIR, "results", "step4_s5.json")
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

THRESHOLDS = {"orbit_altitude_km": 8397.4, "atmospheric_index": 1.06}

def compute_train_means():
    altitudes, atm = [], []
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            altitudes.append(float(row["orbit_altitude_km"]))
            atm.append(float(row["atmospheric_index"]))
    return {
        "orbit_altitude_km": round(sum(altitudes)/len(altitudes), 2),
        "atmospheric_index": round(sum(atm)/len(atm), 2),
    }

entries = []
with open(LOG_FILE) as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line.strip()))

train_means = compute_train_means()
preds    = [e["prediction"] for e in entries]
live_alt = [e["input"]["orbit_altitude_km"] for e in entries]
live_atm = [e["input"]["atmospheric_index"]  for e in entries]

live_means = {
    "orbit_altitude_km": round(sum(live_alt)/len(live_alt), 2),
    "atmospheric_index": round(sum(live_atm)/len(live_atm), 2),
}

alerts = []
drift_detected = False
for feature, threshold in THRESHOLDS.items():
    shift  = round(abs(live_means[feature] - train_means[feature]), 2)
    status = "ALERT" if shift > threshold else "OK"
    if status == "ALERT":
        drift_detected = True
    alerts.append({
        "feature":    feature,
        "train_mean": train_means[feature],
        "live_mean":  live_means[feature],
        "shift":      shift,
        "threshold":  threshold,
        "status":     status,
    })

result = {
    "total_predictions": len(entries),
    "mean_prediction":   round(sum(preds)/len(preds), 4),
    "drift_detected":    drift_detected,
    "alerts":            alerts,
}

with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved → {OUT_FILE}")