import argparse, json, os, joblib
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbit_altitude_km",    type=float, required=True)
    parser.add_argument("--ground_station_count", type=int,   required=True)
    parser.add_argument("--atmospheric_index",    type=float, required=True)
    parser.add_argument("--is_polar_orbit",       type=int,   required=True)
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    features = np.array([[args.orbit_altitude_km, args.ground_station_count,
                          args.atmospheric_index, args.is_polar_orbit]])
    prediction = round(float(model.predict(features)[0]), 4)

    output = {
        "input": {
            "orbit_altitude_km":    args.orbit_altitude_km,
            "ground_station_count": args.ground_station_count,
            "atmospheric_index":    args.atmospheric_index,
            "is_polar_orbit":       args.is_polar_orbit,
        },
        "prediction_signal_delay_ms": prediction
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()