from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("senseband_model.pkl")

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.post("/predict")
def predict():
    data = request.get_json()

    try:
        row = pd.DataFrame([{
            "heart_rate": float(data["heart_rate"]),
            "sound": float(data["sound"]),
            "light": float(data["light"]),
            "movement": float(data["movement"])
        }])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid sensor data"}), 400

    prediction = str(model.predict(row)[0])

    # Return prediction plus the values used by the model.
    return jsonify({
        "prediction": prediction,
        "heart_rate": row["heart_rate"].iloc[0],
        "sound": row["sound"].iloc[0],
        "light": row["light"].iloc[0],
        "movement": row["movement"].iloc[0]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
