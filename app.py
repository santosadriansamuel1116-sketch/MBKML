from flask import Flask, request, jsonify
from predict_team import predict_team   # your ML logic

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({
                "success": False,
                "error": "Missing JSON payload"
            }), 400

        required = [
            "gender_preference", "hair_style", "makeup_style",
            "event_type", "price_range", "face_shape", "skin_tone",
            "hair_length", "booking_date", "booking_time"
        ]

        missing = [f for f in required if f not in payload]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing field(s): {', '.join(missing)}"
            }), 400

        # 🔍 Log the input payload (shows up in Render logs)
        print("[API] /predict called with payload:", payload, flush=True)

        team_id = predict_team(payload)

        # 🔍 Log the model's decision
        print("[API] Random Forest recommended team:", team_id, flush=True)

        return jsonify({
            "success": True,
            "recommended_team": team_id,
            "source": "random_forest"   # 👈 helps you distinguish in PHP
        })

    except Exception as e:
        # 🔍 Log the error for debugging in Render logs
        print("[API] Error in /predict:", str(e), flush=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def home():
    return "MBK ML API is running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
