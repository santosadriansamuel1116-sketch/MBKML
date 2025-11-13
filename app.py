from flask import Flask, request, jsonify
from predict_team import predict_team   # <-- uses your uploaded logic

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({"error": "Missing JSON payload"}), 400

        required = [
            "gender_preference", "hair_style", "makeup_style",
            "event_type", "price_range", "face_shape", "skin_tone",
            "hair_length", "booking_date", "booking_time"
        ]

        for field in required:
            if field not in payload:
                return jsonify({"error": f"Missing field: {field}"}), 400

        team_id = predict_team(payload)

        return jsonify({
            "success": True,
            "recommended_team": team_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "MBK ML API is running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
