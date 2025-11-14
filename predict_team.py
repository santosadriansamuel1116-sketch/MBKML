# predict_team.py
import os
import json
import joblib
import numpy as np
from typing import Dict, List

# ----------------------------
# CONFIG
# ----------------------------
MODEL_DIR = os.path.join("models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "team_rf_global.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "team_rf_features.json")

# ----------------------------
# MODEL LOADING
# ----------------------------
def load_features() -> List[str]:
    """
    Load the feature order used during training.
    This is the list of *raw* feature names (before OneHotEncoder),
    e.g. ["gender_preference", "hair_style", ...].
    """
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["feature_order"]

def load_model_for_event(event_type: int):
    """
    Try to load a per-event model first.
    If not found, fall back to the global model.
    """
    path = os.path.join(MODEL_DIR, f"team_rf_event_{int(event_type)}.pkl")
    if os.path.exists(path):
        print(f"[ML] Using per-event model: {path}", flush=True)
        return joblib.load(path)

    print(f"[ML] Per-event model not found for event_type={event_type}, "
          f"using global model: {GLOBAL_MODEL_PATH}", flush=True)
    return joblib.load(GLOBAL_MODEL_PATH)

def make_feature_vector(payload: Dict, feature_order: List[str]) -> np.ndarray:
    """
    Build a 1xN numeric feature vector in the exact order used during training.
    Missing values default to 0.
    """
    vals = [int(payload.get(col, 0) or 0) for col in feature_order]
    x = np.array(vals, dtype=np.int32).reshape(1, -1)
    print("[ML] Feature vector:", vals, flush=True)
    return x

# ----------------------------
# PREDICT (NO DB ACCESS)
# ----------------------------
def predict_team(payload: Dict) -> int:
    """
    Pure ML prediction:
      - NO database access
      - NO availability filtering
    Availability and gender-based assignment are handled in PHP.

    Expected payload keys:
      gender_preference, hair_style, makeup_style,
      event_type, price_range, face_shape, skin_tone, hair_length,
      booking_date, booking_time
    """
    print("[ML] Incoming payload:", payload, flush=True)

    # Load feature order + appropriate model
    feature_order = load_features()
    event_type = int(payload.get("event_type", 0) or 0)
    model = load_model_for_event(event_type)

    # Build feature vector and predict
    x = make_feature_vector(payload, feature_order)
    pred = int(model.predict(x)[0])

    print("[ML] Raw model prediction (team_id):", pred, flush=True)
    return pred
