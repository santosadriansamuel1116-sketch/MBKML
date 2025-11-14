# predict_team.py
import os
import json
import joblib
import numpy as np
import requests
from typing import Dict, List

# ----------------------------
# CONFIG
# ----------------------------
MODEL_DIR = os.path.join("models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "team_rf_global.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "team_rf_features.json")

# Convert your Google Drive links to direct-download URLs
GLOBAL_MODEL_URL = "https://drive.google.com/uc?export=download&id=17v7bu-UOz5JARlFXfSFqlRwvTKYpnvoW"
FEATURES_URL     = "https://drive.google.com/uc?export=download&id=1ssZ4In41lQpw4p6aer385fRP9DYH66G7"

EVENT_MODEL_URLS = {
    0: "https://drive.google.com/uc?export=download&id=1iuUee-8atAln8X6GZiwWXhQJlaiJlN2s",  # Wedding
    1: "https://drive.google.com/uc?export=download&id=1suIEvP1v_t5bNRbT0fV-UMwamOaGbOVe",  # Debut
    2: "https://drive.google.com/uc?export=download&id=12_KU-RNLUtTm5TuUlXlEJUnMpXIerZWX",  # Photoshoot
    3: "https://drive.google.com/uc?export=download&id=1LuMzaggrJ20foiLDakO3kupDJIALpDwy",  # Graduation
    4: "https://drive.google.com/uc?export=download&id=1MYoS4IAbucF0VtnJB6r1l16wiB-bHobV",  # Birthday
    # 5 (Others) will fall back to global
}


def download_if_missing(url: str, path: str):
    """
    Download a file from `url` if it does not exist at `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return

    print(f"[ML] Downloading {path} from {url} ...", flush=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with open(path, "wb") as f:
        f.write(resp.content)

    print(f"[ML] Downloaded {path} ({len(resp.content)} bytes)", flush=True)


# ----------------------------
# MODEL LOADING
# ----------------------------
def load_features() -> List[str]:
    """
    Load the feature order used during training.
    Downloads the JSON file first if needed.
    """
    download_if_missing(FEATURES_URL, FEATURES_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    feature_order = data["feature_order"]
    print("[ML] Loaded feature_order:", feature_order, flush=True)
    return feature_order


def load_model_for_event(event_type: int):
    """
    Try to use a per-event model if we have a URL for this event_type.
    Otherwise fall back to the global model.
    event_type mapping (from your PHP):
      0: Wedding
      1: Debut
      2: Photoshoot
      3: Graduation
      4: Birthday
      5: Others
    """
    # First, try per-event model if URL is configured
    if event_type in EVENT_MODEL_URLS:
        event_path = os.path.join(MODEL_DIR, f"team_rf_event_{int(event_type)}.pkl")
        download_if_missing(EVENT_MODEL_URLS[event_type], event_path)
        print(f"[ML] Using per-event model: {event_path}", flush=True)
        return joblib.load(event_path)

    # Fall back to global model
    download_if_missing(GLOBAL_MODEL_URL, GLOBAL_MODEL_PATH)
    print(f"[ML] Using global model: {GLOBAL_MODEL_PATH}", flush=True)
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
      - Availability & gender logic handled in PHP

    Expected payload keys:
      gender_preference, hair_style, makeup_style,
      event_type, price_range, face_shape, skin_tone, hair_length,
      booking_date, booking_time
    """
    print("[ML] Incoming payload:", payload, flush=True)

    feature_order = load_features()
    event_type = int(payload.get("event_type", 0) or 0)
    model = load_model_for_event(event_type)

    x = make_feature_vector(payload, feature_order)
    pred = int(model.predict(x)[0])

    print("[ML] Raw model prediction (team_id):", pred, flush=True)
    return pred
