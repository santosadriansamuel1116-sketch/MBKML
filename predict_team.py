# predict_team.py
import os, json, joblib, numpy as np
from typing import Dict, List, Set
from sqlalchemy import create_engine, text

# ----------------------------
# CONFIG
# ----------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "u311577524_mbk_user")
DB_PASS = os.getenv("DB_PASS", "@MBKGlamhub0812")
DB_NAME = os.getenv("DB_NAME", "u311577524_mbk_db")

ENGINE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

MODEL_DIR = os.path.join("models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "team_rf_global.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "team_rf_features.json")

# ----------------------------
# DB HELPERS
# ----------------------------
def fetch_valid_teams(booking_date: str, booking_time: str) -> Set[int]:
    sql = text("""
        SELECT t.team_id
        FROM teams t
        LEFT JOIN booking_clients bc ON bc.team_id = t.team_id
        LEFT JOIN bookings b ON b.booking_id = bc.booking_id
             AND b.booking_date = :date
             AND b.booking_time = :time
        GROUP BY t.team_id
        HAVING SUM(CASE WHEN b.booking_id IS NULL THEN 0 ELSE 1 END) = 0
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"date": booking_date, "time": booking_time}).mappings().all()
    return {int(r["team_id"]) for r in rows}

def filter_teams_by_gender(team_ids: Set[int], gender_pref_idx: int) -> Set[int]:
    if not team_ids or not gender_pref_idx or gender_pref_idx == 0:
        return team_ids
    want = "Male" if gender_pref_idx == 1 else "Female"
    placeholders = ",".join([":id"+str(i) for i,_ in enumerate(team_ids)])
    params = {("id"+str(i)): int(t) for i, t in enumerate(team_ids)}
    sql = text(f"""
        SELECT t.team_id, ma.sex AS ma_sex, hs.sex AS hs_sex
        FROM teams t
        LEFT JOIN users ma ON t.makeup_artist_id = ma.user_id
        LEFT JOIN users hs ON t.hairstylist_id = hs.user_id
        WHERE t.team_id IN ({placeholders})
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, params).mappings().all()
    good = {int(r["team_id"]) for r in rows if r["ma_sex"] == want and r["hs_sex"] == want}
    return good if good else team_ids

# ----------------------------
# MODEL LOADING
# ----------------------------
def load_features() -> List[str]:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["feature_order"]

def load_model_for_event(event_type: int):
    path = os.path.join(MODEL_DIR, f"team_rf_event_{int(event_type)}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    # fallback to global
    return joblib.load(GLOBAL_MODEL_PATH)

def make_feature_vector(payload: Dict, feature_order: List[str]) -> np.ndarray:
    # exact order, values default to 0 if missing
    v = [int(payload.get(col, 0) or 0) for col in feature_order]
    return np.array(v, dtype=np.int32).reshape(1, -1)

# ----------------------------
# PREDICT
# ----------------------------
def predict_team(payload: Dict) -> int:
    """
    payload (only form fields + date/time):
      gender_preference, hair_style, makeup_style,
      event_type, price_range, face_shape, skin_tone, hair_length,
      booking_date, booking_time
    """
    print("[ML] Incoming payload:", payload, flush=True)

    valid = fetch_valid_teams(payload["booking_date"], payload["booking_time"])
    print("[ML] Valid teams by schedule:", valid, flush=True)

    valid = filter_teams_by_gender(valid, int(payload.get("gender_preference", 0)))
    print("[ML] Valid teams after gender filter:", valid, flush=True)

    feature_order = load_features()
    model = load_model_for_event(int(payload.get("event_type", 0)))
    x = make_feature_vector(payload, feature_order)

    pred = int(model.predict(x)[0])
    print("[ML] Raw model prediction:", pred, flush=True)

    if pred not in valid and valid:
        probs = model.predict_proba(x)[0]
        classes = list(model.classes_)
        best = max(valid, key=lambda t: probs[classes.index(t)] if t in classes else -1.0)
        print("[ML] Adjusted prediction (respecting availability):", best, flush=True)
        return int(best)

    print("[ML] Final prediction:", pred, flush=True)
    return pred
