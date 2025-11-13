# evaluate_and_train_rf.py
import os, json, joblib, numpy as np, pandas as pd
from typing import List, Dict, Tuple

from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, top_k_accuracy_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline  # safe even if no sampler
# from imblearn.over_sampling import RandomOverSampler  # keep commented unless you introduce imbalance

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "mbk_db")

ENGINE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
CSV_PATH = os.path.join(os.getcwd(), "historical_bookings.csv")

MODEL_DIR = os.path.join("models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "team_rf_global.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "team_rf_features.json")

engine = create_engine(ENGINE_URL, pool_pre_ping=True)

FEATURES: List[str] = [
    "gender_preference","hair_style","makeup_style",
    "event_type","price_range","face_shape","skin_tone","hair_length"
]
TARGET = "team_id"

def load_dataframe() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        print(f"Using CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        print("CSV not found; falling back to DB.")
        sql = """
            SELECT
                bc.gender_preference, bc.hair_style, bc.makeup_style,
                bc.event_type, bc.price_range, bc.face_shape, bc.skin_tone, bc.hair_length,
                bc.team_id
            FROM booking_clients bc
            JOIN bookings b ON bc.booking_id = b.booking_id
            WHERE bc.team_id IS NOT NULL
        """
        df = pd.read_sql(sql, engine)

    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    for c in FEATURES:
        df[c] = df[c].fillna(0).astype(int)

    # drop zero-only rows if there is enough data left
    mask = (df[FEATURES].sum(axis=1) > 0)
    if mask.sum() >= 50:
        df = df[mask].copy()
    return df[FEATURES + [TARGET]]

def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(len(FEATURES))))],
        remainder="drop",
        verbose_feature_names_out=False
    )
    pipe = Pipeline(steps=[
        ("onehot", pre),
        # ("ros", RandomOverSampler(random_state=42)),  # data is balanced; enable only if needed
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return pipe

def tune_and_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    skf = StratifiedKFold(n_splits=max(2, min(5, n_splits)), shuffle=True, random_state=42)
    pipe = make_pipeline()

    param_dist = {
        "rf__n_estimators":      [1200, 1600],
        "rf__max_depth":         [30, 36, 40],
        "rf__min_samples_split": [2, 4, 6],
        "rf__min_samples_leaf":  [1, 2, 3],
        "rf__max_features":      ["sqrt", "log2", None],
        "rf__bootstrap":         [True, False],
    }

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=24,
        scoring="f1_macro", cv=skf, random_state=42, n_jobs=-1, verbose=0, error_score="raise"
    )
    search.fit(X, y)
    best = search.best_estimator_

    fold_top1, fold_top2, fold_f1 = [], [], []
    last_report, last_cm = None, None
    classes = sorted(y.unique())

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        best.fit(X_tr, y_tr)
        y_pr = best.predict(X_te)
        y_proba = best.predict_proba(X_te)

        fold_top1.append(accuracy_score(y_te, y_pr))
        fold_top2.append(top_k_accuracy_score(y_te, y_proba, k=2, labels=best.named_steps["rf"].classes_))
        fold_f1.append(f1_score(y_te, y_pr, average="macro", zero_division=0))
        last_report = classification_report(y_te, y_pr, zero_division=0)
        last_cm = confusion_matrix(y_te, y_pr, labels=classes)

    return {
        "best_estimator": best,
        "avg_top1": float(np.mean(fold_top1)),
        "avg_top2": float(np.mean(fold_top2)),
        "avg_macro_f1": float(np.mean(fold_f1)),
        "last_report": last_report,
        "last_cm": last_cm.tolist() if last_cm is not None else None,
        "classes": classes,
        "best_params": search.best_params_,
    }

def save_model(estimator, path: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(estimator, path)

def log_metrics_to_db(model_name: str, top1: float, top2: float, f1m: float, classes_covered: int):
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS model_metrics (
              id INT AUTO_INCREMENT PRIMARY KEY,
              model_name VARCHAR(100) NOT NULL,
              accuracy DECIMAL(6,3) NOT NULL,
              macro_f1 DECIMAL(6,3) NOT NULL,
              classes_covered INT NOT NULL,
              evaluated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.exec_driver_sql(
            "INSERT INTO model_metrics (model_name, accuracy, macro_f1, classes_covered) VALUES (%s, %s, %s, %s)",
            (model_name, round(top1, 3), round(f1m, 3), classes_covered)
        )

def main():
    print("Loading training data...")
    df = load_dataframe()
    if df.empty: raise ValueError("Training set is empty.")
    Xg, yg = df[FEATURES], df[TARGET]

    print(f"Teams in data: {sorted(yg.unique())} | Samples: {len(yg)}")
    print("Class distribution (team_id -> count):")
    for k, v in yg.value_counts().sort_index().items():
        print(f"  Team {int(k)} -> {int(v)}")
    baseline = yg.value_counts().max()/len(yg)
    print(f"\nMajority-class baseline accuracy: {baseline:.3f}\n")

    # Save shared feature order (the raw column names; OneHot is inside pipeline)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_order": FEATURES}, f, indent=2)

    # ----- GLOBAL -----
    print("Training GLOBAL RandomForest (all events combined)...\n")
    min_per_class = yg.value_counts().min()
    n_splits = max(2, min(5, int(min_per_class)))
    g = tune_and_cv(Xg, yg, n_splits=n_splits)
    print(f"[GLOBAL] Top-1 Accuracy (avg): {g['avg_top1']:.3f}")
    print(f"[GLOBAL] Top-2 Accuracy (avg): {g['avg_top2']:.3f}")
    print(f"[GLOBAL] Macro-F1 (avg): {g['avg_macro_f1']:.3f}")
    print("[GLOBAL] Best params:", g["best_params"])
    if g["last_report"]:
        print("\n[GLOBAL] Last fold classification report:")
        print(g["last_report"])
    if g["last_cm"]:
        print("[GLOBAL] Last fold confusion matrix (rows=true, cols=pred):")
        print(np.array(g["last_cm"]))

    save_model(g["best_estimator"], GLOBAL_MODEL_PATH)
    log_metrics_to_db("team_rf_global", g["avg_top1"], g["avg_top2"], g["avg_macro_f1"], len(g["classes"]))

    # ----- PER EVENT -----
    print("\nTraining PER-EVENT RandomForests...")
    for ev in sorted(df["event_type"].unique()):
        seg = df[df["event_type"] == ev].copy()
        Xs, ys = seg[FEATURES], seg[TARGET]
        if ys.nunique() < 2 or len(ys) < 60:
            print(f"  [event_type={ev}] skipped (insufficient data)")
            continue
        min_per_class = ys.value_counts().min()
        n_splits = max(2, min(5, int(min_per_class)))
        m = tune_and_cv(Xs, ys, n_splits=n_splits)
        print(f"\n  [event_type={ev}] Top-1={m['avg_top1']:.3f}  Top-2={m['avg_top2']:.3f}  Macro-F1={m['avg_macro_f1']:.3f}")
        print(f"  [event_type={ev}] Best params: {m['best_params']}")
        save_model(m["best_estimator"], os.path.join(MODEL_DIR, f"team_rf_event_{int(ev)}.pkl"))
        log_metrics_to_db(f"team_rf_event_{int(ev)}", m["avg_top1"], m["avg_top2"], m["avg_macro_f1"], len(m["classes"]))

    print("\nSaved global model →", GLOBAL_MODEL_PATH)
    print("Saved features    →", FEATURES_PATH)
    print("Per-event models (if any) saved in /models")
    print("Done.")

if __name__ == "__main__":
    main()
