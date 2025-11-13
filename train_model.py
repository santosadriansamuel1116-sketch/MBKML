import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load historical data
data = pd.read_csv("historical_bookings.csv")

# Define features and label
features = [
    "hair_style", "makeup_style", "price_range", "event_type",
    "skin_tone", "face_shape", "gender_preference", "hair_length"
]
X = data[features]
y = data["team_id"]

# Split for evaluation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with balanced class weight
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional but helpful)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "team_recommender_model.pkl")
print("✅ Model saved as team_recommender_model.pkl")
