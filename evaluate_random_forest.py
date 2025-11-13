import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pymysql

# Connect to DB and fetch dataset
conn = pymysql.connect(host='localhost', user='root', password='', database='mbk_db')
query = """
SELECT bc.gender_preference, bc.hair_style, bc.makeup_style, bc.event_type,
       bc.price_range, bc.face_shape, bc.skin_tone, bc.hair_length, bc.team_id
FROM booking_clients bc
JOIN bookings b ON bc.booking_id = b.booking_id
WHERE bc.team_id IS NOT NULL
"""
df = pd.read_sql(query, conn)
conn.close()

# Clean/prepare data
df = df.fillna(0)
X = df.drop(columns=['team_id'])
y = df['team_id']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("✅ Model Evaluation")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
