import csv
import random

# Define label options
options = {
    "hair_style": list(range(6)),            # 0 to 5
    "makeup_style": list(range(6)),          # 0 to 5
    "price_range": list(range(3)),           # 0 to 2 (Low, Med, High)
    "event_type": list(range(6)),            # 0 to 5
    "skin_tone": list(range(4)),             # 0 to 3
    "face_shape": list(range(6)),            # 0 to 5
    "gender_preference": list(range(3)),     # 0 to 2 (No pref, Male, Female)
    "hair_length": list(range(3)),           # 0 to 2
    "team_id": [1, 2, 3, 4, 5]
}

# Load existing data
with open("historical_bookings.csv", mode="r", newline='') as infile:
    reader = list(csv.reader(infile))
    header = reader[0]
    existing_data = reader[1:]

# Generate 50 new rows
new_rows = []
for _ in range(50):
    row = [
        random.choice(options["hair_style"]),
        random.choice(options["makeup_style"]),
        random.choice(options["price_range"]),
        random.choice(options["event_type"]),
        random.choice(options["skin_tone"]),
        random.choice(options["face_shape"]),
        random.choice(options["gender_preference"]),
        random.choice(options["hair_length"]),
        random.choice(options["team_id"])
    ]
    new_rows.append(row)

# Append to file
with open("historical_bookings.csv", mode="a", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(new_rows)

print("✅ 50 new rows added to historical_bookings.csv")
