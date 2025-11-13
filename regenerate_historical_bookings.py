# regenerate_historical_bookings_deterministic.py
import csv, os, random
from datetime import datetime
from collections import defaultdict, Counter

CSV_NAME = "historical_bookings.csv"
BACKUP_WITH_TIMESTAMP = True

EVENT_TYPES = [0, 1, 2, 3, 4]  # your form's event_type values
TEAMS = [1, 2, 3, 4, 5]

SAMPLES_PER_EVENT = 1600   #(total 8,000)
TARGET_PER_TEAM   = SAMPLES_PER_EVENT // len(TEAMS)

# Enumerations (int codes only; matches your form)
RANGES = {
    "hair_style":        (0, 4),
    "makeup_style":      (0, 5),
    "price_range":       (0, 2),
    "skin_tone":         (0, 3),
    "face_shape":        (0, 5),
    "gender_preference": (0, 2),  # 0=None, 1=Male, 2=Female
    "hair_length":       (0, 2),
}

RANDOM_SEED = 424242
random.seed(RANDOM_SEED)

def sample_features_for_event(ev):
    return {
        "hair_style":        random.randint(*RANGES["hair_style"]),
        "makeup_style":      random.randint(*RANGES["makeup_style"]),
        "price_range":       random.randint(*RANGES["price_range"]),
        "event_type":        ev,
        "skin_tone":         random.randint(*RANGES["skin_tone"]),
        "face_shape":        random.randint(*RANGES["face_shape"]),
        "gender_preference": random.randint(*RANGES["gender_preference"]),
        "hair_length":       random.randint(*RANGES["hair_length"]),
    }

def tuple_key(r):
    return (
        r["hair_style"], r["makeup_style"], r["price_range"], r["event_type"],
        r["skin_tone"], r["face_shape"], r["gender_preference"], r["hair_length"]
    )

# Strong, learnable per-event weights (only existing features; no new fields)
# Feature order used for scoring below:
# [gender_pref, hair_style, makeup_style, price_range, face_shape, skin_tone, hair_length, BIAS]
def strong_weights_for_event(ev):
    base = [
        [16, 12,  4, 10,  4,  3,  2, 10],  # team 1
        [ 4, 16, 12,  4, 10,  3,  2, 10],  # team 2
        [10,  4, 16,  4,  4, 12,  2, 10],  # team 3
        [ 4, 12,  4, 16, 12,  4,  2, 10],  # team 4
        [12,  4, 10, 10,  4, 16,  2, 10],  # team 5
    ]
    tweaked = []
    for i, row in enumerate(base):
        row2 = row[:]
        row2[(i + ev) % 7] += 5
        row2[-1] += 3 * ev
        tweaked.append(row2)
    if ev == 3:
        boost = [3, 3, 3, 5, 4, 3, 1, 3]
        tweaked = [[w + b for w, b in zip(wt, boost)] for wt in tweaked]
    return tweaked



def predict_team_by_rule(row, weights):
    feats = [
        row["gender_preference"],
        row["hair_style"],
        row["makeup_style"],
        row["price_range"],
        row["face_shape"],
        row["skin_tone"],
        row["hair_length"],
        1,  # bias term
    ]
    best_team, best_score = None, None
    for t_idx, w in enumerate(weights):  # 0..4 => team_id 1..5
        score = sum(fi * wi for fi, wi in zip(feats, w))
        if best_score is None or score > best_score:
            best_score = score
            best_team = TEAMS[t_idx]
    return best_team

def backup_csv(path):
    if not os.path.exists(path): return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{os.path.splitext(path)[0]}.backup_{ts}.csv"
    os.replace(path, backup_name)
    return backup_name

def write_csv(path, rows):
    fields = ["hair_style","makeup_style","price_range","event_type",
              "skin_tone","face_shape","gender_preference","hair_length","team_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def verify_no_conflicts(rows):
    mp = defaultdict(set)
    for r in rows:
        mp[tuple_key(r)].add(r["team_id"])
    return sum(1 for s in mp.values() if len(s) > 1)

def per_event_distribution(rows):
    dist = defaultdict(Counter)
    for r in rows:
        dist[r["event_type"]][r["team_id"]] += 1
    return dist

def main():
    rows = []
    per_event_map = {ev: {} for ev in EVENT_TYPES}  # tuple->team (per event)
    quotas = {ev: {t: TARGET_PER_TEAM for t in TEAMS} for ev in EVENT_TYPES}

    for ev in EVENT_TYPES:
        W = strong_weights_for_event(ev)
        need, tries, max_tries = SAMPLES_PER_EVENT, 0, SAMPLES_PER_EVENT*200
        while need > 0 and tries < max_tries:
            tries += 1
            row = sample_features_for_event(ev)
            key = tuple_key(row)

            if key in per_event_map[ev]:
                team = per_event_map[ev][key]
                if quotas[ev][team] > 0:
                    row["team_id"] = team
                    quotas[ev][team] -= 1
                    rows.append(row); need -= 1
                else:
                    continue
            else:
                team = predict_team_by_rule(row, W)
                if quotas[ev][team] == 0:
                    # pick team with max remaining quota (tie -> smaller id)
                    avail = [(t,q) for t,q in quotas[ev].items() if q>0]
                    if not avail: break
                    avail.sort(key=lambda x:(-x[1], x[0]))
                    team = avail[0][0]
                per_event_map[ev][key] = team
                row["team_id"] = team
                quotas[ev][team] -= 1
                rows.append(row); need -= 1

    # Trim if somehow over
    target = SAMPLES_PER_EVENT * len(EVENT_TYPES)
    if len(rows) > target:
        rows = rows[:target]

    conflicts = verify_no_conflicts(rows)
    print("Conflicts (should be 0):", conflicts)
    dist = per_event_distribution(rows)
    for ev in sorted(dist.keys()):
        print(f"[event_type={ev}] dist:", dict(dist[ev]))

    b = backup_csv(CSV_NAME)
    print(f"Backed up to: {b}" if b else "No old CSV to back up.")
    write_csv(CSV_NAME, rows)
    print(f"Wrote {len(rows)} rows to {CSV_NAME}")

if __name__ == "__main__":
    main()
