import csv
import os
from datetime import datetime, timedelta

ATTENDANCE_FILE = "attendance.csv"
COOLDOWN_MINUTES = 10

def mark_attendance(name, confidence):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    # Create file if missing
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time", "Confidence"])

    # Read records safely
    with open(ATTENDANCE_FILE, "r") as f:
        reader = list(csv.DictReader(f))

    for row in reversed(reader):
        if row["Name"] != name:
            continue

        try:
            last_seen = datetime.strptime(
                f'{row["Date"]} {row["Time"]}',
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            # 🔥 Skip corrupted rows instead of crashing
            continue

        if now - last_seen < timedelta(minutes=COOLDOWN_MINUTES):
            return False

    # Append clean row
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            today,
            time_now,
            f"{confidence:.2f}"
        ])

    return True
