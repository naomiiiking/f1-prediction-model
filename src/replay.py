import time
import pandas as pd
from src.fetch import fetch_session_data, fetch_qualifying_positions, fetch_qualifying_time
from src.features import build_lap_snapshot
from src.predict import load_artifacts, predict_from_snapshot


def run_replay(session_key: int, meeting_key: int, circuit: str, delay: float = 1.0):
    session = fetch_session_data(session_key, "Race")
    session["meta"] = {
        "session_key": session_key,
        "meeting_key": meeting_key,
        "circuit_short_name": circuit,
        "circuit_key": -1,
        "year": 2026,
        "session_name": "Race",
    }
    session["qualifying"] = fetch_qualifying_positions(meeting_key, "Race")
    session["starting_grid"] = fetch_qualifying_time(meeting_key, "Race")

    laps_df = pd.DataFrame(session["laps"])
    if laps_df.empty:
        print("No lap data found.")
        return

    max_lap = int(laps_df["lap_number"].max())
    model, team_enc, driver_enc = load_artifacts()

    for lap in range(1, max_lap + 1):
        snapshot = build_lap_snapshot(session, lap)
        result = predict_from_snapshot(snapshot, model, team_enc, driver_enc)

        if result.empty:
            continue

        print(f"\n=== Lap {lap}/{max_lap} — {circuit} ===")
        print(result.to_string(index=False))
        time.sleep(delay)


if __name__ == "__main__":
    run_replay(session_key=11280, meeting_key=1284, circuit="Miami", delay=1.0)
