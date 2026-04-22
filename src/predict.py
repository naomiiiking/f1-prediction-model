import time
import pandas as pd
import joblib
import xgboost as xgb
from config import MODELS_DIR, FEATURE_COLUMNS


def load_artifacts():
    model = xgb.XGBRegressor()
    model.load_model(MODELS_DIR / "model.ubj")
    team_enc = joblib.load(MODELS_DIR / "team_encoder.pkl")
    driver_enc = joblib.load(MODELS_DIR / "driver_encoder.pkl")
    return model, team_enc, driver_enc


def safe_encode(series: pd.Series, encoder) -> pd.Series:
    known = set(encoder.classes_)
    mapped = series.fillna("unknown").apply(lambda x: x if x in known else "unknown")
    return pd.Series(encoder.transform(mapped), index=series.index)


def encode(df: pd.DataFrame, team_enc, driver_enc) -> pd.DataFrame:
    df = df.copy()
    df["team_encoded"] = safe_encode(df["team_name"], team_enc)
    df["driver_encoded"] = safe_encode(df["full_name"], driver_enc)
    return df


def predict_from_snapshot(df: pd.DataFrame, model, team_enc, driver_enc) -> pd.DataFrame:
    df = encode(df, team_enc, driver_enc)
    df = df.dropna(subset=FEATURE_COLUMNS)
    if df.empty:
        return pd.DataFrame()

    X = df[FEATURE_COLUMNS]
    df = df.copy()
    df["predicted_score"] = model.predict(X)
    df["predicted_position"] = df["predicted_score"].rank().astype(int)

    return df[["full_name", "team_name", "current_position", "predicted_position"]].sort_values("predicted_position")


def run_live():
    from src.fetch import fetch_live, fetch_session_data, fetch_qualifying_positions, fetch_qualifying_time
    from src.features import build_lap_snapshot

    sessions = fetch_live("sessions", {"session_type": "Race", "session_key": "latest"})
    if not sessions:
        print("No live session found.")
        return

    session_meta = sessions[0]
    session_key = session_meta["session_key"]
    meeting_key = session_meta.get("meeting_key", session_key)
    session_name = session_meta.get("session_name", "Race")
    circuit = session_meta.get("circuit_short_name", "unknown")
    print(f"Live session: {circuit} (key={session_key})")

    session = fetch_session_data(session_key, session_name)
    session["meta"] = session_meta
    session["qualifying"] = fetch_qualifying_positions(meeting_key, session_name)
    session["starting_grid"] = fetch_qualifying_time(meeting_key, session_name)

    model, team_enc, driver_enc = load_artifacts()
    last_predicted_lap = 0

    while True:
        laps = fetch_live("laps", {"session_key": session_key})
        if not laps:
            print("Waiting for race to start...")
            time.sleep(30)
            continue

        laps_df = pd.DataFrame(laps)
        latest_lap = int(laps_df["lap_number"].max())

        if latest_lap <= last_predicted_lap:
            time.sleep(10)
            continue

        session["laps"] = laps
        session["position"] = fetch_live("position", {"session_key": session_key})
        session["pit"] = fetch_live("pit", {"session_key": session_key})

        snapshot = build_lap_snapshot(session, latest_lap)
        result = predict_from_snapshot(snapshot, model, team_enc, driver_enc)

        if result.empty:
            time.sleep(10)
            continue

        print(f"\n=== Lap {latest_lap} — {circuit} ===")
        print(result.to_string(index=False))
        last_predicted_lap = latest_lap
        time.sleep(10)


if __name__ == "__main__":
    run_live()
