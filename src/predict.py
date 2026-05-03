import time
import pandas as pd
import joblib
import xgboost as xgb
from config import MODELS_DIR, FEATURE_COLUMNS, PRE_RACE_FEATURE_COLUMNS


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


def predict_pre_race():
    from src.fetch import fetch_live
    from src.features import get_grid_positions, get_qualifying_time

    sessions = fetch_live("sessions", {"session_type": "Qualifying", "session_key": "latest"})
    if not sessions:
        print("No qualifying session found.")
        return

    session_meta = sessions[0]
    qual_key = session_meta["session_key"]
    circuit = session_meta.get("circuit_short_name", "unknown")
    circuit_key = session_meta.get("circuit_key", -1)
    print(f"Pre-race prediction: {circuit} (key={qual_key})")

    drivers_raw = fetch_live("drivers", {"session_key": qual_key})
    quali_positions = fetch_live("position", {"session_key": qual_key})
    starting_grid = fetch_live("starting_grid", {"session_key": qual_key})

    if not drivers_raw:
        print("No driver data found.")
        return

    drivers = pd.DataFrame(drivers_raw)[["driver_number", "full_name", "team_name"]].copy()
    drivers["driver_number"] = drivers["driver_number"].astype(int)
    drivers["grid_position"] = drivers["driver_number"].map(get_grid_positions(quali_positions))
    drivers["quali_lap_time"] = drivers["driver_number"].map(get_qualifying_time(starting_grid))
    drivers["circuit_key"] = circuit_key

    model = xgb.XGBRegressor()
    model.load_model(MODELS_DIR / "model_prerace.ubj")
    team_enc = joblib.load(MODELS_DIR / "team_encoder.pkl")
    driver_enc = joblib.load(MODELS_DIR / "driver_encoder.pkl")

    drivers = encode(drivers, team_enc, driver_enc)
    drivers = drivers.dropna(subset=PRE_RACE_FEATURE_COLUMNS)
    if drivers.empty:
        print("Not enough data for prediction.")
        return

    drivers["predicted_score"] = model.predict(drivers[PRE_RACE_FEATURE_COLUMNS])
    drivers["predicted_position"] = drivers["predicted_score"].rank().astype(int)

    result = drivers[["predicted_position", "full_name", "team_name", "grid_position"]].sort_values("predicted_position")
    print(f"\n=== Pre-race prediction: {circuit} ===")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    import sys
    if "--pre-race" in sys.argv:
        predict_pre_race()
    else:
        run_live()
