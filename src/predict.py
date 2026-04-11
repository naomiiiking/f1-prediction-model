import pandas as pd
import joblib
import xgboost as xgb
from config import MODELS_DIR

FEATURE_COLUMNS = [
    "grid_position",
    "pit_stop_count",
    "avg_lap_time",
    "best_lap_time",
    "lap_time_delta",
    "lap_count",
    "positions_gained",
    "circuit_key",
    "team_encoded",
    "driver_encoded",
]


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


def predict(session: dict) -> pd.DataFrame:
    from src.features import build_race_features

    model, team_enc, driver_enc = load_artifacts()

    df = build_race_features(session)
    if df.empty:
        raise ValueError("Could not build features for this session.")

    df = encode(df, team_enc, driver_enc)
    df = df.dropna(subset=FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS]
    raw_scores = model.predict(X)

    df["predicted_score"] = raw_scores
    df["predicted_position"] = df["predicted_score"].rank().astype(int)

    result = df[["full_name", "team_name", "predicted_position"]].sort_values("predicted_position")
    return result


if __name__ == "__main__":
    from src.fetch import fetch_all

    print("Fetching 2026 sessions...")
    all_sessions = fetch_all([2026])
    latest_session = next(
        s for s in sorted(all_sessions.values(), key=lambda s: s["meta"]["session_key"], reverse=True)
        if s.get("position")
    )

    circuit = latest_session["meta"].get("circuit_short_name", "unknown")
    print(f"\nPredicting: {circuit} {latest_session['meta'].get('year', 2026)}")
    result = predict(latest_session)
    print(result.to_string(index=False))
