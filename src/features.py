import pandas as pd
from config import DATA_PROCESSED


def get_final_positions(position_data: list[dict]) -> dict[int, int]:
    """Return a dict of driver_number: final_position"""
    df = pd.DataFrame(position_data)
    if df.empty:
        return {}
    df = df.sort_values("date")
    return df.groupby("driver_number")["position"].last().to_dict()


def get_grid_positions(qualifying_data: list[dict]) -> dict[int, int]:
    """Return a dict of driver_number: quali position"""
    df = pd.DataFrame(qualifying_data)
    if df.empty:
        return {}
    df = df.sort_values("date")
    return df.groupby("driver_number")["position"].last().to_dict()


def get_pit_counts(pit_data: list[dict]) -> dict[int, int]:
    """Return a dict of driver_number: pit counts"""
    df = pd.DataFrame(pit_data)
    if df.empty:
        return {}
    return df.groupby("driver_number").size().to_dict()


def get_lap_stats(lap_data: list[dict]) -> pd.DataFrame:
    """Return a list of lap times in a race by driver"""
    df = pd.DataFrame(lap_data)
    if df.empty:
        return pd.DataFrame()
    df = df[df["lap_duration"].notna()] # Drops rows where lab_duration is missing (driver has retired)
    stats = df.groupby("driver_number")["lap_duration"].agg(
        avg_lap_time="mean",
        best_lap_time="min",
        lap_count="count",
    ).reset_index()
    return stats


def build_race_features(session: dict) -> pd.DataFrame:
    """Assembly function, creates race table with all driver records"""
    final_positions = get_final_positions(session.get("position", []))
    grid_positions = get_grid_positions(session.get("qualifying", []))
    pit_counts = get_pit_counts(session.get("pit", []))
    lap_stats = get_lap_stats(session.get("laps", []))

    drivers = pd.DataFrame(session.get("drivers", []))
    if drivers.empty:
        return pd.DataFrame()

    drivers = drivers[["driver_number", "full_name", "team_name"]].copy()
    drivers["driver_number"] = drivers["driver_number"].astype(int)

    drivers["final_position"] = drivers["driver_number"].map(final_positions)
    drivers["grid_position"] = drivers["driver_number"].map(grid_positions)
    drivers["pit_stop_count"] = drivers["driver_number"].map(pit_counts).fillna(0)

    if not lap_stats.empty:
        lap_stats["driver_number"] = lap_stats["driver_number"].astype(int)
        drivers = drivers.merge(lap_stats, on="driver_number", how="left")

    meta = session.get("meta", {})
    drivers["circuit_key"] = meta.get("circuit_key", -1)
    drivers["circuit_short_name"] = meta.get("circuit_short_name", "unknown")
    drivers["session_key"] = meta.get("session_key", -1)
    drivers["year"] = meta.get("year", -1)

    drivers = drivers.dropna(subset=["final_position", "grid_position"])
    drivers["final_position"] = drivers["final_position"].astype(int)
    drivers["grid_position"] = drivers["grid_position"].astype(int)
    drivers["positions_gained"] = drivers["grid_position"] - drivers["final_position"]

    if "best_lap_time" in drivers.columns:
        best_time = drivers["best_lap_time"].min()
        drivers["lap_time_delta"] = drivers["best_lap_time"] - best_time
    else:
        drivers["avg_lap_time"] = None
        drivers["best_lap_time"] = None
        drivers["lap_count"] = None
        drivers["lap_time_delta"] = None

    return drivers


def build_dataset(all_sessions: dict) -> pd.DataFrame:
    frames = []
    for session in all_sessions.values():
        df = build_race_features(session)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "features.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {len(dataset)} rows to {out_path}")
    return dataset
