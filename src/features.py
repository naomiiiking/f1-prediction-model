import pandas as pd
from tqdm import tqdm
from config import DATA_PROCESSED


def get_final_positions(position_data: list[dict]) -> dict[int, int]:
    df = pd.DataFrame(position_data)
    if df.empty:
        return {}
    df = df.sort_values("date")
    return df.groupby("driver_number")["position"].last().to_dict()


def get_grid_positions(qualifying_data: list[dict]) -> dict[int, int]:
    df = pd.DataFrame(qualifying_data)
    if df.empty:
        return {}
    df = df.sort_values("date")
    return df.groupby("driver_number")["position"].last().to_dict()


def get_qualifying_time(starting_grid: list[dict]) -> dict[int, float]:
    df = pd.DataFrame(starting_grid)
    if df.empty:
        return {}
    return df.set_index("driver_number")["lap_duration"].to_dict()


def get_pit_counts_up_to(pit_data: list[dict], up_to_lap: int) -> dict[int, int]:
    df = pd.DataFrame(pit_data)
    if df.empty:
        return {}
    if "lap_number" in df.columns:
        df = df[df["lap_number"] <= up_to_lap]
    return df.groupby("driver_number").size().to_dict()


def get_lap_stats_up_to(lap_data: list[dict], up_to_lap: int) -> pd.DataFrame:
    df = pd.DataFrame(lap_data)
    if df.empty:
        return pd.DataFrame()
    df = df[df["lap_number"] <= up_to_lap]
    df = df[df["lap_duration"].notna()]
    if df.empty:
        return pd.DataFrame()
    stats = df.groupby("driver_number")["lap_duration"].agg(
        avg_lap_time="mean",
        best_lap_time="min",
        lap_count="count",
    ).reset_index()
    return stats


def get_position_at_lap(position_data: list[dict], laps_data: list[dict], up_to_lap: int) -> dict[int, int]:
    pos_df = pd.DataFrame(position_data)
    laps_df = pd.DataFrame(laps_data)

    if pos_df.empty or laps_df.empty:
        return {}

    pos_df["date"] = pd.to_datetime(pos_df["date"], utc=True)
    laps_df["date_start"] = pd.to_datetime(laps_df["date_start"], utc=True)

    next_lap_rows = laps_df[laps_df["lap_number"] == up_to_lap + 1]
    if not next_lap_rows.empty:
        cutoff = next_lap_rows["date_start"].min()
    else:
        cutoff = pos_df["date"].max()

    filtered = pos_df[pos_df["date"] <= cutoff].sort_values("date")
    if filtered.empty:
        return {}

    return filtered.groupby("driver_number")["position"].last().to_dict()


def build_lap_snapshot(session: dict, up_to_lap: int) -> pd.DataFrame:
    final_positions = get_final_positions(session.get("position", []))
    grid_positions = get_grid_positions(session.get("qualifying", []))
    quali_time = get_qualifying_time(session.get("starting_grid", []))
    current_positions = get_position_at_lap(
        session.get("position", []),
        session.get("laps", []),
        up_to_lap,
    )
    pit_counts = get_pit_counts_up_to(session.get("pit", []), up_to_lap)
    lap_stats = get_lap_stats_up_to(session.get("laps", []), up_to_lap)

    drivers = pd.DataFrame(session.get("drivers", []))
    if drivers.empty:
        return pd.DataFrame()

    drivers = drivers[["driver_number", "full_name", "team_name"]].copy()
    drivers["driver_number"] = drivers["driver_number"].astype(int)

    drivers["final_position"] = drivers["driver_number"].map(final_positions)
    drivers["grid_position"] = drivers["driver_number"].map(grid_positions)
    drivers["quali_lap_time"] = drivers["driver_number"].map(quali_time)
    drivers["current_position"] = drivers["driver_number"].map(current_positions)
    drivers["pit_stop_count"] = drivers["driver_number"].map(pit_counts).fillna(0)
    drivers["current_lap"] = up_to_lap

    if not lap_stats.empty:
        lap_stats["driver_number"] = lap_stats["driver_number"].astype(int)
        drivers = drivers.merge(lap_stats, on="driver_number", how="left")

    meta = session.get("meta", {})
    drivers["circuit_key"] = meta.get("circuit_key", -1)
    drivers["circuit_short_name"] = meta.get("circuit_short_name", "unknown")
    drivers["session_key"] = meta.get("session_key", -1)
    drivers["year"] = meta.get("year", -1)

    drivers = drivers.dropna(subset=["grid_position"])

    if "best_lap_time" in drivers.columns:
        best_time = drivers["best_lap_time"].min()
        drivers["lap_time_delta"] = drivers["best_lap_time"] - best_time
    else:
        drivers["avg_lap_time"] = None
        drivers["best_lap_time"] = None
        drivers["lap_count"] = None
        drivers["lap_time_delta"] = None

    return drivers


def build_dataset(all_sessions: dict, lap_step: int = 1) -> pd.DataFrame:
    frames = []

    for session_key, session in tqdm(all_sessions.items(), desc="Sessions"):
        laps_df = pd.DataFrame(session.get("laps", []))
        if laps_df.empty:
            continue

        max_lap = int(laps_df["lap_number"].max())

        for lap in range(lap_step, max_lap + 1, lap_step):
            df = build_lap_snapshot(session, lap)
            if df.empty:
                continue
            df = df.dropna(subset=["final_position"])
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
