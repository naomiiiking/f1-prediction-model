"""
Fetches data from the OpenF1 API and caches it locally as JSON.

OpenF1 docs: https://openf1.org
Key endpoints we use:
  /sessions  - race/qualifying session metadata
  /drivers   - driver info per session
  /position  - driver positions sampled throughout the race
  /laps      - per-lap timing data
  /pit       - pit stop events
  /weather   - weather snapshots during session
"""

import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

from config import OPENF1_BASE_URL, DATA_RAW, SEASONS


def get(endpoint: str, params: dict, retries: int = 8) -> list[dict]:
    """Make a single GET request to the OpenF1 API, retrying on 429s."""
    url = f"{OPENF1_BASE_URL}/{endpoint}"
    for attempt in range(retries):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 429:
            wait = min(2 ** attempt, 60)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def cache_path(name: str) -> Path:
    """Return a path for a cached JSON file, creating dirs if needed."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW / f"{name}.json"


def load_or_fetch(name: str, endpoint: str, params: dict) -> list[dict]:
    """Fetch or load data from cache/API"""
    path = cache_path(name)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    print(f"  Fetching {endpoint} ({params})...")
    data = get(endpoint, params)
    with open(path, "w") as f:
        json.dump(data, f)
    time.sleep(2.0)
    return data


def fetch_race_sessions(year: int) -> list[dict]:
    """Get all race sessions (not practice/qualifying/sprint) for a given year."""
    all_sessions = load_or_fetch(
        name=f"sessions_{year}",
        endpoint="sessions",
        params={"year": year, "session_type": "Race"},
    )
    return [s for s in all_sessions if s.get("session_name") == "Race"]


def fetch_session_data(session_key: int, session_name: str) -> dict:
    """Fetch all relevant data for a single race session."""
    key = session_key
    name = session_name.replace(" ", "_").lower()
    prefix = f"session_{key}_{name}"

    print(f"\nFetching data for: {session_name} (key={key})")

    endpoints = {
        "drivers": ("drivers", {"session_key": key}),
        "laps": ("laps", {"session_key": key}),
        "pit": ("pit", {"session_key": key}),
        "position": ("position", {"session_key": key}),
        "weather": ("weather", {"session_key": key}),
    }

    result = {}
    for data_type, (endpoint, params) in endpoints.items():
        result[data_type] = load_or_fetch(
            name=f"{prefix}_{data_type}",
            endpoint=endpoint,
            params=params,
        )

    return result

def fetch_qualifying_time(meeting_key: int, session_name: str) -> list[dict]:
    meeting_sessions = load_or_fetch(
        name=f"meeting_sessions_{meeting_key}",
        endpoint="sessions",
        params={"meeting_key": meeting_key},
    )

    qual_sessions = [s for s in meeting_sessions if s.get("session_type") == "Qualifying"]
    if not qual_sessions:
        return []

    qual_key = qual_sessions[0]["session_key"]
    name = session_name.replace(" ", "_").lower()

    return load_or_fetch(
        name=f"qualifying_{qual_key}_{name}_starting_grid",
        endpoint="starting_grid",
        params={"session_key": qual_key},
    )

def fetch_qualifying_positions(meeting_key: int, session_name: str) -> list[dict]:
    """Fetch the qualifying session that corresponds to a race weeken """
    meeting_sessions = load_or_fetch(
        name=f"meeting_sessions_{meeting_key}",
        endpoint="sessions",
        params={"meeting_key": meeting_key},
    )

    qual_sessions = [s for s in meeting_sessions if s.get("session_type") == "Qualifying"]
    if not qual_sessions:
        return []

    qual_key = qual_sessions[0]["session_key"]
    name = session_name.replace(" ", "_").lower()

    return load_or_fetch(
        name=f"qualifying_{qual_key}_{name}_position",
        endpoint="position",
        params={"session_key": qual_key},
    )


def fetch_all(seasons: list[int] = SEASONS) -> dict[int, dict]:
    """Assembly funciton, fetch everything we need for all seasons."""
    all_data = {}

    for year in seasons:
        print(f"\n=== Season {year} ===")
        sessions = fetch_race_sessions(year)
        print(f"Found {len(sessions)} race sessions")

        for session in tqdm(sessions, desc=f"{year} races"):
            key = session["session_key"]
            meeting_key = session.get("meeting_key", key)
            session_name = session.get("session_name", str(key))

            session_data = fetch_session_data(key, session_name)
            session_data["meta"] = session
            session_data["qualifying"] = fetch_qualifying_positions(meeting_key, session_name)
            session_data["starting_grid"] = fetch_qualifying_time(meeting_key, session_name)
            all_data[key] = session_data

    print(f"\nDone. Fetched {len(all_data)} sessions total.")
    return all_data


if __name__ == "__main__":
    fetch_all()
