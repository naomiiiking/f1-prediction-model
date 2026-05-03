from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

OPENF1_BASE_URL = "https://api.openf1.org/v1"

SEASONS = [2024, 2025, 2026]

# How much to weight each season's races during training
# Higher = model pays more attention to that season
SEASON_WEIGHTS = {2024: 1.0, 2025: 1.5, 2026: 3.0}

HF_REPO_ID = "username/f1-race-predictor"  # TODO: update this later

PRE_RACE_FEATURE_COLUMNS = [
    "grid_position",
    "quali_lap_time",
    "circuit_key",
    "team_encoded",
    "driver_encoded",
]

FEATURE_COLUMNS = [
    "grid_position",
    "quali_lap_time",
    "current_lap",
    "current_position",
    "pit_stop_count",
    "avg_lap_time",
    "best_lap_time",
    "lap_time_delta",
    "lap_count",
    "circuit_key",
    "team_encoded",
    "driver_encoded",
]

TARGET_COLUMN = "final_position"
RANDOM_SEED = 42 # Reused training seed
