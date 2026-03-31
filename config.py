from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

OPENF1_BASE_URL = "https://api.openf1.org/v1"

SEASONS = [2024]

HF_REPO_ID = "username/f1-race-predictor"  # TODO: update this later

TARGET_COLUMN = "final_position"
RANDOM_SEED = 42
