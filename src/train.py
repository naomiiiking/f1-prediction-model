import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import joblib
from config import DATA_PROCESSED, MODELS_DIR, TARGET_COLUMN, RANDOM_SEED, SEASON_WEIGHTS, SEASONS, FEATURE_COLUMNS, PRE_RACE_FEATURE_COLUMNS

def load_data() -> pd.DataFrame:
    """Load input data from config"""
    path = DATA_PROCESSED / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"No processed data at {path}. Run features.py first.")
    df = pd.read_csv(path)
    return df[df["year"].isin(SEASONS)]


def encode_categoricals(df: pd.DataFrame):
    """Encode the team and driver name for model to read"""
    df = df.copy()
    team_enc = LabelEncoder()
    driver_enc = LabelEncoder()
    df["team_encoded"] = team_enc.fit_transform(df["team_name"].fillna("unknown"))
    df["driver_encoded"] = driver_enc.fit_transform(df["full_name"].fillna("unknown"))
    return df, team_enc, driver_enc


def split_by_race(df: pd.DataFrame):
    """Create random test/train splits of different races"""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    groups = df["session_key"]
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


def train(df: pd.DataFrame):
    """Assembly function, trains model and stores"""
    df, team_enc, driver_enc = encode_categoricals(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    train_df, test_df = split_by_race(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric="mae",
    )

    sample_weights = train_df["year"].map(SEASON_WEIGHTS).fillna(1.0)

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / "model.ubj")
    joblib.dump(team_enc, MODELS_DIR / "team_encoder.pkl")
    joblib.dump(driver_enc, MODELS_DIR / "driver_encoder.pkl")

    print(f"\nModel saved to {MODELS_DIR}")
    return model, X_test, y_test


def train_pre_race(df: pd.DataFrame):
    df, _team_enc, _driver_enc = encode_categoricals(df)
    df = df.drop_duplicates(subset=["driver_number", "session_key"])
    df = df.dropna(subset=PRE_RACE_FEATURE_COLUMNS + [TARGET_COLUMN])

    train_df, test_df = split_by_race(df)

    X_train = train_df[PRE_RACE_FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[PRE_RACE_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric="mae",
    )

    sample_weights = train_df["year"].map(SEASON_WEIGHTS).fillna(1.0)

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / "model_prerace.ubj")
    print(f"\nPre-race model saved to {MODELS_DIR}")
    return model, X_test, y_test


if __name__ == "__main__":
    import sys
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['session_key'].nunique()} races")
    if "--pre-race" in sys.argv:
        train_pre_race(df)
    else:
        train(df)
