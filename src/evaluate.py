import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
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


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODELS_DIR / "model.ubj")
    return model


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Average positions off per driver"""
    return mean_absolute_error(y_true, y_pred)


def spearman(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Correlation between predicted and actual final positions 1.0 = perfect order, 0 = random, -1 = perfectly backwards"""
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def podium_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """% of podium predicted correctly"""
    true_podium = set(y_true.nsmallest(3).index)
    pred_podium = set(pd.Series(y_pred, index=y_true.index).nsmallest(3).index)
    return len(true_podium & pred_podium) / 3


def evaluate_race(y_true: pd.Series, y_pred: np.ndarray, race_name: str = ""):
    print(f"\n{'='*40}")
    if race_name:
        print(f"Race: {race_name}")
    print(f"MAE:              {mae(y_true, y_pred):.2f} positions")
    print(f"Spearman r:       {spearman(y_true, y_pred):.3f}")
    print(f"Podium accuracy:  {podium_accuracy(y_true, y_pred):.0%}")


def evaluate_all(X_test: pd.DataFrame, y_test: pd.Series, model: xgb.XGBRegressor):
    y_pred = model.predict(X_test)

    print("\n=== Overall Test Set ===")
    print(f"MAE:              {mae(y_test, y_pred):.2f} positions")
    print(f"Spearman r:       {spearman(y_test, y_pred):.3f}")

    plot_feature_importance(model)
    return y_pred


def plot_feature_importance(model: xgb.XGBRegressor):
    importance = model.get_booster().get_score(importance_type="gain")
    features = list(importance.keys())
    scores = list(importance.values())

    plt.figure(figsize=(8, 5))
    plt.barh(features, scores)
    plt.xlabel("Gain")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png")
    print(f"\nFeature importance plot saved to {MODELS_DIR / 'feature_importance.png'}")
