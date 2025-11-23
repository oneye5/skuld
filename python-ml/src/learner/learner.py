from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.utils.csv_utils import load_csv, save_csv
from src.config.config import *


# =======================================================
# === MODEL FACTORY ====================================
# =======================================================

def build_default_model() -> VotingClassifier:
    """
    Assemble the default ensemble model (soft voting).
    Easily extensible by changing the estimators list or adding logic here.
    """
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('et', ExtraTreesClassifier(random_state=42)),
        ('hgb', HistGradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
    ]
    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    return model


# =======================================================
# === TRAINING PIPELINE =================================
# =======================================================

def train_model(train_csv_path: str, model_save_path: str):
    """
    Train a model using the specified training set and save it to disk.
    """
    df = load_csv(train_csv_path)

    # Split into X, y
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    # Build model
    model = build_default_model()

    # Fit model
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


# =======================================================
# === PREDICTION PIPELINE ===============================
# =======================================================

def predict(model_path: str, input_csv_path: str, output_csv_path: str):
    """
    Load model and input data, generate ONLY probabilities, and save them.
    No discrete class labels are produced here.
    """
    # Load model
    model = joblib.load(model_path)

    # Load data
    df = load_csv(input_csv_path)

    # X only (drop label if present)
    X = df.drop(columns=[LABEL_COL]) if LABEL_COL in df.columns else df.copy()

    # Predict probabilities (only class 1 probability)
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    save_csv(df, output_csv_path)
    print(f"Probability predictions saved to {output_csv_path}")



# =======================================================
# === ENTRYPOINT ========================================
# =======================================================

if __name__ == "__main__":
    print("Training model...")
    train_model(str(TRAIN_CSV_PATH), str(MODEL_PKL_PATH))

    print("Generating predictions...")
    predict(str(MODEL_PKL_PATH), str(TEST_CSV_PATH), str(PREDICTION_CSV_PATH))
