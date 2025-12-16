"""Prediction pipeline with proper scaling to prevent data leakage."""
from pathlib import Path

from src.config.config import (
    PREPROCESSED_CSV_PATH, LONG_CSV_PATH, WIDE_CSV_PATH, TRAIN_CSV_PATH,
    TEST_CSV_PATH, MODEL_PKL_PATH, PREDICTION_CSV_PATH, FINAL_PREDICTIONS_CSV_PATH,
    PY_DATA_DIR_PATH
)
from src.learner.learner import predict, train_model
from src.preprocessing.data_stripper import strip_data
from src.preprocessing.long_to_wide_csv import long_to_wide_and_impute
from src.preprocessing.pre_split_preprocessing import pre_split_preprocess, restore_ticker_delete_one_hot_and_save
from src.preprocessing.train_test_split import split_last_occurring_tickers
from src.preprocessing.post_split_preprocessing import post_split_preprocessing_train, post_split_preprocessing_test


def run():
    """
    Execute the complete prediction pipeline with proper data handling.
    
    Steps:
    1. Convert long to wide format with imputation
    2. Pre-split preprocessing (labeling, encoding, feature selection)
    3. Split data into training and test (last rows)
    4. Scale training and test data separately to prevent leakage
    5. Train model on scaled training data
    6. Generate predictions on scaled test data
    7. Restore ticker information and finalize predictions
    """
    long_to_wide_and_impute(LONG_CSV_PATH, WIDE_CSV_PATH)
    pre_split_preprocess(WIDE_CSV_PATH, PREPROCESSED_CSV_PATH)
    split_last_occurring_tickers(PREPROCESSED_CSV_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH)
    # Scale data separately to prevent leakage
    scaler_path = str(Path(PY_DATA_DIR_PATH) / "scaler_final.pkl")
    train_scaled_path = str(TRAIN_CSV_PATH).replace('.csv', '_scaled.csv')
    test_scaled_path = str(TEST_CSV_PATH).replace('.csv', '_scaled.csv')
    # Fit scaler on training data and scale it
    post_split_preprocessing_train(str(TRAIN_CSV_PATH), train_scaled_path, scaler_path)
    # Apply training scaler to test data
    post_split_preprocessing_test(str(TEST_CSV_PATH), test_scaled_path, scaler_path)
    # Train and predict using scaled data
    train_model(train_scaled_path, str(MODEL_PKL_PATH))
    predict(str(MODEL_PKL_PATH), test_scaled_path, str(PREDICTION_CSV_PATH))
    restore_ticker_delete_one_hot_and_save(str(PREDICTION_CSV_PATH), str(FINAL_PREDICTIONS_CSV_PATH))
    strip_data(str(FINAL_PREDICTIONS_CSV_PATH), str(FINAL_PREDICTIONS_CSV_PATH))


if __name__ == "__main__":
    run()
