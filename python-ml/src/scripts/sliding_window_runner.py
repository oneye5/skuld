"""Sliding window evaluation pipeline for backtesting the ML model."""
from pathlib import Path

from src.preprocessing.long_to_wide_csv import long_to_wide_and_impute
from src.preprocessing.pre_split_preprocessing import pre_split_preprocess, remove_unlabeled
from src.preprocessing.train_test_split import split_and_save
from src.preprocessing.post_split_preprocessing import post_split_preprocessing_train, post_split_preprocessing_test
from src.utils.clean_data_directory import clean_data_directory
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root
from src.config.config import *
from src.learner.learner import train_model, predict


def get_scaler_path_for_iteration(iteration: int) -> str:
    """Get the path for the scaler for a given iteration."""
    return str(Path(PY_DATA_DIR_PATH) / f"scaler_iteration_{iteration}.pkl")


def run() -> None:
    """
    Execute the complete sliding window evaluation pipeline.
    
    Steps:
    1. Clean previous data
    2. Convert long to wide format and impute missing values
    3. Apply pre-split preprocessing (labeling, encoding, feature selection)
    4. Remove rows with invalid labels
    5. Run multiple iterations of sliding window evaluation
    """
    clean_data_directory()
    long_to_wide_and_impute(str(LONG_CSV_PATH), str(WIDE_CSV_PATH))
    pre_split_preprocess(str(WIDE_CSV_PATH), str(PREPROCESSED_CSV_PATH))
    remove_unlabeled(str(PREPROCESSED_CSV_PATH), str(PREPROCESSED_CSV_PATH))

    try:
        full_df = load_csv(str(PREPROCESSED_CSV_PATH))
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    if full_df.empty:
        print("Error: Preprocessed data is empty")
        return

    data_end_ts = full_df[TIMESTAMP_COL].max()
    print(f"\nStarting {EVAL_TEST_ITERATIONS} sliding window iterations...")

    for i in range(0, EVAL_TEST_ITERATIONS):
        try:
            run_iteration(i, data_end_ts)
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue


def run_iteration(iteration: int, anchor_ts: int) -> None:
    """
    Execute a single sliding window evaluation iteration.
    
    Trains model on historical data and evaluates on future data.
    Scaling is applied AFTER train/test split using training data statistics.
    
    Args:
        iteration: Current iteration number (0-indexed).
        anchor_ts: Anchor timestamp (data end) for reference.
    """
    print(f"Iteration {iteration + 1}/{EVAL_TEST_ITERATIONS}...", end=' ')
    
    shift_ts = EVAL_TIME_PROGRESSION * -iteration
    to_ts = anchor_ts - TEST_SPLIT_DURATION_MILLIS * iteration + shift_ts
    from_ts = to_ts - TEST_SPLIT_DURATION_MILLIS + shift_ts
    
    try:
        # Split data BEFORE scaling to prevent leakage
        split_and_save(str(PREPROCESSED_CSV_PATH), from_ts, to_ts)
        
        # Scale training and test data separately
        scaler_path = get_scaler_path_for_iteration(iteration)
        train_scaled_path = str(TRAIN_CSV_PATH).replace('.csv', '_scaled.csv')
        test_scaled_path = str(TEST_CSV_PATH).replace('.csv', '_scaled.csv')
        
        # Fit scaler on training data and scale it
        post_split_preprocessing_train(str(TRAIN_CSV_PATH), train_scaled_path, scaler_path)
        
        # Apply training scaler to test data
        post_split_preprocessing_test(str(TEST_CSV_PATH), test_scaled_path, scaler_path)
        
        # Train and predict using scaled data
        train_model(train_scaled_path, str(MODEL_PKL_PATH))
        out_path = str(PREDICTION_CSV_PATH).replace(".csv", "") + str(iteration) + ".csv"
        predict(str(MODEL_PKL_PATH), test_scaled_path, out_path)
        
        print("✓")
        
    except Exception as e:
        print(f"Error during iteration {iteration}: {e}")
        raise


if __name__ == "__main__":
    run()

