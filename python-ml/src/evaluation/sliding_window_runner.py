
from src.preprocessing.long_to_wide_csv import long_to_wide_and_impute
from src.preprocessing.preprocessing import preprocess, remove_unlabeled
from src.preprocessing.train_test_split import split_and_save
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root
from src.config.config import *
from src.learner.learner import train_model, predict

def run():
    long_to_wide_and_impute(str(LONG_CSV_PATH), str(WIDE_CSV_PATH))
    preprocess(str(WIDE_CSV_PATH), str(PREPROCESSED_CSV_PATH))
    remove_unlabeled(str(PREPROCESSED_CSV_PATH), str(PREPROCESSED_CSV_PATH))

    full_df = load_csv(str(PREPROCESSED_CSV_PATH))
    data_end_ts = full_df[TIMESTAMP_COL].max()

    print(f"Raw Data End: {load_csv(str(LONG_CSV_PATH))[TIMESTAMP_COL].max()}")
    print(f"Preprocessed Data End (Anchor): {data_end_ts}")

    for i in range(0, EVAL_TEST_ITERATIONS):
        run_iteration(i, data_end_ts)

def run_iteration(i, anchor_ts):
    to_ts = anchor_ts - TEST_SPLIT_DURATION_MILLIS * i
    from_ts = to_ts - TEST_SPLIT_DURATION_MILLIS
    split_and_save(str(PREPROCESSED_CSV_PATH), from_ts, to_ts)
    train_model(str(TRAIN_CSV_PATH), str(MODEL_PKL_PATH))
    predict(str(MODEL_PKL_PATH), str(TEST_CSV_PATH), str(PREDICTION_CSV_PATH) + str(i) + ".csv")

if __name__ == "__main__":
    run()
