from src.config.config import PREPROCESSED_CSV_PATH, LONG_CSV_PATH, WIDE_CSV_PATH
from src.learner.learner import predict, train_model
from src.preprocessing.data_stripper import strip_data
from src.preprocessing.long_to_wide_csv import long_to_wide_and_impute
from src.preprocessing.preprocessing import preprocess
from src.preprocessing.train_test_split import *
from src.utils.csv_utils import load_csv
from src.preprocessing.preprocessing import *

def run():
    long_to_wide_and_impute(LONG_CSV_PATH, WIDE_CSV_PATH)
    preprocess(WIDE_CSV_PATH, PREPROCESSED_CSV_PATH)
    split_last_occurring_tickers(PREPROCESSED_CSV_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH)
    train_model(TRAIN_CSV_PATH, MODEL_PKL_PATH)
    predict(MODEL_PKL_PATH, TEST_CSV_PATH, PREDICTION_CSV_PATH)
    restore_ticker_delete_one_hot_and_save(PREDICTION_CSV_PATH, PREDICTION_CSV_PATH)
    strip_data(FINAL_PREDICTIONS_CSV_PATH, FINAL_PREDICTIONS_CSV_PATH)

if __name__ == "__main__":
    run()