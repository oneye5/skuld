from typing import Union, Tuple, Dict, List
from src.evaluation.ml_evaluation import calculate_ml_metrics
from src.evaluation.trading_evaluation import simulate_trades
from src.evaluation.utils import *
from src.preprocessing.preprocessing import restore_ticker_column
from src.utils.csv_utils import load_csv, save_csv

# =======================================================
# === MAIN EVALUATION FUNCTION ==========================
# =======================================================

def run_evaluation(
        predictions: Union[str, pd.DataFrame],
        labeled_csv_path: str,
        raw_price_csv_path: str,
        probability_threshold: float
):
    # Load predictions
    if isinstance(predictions, (str, Path)):
        preds_df = load_csv(str(predictions))
    elif isinstance(predictions, pd.DataFrame):
        preds_df = predictions
    else:
        raise ValueError("predictions must be a file path or DataFrame")

    if preds_df.empty:
        print("Warning: Predictions DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()

    print("Loading preprocessed data...")
    preprocessed_data = load_csv(labeled_csv_path)
    preprocessed_data = restore_ticker_column(preprocessed_data)

    print("Using preprocessed data for simulation...")

    trade_results = simulate_trades(preds_df, preprocessed_data, probability_threshold)
    ml_metrics = calculate_ml_metrics(preds_df, probability_threshold)

    return trade_results, ml_metrics

# =======================================================
# === MAIN EXECUTION ====================================
# =======================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRADING MODEL EVALUATION")
    print("=" * 60)

    print("\nLoading and combining prediction files...")
    combined_preds = load_combined_predictions(
        PY_DATA_DIR_PATH,
        pattern="predictions*.csv"
    )

    if combined_preds.empty:
        print("Error: No predictions to evaluate.")
    else:
        print(f"\nTotal predictions loaded: {len(combined_preds):,}")

        # Save combined predictions
        save_csv(combined_preds, str(AGGREGATE_PREDICTIONS_CSV_PATH))
        print(f"Saved combined predictions to: {AGGREGATE_PREDICTIONS_CSV_PATH}")

        # Run evaluation
        trades, metrics = run_evaluation(
            combined_preds,
            str(PREPROCESSED_CSV_PATH),
            str(PREPROCESSED_CSV_PATH),
            EVAL_CLASSIFICATION_BOUNDARY
        )

        # Save results
        if not trades.empty:
            save_csv(trades, str(TRADE_SIMULATION_CSV_PATH))
            print(f"\nSaved trade results to: {TRADE_SIMULATION_CSV_PATH}")

        if not metrics.empty:
            save_csv(metrics, str(EVALUATION_RESULTS_CSV_PATH))
            print(f"Saved metrics to: {EVALUATION_RESULTS_CSV_PATH}")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)