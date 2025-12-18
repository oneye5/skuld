"""Quick test script to verify pipeline components work correctly."""

import pandas as pd
import numpy as np

# Setup paths
import sys
from pathlib import Path

ml_pipeline = Path(__file__).parent
sys.path.insert(0, str(ml_pipeline))
sys.path.insert(0, str(ml_pipeline / "data-preparation" / "transformations"))
sys.path.insert(0, str(ml_pipeline / "data-preparation" / "long-to-wide"))
sys.path.insert(0, str(ml_pipeline / "data-preparation" / "data-splitting" / "train-test"))
sys.path.insert(0, str(ml_pipeline / "data-preparation" / "labeling"))

from config.column_names import TIMESTAMP, TICKER, FEATURE, VALUE, CLOSE, OPEN, TARGET
from config.model_config import MS_PER_DAY

# Import modules
from macro_prefix import add_macro_prefix
from converter import long_to_wide
from splitter import split_by_timestamp
from labeler import create_labels
from imputation import compute_imputation_stats, impute_data
from feature_engineering import add_cyclical_time_features
from scaling import fit_scalers, transform_data
from learner.trainer import train_model
from learner.predictor import predict

print("Creating synthetic test data...")

# Create synthetic data - 3 years of daily data for 3 tickers
dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
timestamps = (dates.astype(int) // 10**6).astype(int)  # Convert to ms

rows = []
for ticker in ["ANZ.NZ", "BNZ.NZ", "AIR.NZ"]:
    base_price = np.random.uniform(10, 100)
    prices = base_price * np.exp(np.cumsum(np.random.normal(0, 0.01, len(timestamps))))
    
    for ts, price in zip(timestamps, prices):
        rows.append({TIMESTAMP: ts, TICKER: ticker, FEATURE: CLOSE, VALUE: price})
        rows.append({TIMESTAMP: ts, TICKER: ticker, FEATURE: OPEN, VALUE: price * 0.99})
        rows.append({TIMESTAMP: ts, TICKER: ticker, FEATURE: "Volume", VALUE: np.random.randint(1000, 10000)})

# Add macro data
for ts in timestamps[::30]:  # Monthly macro data
    rows.append({TIMESTAMP: ts, TICKER: "", FEATURE: "GDP", VALUE: np.random.uniform(100, 200)})

long_df = pd.DataFrame(rows)
print(f"Created {len(long_df):,} rows of synthetic data")

# Test macro prefix
print("\n1. Adding macro prefix...")
df = add_macro_prefix(long_df)
macro_count = df[FEATURE].str.startswith("MACRO_").sum()
print(f"   Macro features: {macro_count}")

# Test long to wide
print("\n2. Converting long to wide...")
wide_df = long_to_wide(df)
print(f"   Wide shape: {wide_df.shape}")

# Test train/test split
print("\n3. Splitting train/test...")
train_end = timestamps[int(len(timestamps) * 0.7)]
split = split_by_timestamp(wide_df, train_end)
print(f"   Train: {len(split.train)}, Test: {len(split.test)}")

# Test labeling
print("\n4. Creating labels...")
train_labeled = create_labels(split.train, lookahead_days=30, gain_threshold_pct=1.0)
test_labeled = create_labels(split.test, lookahead_days=30, gain_threshold_pct=1.0)
print(f"   Train labeled: {len(train_labeled)}, Test labeled: {len(test_labeled)}")
if len(train_labeled) > 0:
    print(f"   Label distribution: {train_labeled[TARGET].value_counts().to_dict()}")

# Test imputation
print("\n5. Imputation...")
stats = compute_imputation_stats(train_labeled)
train_imputed = impute_data(train_labeled, stats, add_indicators=True)
test_imputed = impute_data(test_labeled, stats, add_indicators=True)
print(f"   Train columns: {len(train_imputed.columns)}")

# Test feature engineering
print("\n6. Feature engineering...")
train_features = add_cyclical_time_features(train_imputed)
test_features = add_cyclical_time_features(test_imputed)
print(f"   Train columns after features: {len(train_features.columns)}")

# Test scaling
print("\n7. Scaling...")
scaler_set = fit_scalers(train_features)
train_scaled = transform_data(train_features, scaler_set)
test_scaled = transform_data(test_features, scaler_set)
print(f"   Scaled train shape: {train_scaled.shape}")

# Test training
print("\n8. Training model...")
model, feature_cols = train_model(train_scaled)
print(f"   Model trained with {len(feature_cols)} features")

# Test prediction
print("\n9. Making predictions...")
predictions = predict(model, test_scaled, feature_cols)
print(f"   Predictions: {len(predictions)}")
print(f"   Prediction range: {predictions['prediction_probability'].min():.3f} - {predictions['prediction_probability'].max():.3f}")

print("\n✓ All pipeline components working correctly!")
