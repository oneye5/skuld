import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import scale_data_with_scaler


def test_scaler_continuous_cols_persistence():
    # Training data: includes a binary column 'b' with values {0,1}
    df_train = pd.DataFrame({
        'timestamp': range(5),
        'a': np.arange(5).astype(float),
        'b': [0, 1, 0, 1, 0],
        'c': [10.0, 20.0, 30.0, 40.0, 50.0],
        'label': [0, 1, 0, 1, 0],
    })

    # Fit scaler on training data and capture continuous columns
    df_train_scaled, scaler, continuous_cols = scale_data_with_scaler(
        df_train, scaler=None, fit_scaler=True
    )

    # Test data: the binary column 'b' is now all zeros (nunique == 1)
    df_test = pd.DataFrame({
        'timestamp': range(3),
        'a': [5.0, 6.0, 7.0],
        'b': [0, 0, 0],
        'c': [15.0, 25.0, 35.0],
        'label': [0, 1, 0],
    })

    # Ensure same column order as train (as post-split alignment would do)
    df_test = df_test[df_train.columns]

    # Apply scaler using the continuous_cols captured from training
    df_test_scaled, _, _ = scale_data_with_scaler(
        df_test, scaler=scaler, fit_scaler=False, continuous_cols=continuous_cols
    )

    # Check that scaling was applied to numeric continuous column 'a'
    assert 'a' in df_test_scaled.columns
    assert not np.allclose(df_test_scaled['a'].values, df_test['a'].values)
