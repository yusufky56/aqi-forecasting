"""Data preparation shared by training and evaluation.

Everything that learns from the data (outlier bounds, scalers) is fitted on
the training period only, so no information from the validation or test
period leaks into the model.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data")  # city_day.csv from the Kaggle dataset

CITY = "Delhi"
SEQ_LEN = 14
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler_X: RobustScaler
    scaler_y: MinMaxScaler
    features: list
    data: pd.DataFrame  # engineered features, unscaled


def split_points(n_rows, seq_len=SEQ_LEN):
    """Sequence indices where validation and test start, plus the number of raw
    rows the training sequences cover (inputs and targets)."""
    n_seq = n_rows - seq_len
    train_end = int(n_seq * TRAIN_FRAC)
    val_end = int(n_seq * (TRAIN_FRAC + VAL_FRAC))
    return train_end, val_end, train_end + seq_len


def create_sequences(X, y, seq_len=SEQ_LEN):
    """Window of `seq_len` days -> AQI of the following day."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys).reshape(-1)


def engineer_features(data):
    data = data.copy()
    for col in ['PM2.5', 'PM10', 'CO', 'NO2']:
        data[f'{col}_roll3'] = data[col].rolling(3, min_periods=1).mean()
        data[f'{col}_roll7'] = data[col].rolling(7, min_periods=1).mean()
    for col in ['PM2.5', 'PM10', 'AQI']:
        data[f'{col}_lag1'] = data[col].shift(1)
        data[f'{col}_lag3'] = data[col].shift(3)
    data['PM25_pct'] = data['PM2.5'].pct_change()
    data['PM10_pct'] = data['PM10'].pct_change()
    data['PM_ratio'] = data['PM2.5'] / (data['PM10'] + 1e-8)
    return data.ffill().bfill().replace([np.inf, -np.inf], 0).fillna(0)


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(os.path.join(path, "city_day.csv"))
    df = df[df['City'] == CITY].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['AQI'])

    data = df[POLLUTANTS + ['AQI']].reset_index(drop=True)
    data = data.interpolate(method='linear', limit_direction='both').ffill().bfill()

    train_end, val_end, fit_rows = split_points(len(data))

    # Outlier bounds from the training period only
    train_part = data.iloc[:fit_rows]
    q1, q3 = train_part.quantile(0.25), train_part.quantile(0.75)
    iqr = q3 - q1
    data = data.clip(q1 - 3 * iqr, q3 + 3 * iqr, axis=1)

    data = engineer_features(data)
    features = [c for c in data.columns if c != 'AQI']
    X = data[features].values
    y = data['AQI'].values.reshape(-1, 1)

    # Scalers fitted on the training period only
    scaler_X = RobustScaler().fit(X[:fit_rows])
    scaler_y = MinMaxScaler().fit(y[:fit_rows])
    X_seq, y_seq = create_sequences(scaler_X.transform(X), scaler_y.transform(y))

    return Dataset(
        X_train=X_seq[:train_end], y_train=y_seq[:train_end],
        X_val=X_seq[train_end:val_end], y_val=y_seq[train_end:val_end],
        X_test=X_seq[val_end:], y_test=y_seq[val_end:],
        scaler_X=scaler_X, scaler_y=scaler_y, features=features, data=data,
    )
