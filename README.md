# Air Quality Index Forecasting with LSTM, GRU and Transformer

Next-day **AQI forecasting for Delhi** from daily pollutant measurements (Air Quality Data in India, 2015–2020), comparing three deep learning architectures and their ensembles.

![Summary](results/07_summary_dashboard.png)

## Results (test set, last 15% of the series, 298 days)

| Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| LSTM | 0.874 | 43.18 | 31.32 | 17.6% |
| **GRU** | **0.899** | **38.60** | 28.37 | 15.9% |
| Transformer | 0.813 | 52.55 | 36.44 | 18.5% |
| Ensemble (mean) | 0.884 | 41.32 | 29.34 | 16.2% |
| Weighted ensemble | 0.895 | 39.44 | **28.10** | **15.8%** |

**Bootstrap 95% CI for R²:** GRU 0.873 – 0.921, LSTM 0.843 – 0.901, Transformer 0.778 – 0.846.

**Significance:** GRU's errors are significantly lower than both LSTM and Transformer under the Diebold-Mariano test (p = 0.043 and p < 0.001) and the Wilcoxon signed-rank test (p < 0.001 for both). Details in [`results/statistical_results.csv`](results/statistical_results.csv).

## Method

**Data:** daily measurements for Delhi, 12 pollutants (PM2.5, PM10, NO, NO₂, NOx, NH₃, CO, SO₂, O₃, benzene, toluene, xylene). Missing values are interpolated and outliers clipped at 3×IQR.

**Feature engineering (29 features):** 3- and 7-day rolling means of PM2.5, PM10, CO and NO₂; 1- and 3-day lags of PM2.5, PM10 and AQI; daily percentage change and the PM2.5 / PM10 ratio.

**Sequences:** 14-day windows predict the next day's AQI. The split is chronological (70 / 15 / 15), so the model never sees the future during training.

**No leakage from the test period:** outlier bounds and both scalers are fitted on the training period only and then applied to validation and test data (see [`data_prep.py`](data_prep.py)). The attention variant does the same for its KNN imputer, winsorization limits and feature selection.

**Models**

| Model | Architecture |
|---|---|
| LSTM | 2 × Bidirectional LSTM (128, 64) → LSTM (32) → Dense |
| GRU | 2 × Bidirectional GRU (128, 64) → GRU (32) → Dense |
| Transformer | Conv1D embedding → multi-head self-attention blocks → GRU → Dense |

Training: Adam (lr 5e-4), batch 16, up to 150 epochs with early stopping, L2 regularization and dropout 0.2–0.3.

## Figures

| | |
|---|---|
| ![Training curves](results/01_training_comparison.png) | ![Predictions](results/02_predictions_scatter.png) |
| ![Residuals](results/03_residuals_analysis.png) | ![Ensembles](results/05_ensemble_comparison.png) |

## Running

1. Download [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and put `city_day.csv` in `data/`.
2. Install and train:
   ```bash
   pip install -r requirements.txt
   python train_models.py              # LSTM, GRU, Transformer, ensembles, figures
   python statistical_analysis.py      # bootstrap CIs, Diebold-Mariano and Wilcoxon tests
   ```

`train_attention_models.py` is an extended version with attention layers on top of the recurrent models.

## Files

| | |
|---|---|
| `data_prep.py` | Loading, cleaning, feature engineering and leakage-free scaling (shared) |
| `train_models.py` | The three models, ensembles and all figures |
| `train_attention_models.py` | Variant with attention-augmented LSTM and GRU |
| `statistical_analysis.py` | Bootstrap CIs and statistical comparison of the saved models |
| `results/` | Metrics, test predictions and figures |
